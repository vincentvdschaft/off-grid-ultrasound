# region
import copy
import os
import re
import sys

import yaml

# pylint: disable=C0413
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from jaxus import (
    beamform_das,
    beamform_dmas,
    beamform_mv,
    find_t_peak,
    gcnr_compute_disk,
    gcnr_plot_disk_annulus,
    iterate_axes,
    load_hdf5,
    log,
    log_compress,
    plot_beamformed,
    use_dark_style,
    use_light_style,
)
from jaxus.utils import log
from skimage import exposure
from tqdm import tqdm

from infer import (
    cache_outputs,
    combine_waveform_samples,
    compute_fwhm,
    get_kernel_image,
    get_pixel_grid_from_lims,
    resolve_data_path,
)
from infer.goudarzi import admm_inverse_beamform_compounded

beamform_das = cache_outputs("temp/cache")(beamform_das)
beamform_mv = cache_outputs("temp/cache")(beamform_mv)
beamform_dmas = cache_outputs("temp/cache")(beamform_dmas)
admm_inverse_beamform_compounded = cache_outputs("temp/cache")(
    admm_inverse_beamform_compounded
)
get_kernel_image = cache_outputs("temp/cache")(get_kernel_image)
# compute_fwhm = cache_outputs("temp/cache")(compute_fwhm)

QUICK_MODE = False


class GCNRRingDiskMeasurement:
    def __init__(
        self,
        x,
        y,
        inner_radius,
        outer_radius_start,
        outer_radius_end,
        value=None,
        name=None,
    ):
        self.x = x
        self.y = y
        self.inner_radius = inner_radius
        self.outer_radius_start = outer_radius_start
        self.outer_radius_end = outer_radius_end
        self.value = value
        self.name = name

    def set_value(self, value):
        log.info(f"GCNR value: {log.yellow(value)}")
        self.value = value

    @property
    def position(self):
        return self.x, self.y

    @property
    def gcnr(self):
        return self.value


@dataclass
class FWHMMeasurement:
    def __init__(self, x, y, plot_dot=False, plot_id=False, curve=None):
        self.x = x
        self.y = y
        self.fwhm_x = None
        self.fwhm_y = None
        self.plot_dot = plot_dot
        self.plot_id = plot_id
        self.curve = curve

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        raise KeyError(f"{key} not found in FWHMMeasurement")


class Method:
    """A method is a collection of runs with the same settings that are used to compute an image."""

    def __init__(self, name, runs):
        self.name = name
        self.runs = runs
        self.standard_deviation = None
        self.mean_image = None

    def __repr__(self):
        num = len(self.runs)
        if num == 1:
            return f"Method({self.name}, 1 run)"
        else:
            return f"Method({self.name}, {num} runs)"

    def get_all_images(
        self, apply_log_compression=True, histogram_match_image=None, stack=False
    ):
        images = []
        for run in self.runs:
            image = run.get_image(
                apply_log_compression=apply_log_compression,
                histogram_match_image=histogram_match_image,
            )
            images.append(image)
        if stack:
            return np.stack(images, axis=0)
        return images

    @property
    def fwhm_results(self):
        return list(chain(*[run.fwhm_results for run in self.runs]))

    @property
    def gcnr_results(self):
        return list(chain(*[run.gcnr_results for run in self.runs]))


def get_mean_method(method):
    images = method.get_all_images(apply_log_compression=False, stack=True)
    run = method.runs[0]
    run._image = np.mean(images, axis=0)
    return Method("mean_" + method.name, [run])


class Run:
    """A run is a single run with a specific set of parameters that are used to compute an image."""

    def __init__(self, init_dict):
        self._image = None
        self._images_processed = {}
        self.param_dict = init_dict
        self.fwhm_results = []
        self.gcnr_results = []
        self.contrast_results = []

    def compute_image(self):
        raise NotImplementedError

    def get_image(self, apply_log_compression=True, histogram_match_image=None):

        if self._image is None:
            self.compute_image()

        # key = (apply_log_compression, histogram_match_image is not None)
        # if key in self._images_processed:
        #     return self._images_processed[key]

        if apply_log_compression:
            image = log_compress(self._image, normalize=True)
            # image = np.clip(image, -60, 0)
        else:
            image = self._image

        if histogram_match_image is not None:
            image = exposure.match_histograms(image, histogram_match_image)

        # self._images_processed[key] = image
        return image


class InferRun(Run):
    def compute_image(self):
        state_dict = self.param_dict["state_dict"]
        config_dict = self.param_dict["config_dict"]
        target_region_m = np.array(config_dict["target_region_mm"]) * 1e-3
        xlims = (target_region_m[0], target_region_m[1])
        zlims = (target_region_m[2], target_region_m[3])

        image, grid_inverse = get_kernel_image(
            scatterer_pos_m=state_dict["scat_pos_m"],
            scatterer_amplitudes=np.abs(state_dict["scat_amp"]),
            xlim=xlims,
            zlim=zlims,
            pixel_size=self.param_dict["kernel_image_pixel_size_mm"] * 1e-3,
            radius=self.param_dict["kernel_image_radius_mm"] * 1e-3,
            falloff_power=self.param_dict["falloff_power"],
        )
        self._image = image
        self.param_dict["pixel_grid"] = grid_inverse
        return self._image

    def __repr__(self):
        return "InferRun"


class BeamformerRun(Run):
    def __init__(self, init_dict):
        super().__init__(init_dict)
        data_dict = self.param_dict["data_dict"]
        wavelength = data_dict["sound_speed"] / data_dict["center_frequency"]
        target_region_m = np.array(self.param_dict["target_region_m"])
        xlims = (target_region_m[0], target_region_m[1])
        zlims = (target_region_m[2], target_region_m[3])
        pixel_grid = get_pixel_grid_from_lims((xlims, zlims), pixel_size=wavelength / 4)
        self.param_dict["pixel_grid"] = pixel_grid


class DASRun(BeamformerRun):

    def compute_image(self):
        data_dict = self.param_dict["data_dict"]
        param_dict = self.param_dict
        n_tx = data_dict["raw_data"].shape[1]
        t_peak = find_t_peak(data_dict["waveform_samples_two_way"], 250e6) * np.ones(
            n_tx
        )
        log.debug(f"DAS sound speed: {data_dict['sound_speed']}")
        log.info("Performing DAS beamforming...")
        # Beamform with DAS and remove frame dimension
        self._image = beamform_das(
            rf_data=data_dict["raw_data"],
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            sampling_frequency=data_dict["sampling_frequency"],
            sound_speed=data_dict["sound_speed"],
            carrier_frequency=data_dict["center_frequency"],
            pixel_positions=param_dict["pixel_grid"].pixel_positions_flat,
            t_peak=t_peak,
            initial_times=data_dict["initial_times"],
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
            sound_speed_lens=param_dict["lens_sound_speed_mps"],
            lens_thickness=param_dict["lens_thickness_mm"] * 1e-3,
            f_number=param_dict["f_number"],
            iq_beamform=True,
            progress_bar=True,
        )[0].reshape(param_dict["pixel_grid"].shape_2d)

        print(f"f-number: {param_dict['f_number']}")
        return self._image

    def __repr__(self):
        return "DASRun"


class DMASRun(BeamformerRun):
    def compute_image(self):
        data_dict = self.param_dict["data_dict"]
        config_dict = self.param_dict
        n_tx = data_dict["raw_data"].shape[1]
        t_peak = find_t_peak(data_dict["waveform_samples_two_way"], 250e6) * np.ones(
            n_tx
        )
        log.info("Performing DMAS beamforming...")
        self._image = beamform_dmas(
            rf_data=data_dict["raw_data"],
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            sampling_frequency=data_dict["sampling_frequency"],
            sound_speed=data_dict["sound_speed"],
            carrier_frequency=data_dict["center_frequency"],
            pixel_positions=config_dict["pixel_grid"].pixel_positions_flat,
            t_peak=t_peak,
            initial_times=data_dict["initial_times"],
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
            sound_speed_lens=config_dict["lens_sound_speed_mps"],
            lens_thickness=config_dict["lens_thickness_mm"] * 1e-3,
            f_number=config_dict["f_number"],
            progress_bar=True,
        )[0].reshape(config_dict["pixel_grid"].shape_2d)
        return self._image

    def __repr__(self):
        return "DMASRun"


class MVRun(BeamformerRun):

    def compute_image(self):
        data_dict = self.param_dict["data_dict"]
        param_dict = self.param_dict
        n_tx = data_dict["raw_data"].shape[1]
        t_peak = find_t_peak(data_dict["waveform_samples_two_way"], 250e6) * np.ones(
            n_tx
        )
        log.info("Performing MV beamforming...")
        self._image = beamform_mv(
            rf_data=data_dict["raw_data"],
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            sampling_frequency=data_dict["sampling_frequency"],
            sound_speed=data_dict["sound_speed"],
            carrier_frequency=data_dict["center_frequency"],
            pixel_positions=param_dict["pixel_grid"].pixel_positions_flat,
            t_peak=t_peak,
            initial_times=data_dict["initial_times"],
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
            sound_speed_lens=param_dict["lens_sound_speed_mps"],
            lens_thickness=param_dict["lens_thickness_mm"] * 1e-3,
            f_number=param_dict["f_number"],
            iq_beamform=True,
            subaperture_size=param_dict["subaperture_size"],
            diagonal_loading=param_dict["diagonal_loading"],
            pixel_chunk_size=4096 * 4,
            progress_bar=True,
        )[0].reshape(param_dict["pixel_grid"].shape_2d)
        return self._image

    def __repr__(self):
        return "MVRun"


class REDRun(Run):
    def __init__(self, init_dict):
        super().__init__(init_dict)
        data_dict = self.param_dict["data_dict"]
        target_region_m = np.array(self.param_dict["target_region_m"])
        pitch = np.abs(
            data_dict["probe_geometry"][1, 0] - data_dict["probe_geometry"][0, 0]
        )
        x0, x1, z0, z1 = [float(x) for x in target_region_m]
        xlims, zlims = (x0, x1), (z0, z1)
        fs = data_dict["sampling_frequency"]
        c = data_dict["sound_speed"]
        dx = pitch
        dz = c / (2 * fs)
        pixel_grid_red = get_pixel_grid_from_lims(
            (xlims, zlims),
            pixel_size=(dx, dz),
        )
        self.param_dict["pixel_grid"] = pixel_grid_red

    def compute_image(self):
        data_dict = self.param_dict["data_dict"]
        config_dict = self.param_dict
        tx_waveform_indices = data_dict["tx_waveform_indices"]
        waveform_samples = [
            data_dict["waveform_samples_two_way"][idx] for idx in tx_waveform_indices
        ]
        n_tx = data_dict["raw_data"].shape[1]
        t_peak = find_t_peak(data_dict["waveform_samples_two_way"], 250e6) * np.ones(
            n_tx
        )
        log.info("Performing RED beamforming...")
        self._image = admm_inverse_beamform_compounded(
            data_dict["raw_data"][0, ..., 0],
            data_dict["probe_geometry"],
            data_dict["t0_delays"],
            data_dict["tx_apodizations"],
            data_dict["initial_times"],
            t_peak,
            data_dict["sound_speed"],
            waveform_samples,
            data_dict["sampling_frequency"],
            data_dict["center_frequency"],
            config_dict["pixel_grid"].pixel_positions,
            nlm_h_parameter=0.8,
            mu=2000,
            method="RED",
            epsilon=8e-3,
            f_number=config_dict["f_number"],
            chunk_size=1024,
        ).reshape(config_dict["pixel_grid"].shape_2d)
        return self._image

    def __repr__(self):
        return "REDRun"


class ImageEvaluation:
    def __init__(self):

        self.gcnr_points = None
        self.fwhm_points = None
        self.baseline_methods = None
        self.infer_methods = None
        self.target_region_m = None
        self.probe_geometry_m = None
        self.histogram_match_image = None

        self.gcnr_results = {}
        self.fwhm_results = {}

    @property
    def methods(self):
        return [*self.baseline_methods, *self.infer_methods]


def save_evaluation(evaluation, path):
    with open(path, "wb") as f:
        pickle.dump(evaluation, f)


def load_evaluation(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def collect_run_config_paths(eval_config):
    """Collects all run config paths from a directory or a list of paths."""

    path = eval_config["path"]

    if isinstance(path, (str, Path)):
        if Path(path).is_dir():
            paths = list(Path(path).rglob("run_config.yaml"))
        else:
            paths = [Path(path)]
    elif isinstance(path, list):
        paths = [Path(p) for p in path]
    else:
        raise ValueError("Invalid path type")

    infer_methods = []

    if len(paths) == 0:
        raise ValueError("No run config files found")

    # Handle the case where all config files are in a single directory
    parent_dir = paths[0].parent.parent
    if path == paths[0].parent:
        for p in paths:
            config_dict = yaml.safe_load(open(p, "r", encoding="utf-8"))
            state_dict = dict(np.load(p.parent / "state.npz", allow_pickle=True))
            run_name = config_dict["run_name"]
            init_dict = {
                "config_path": p,
                "config_dict": config_dict,
                "state_dict": state_dict,
            }
            general_dict = copy.deepcopy(eval_config["global_baseline_settings"])
            init_dict = update_dict(general_dict, init_dict)
            infer_run = InferRun(init_dict)
            infer_version = Method(run_name, [infer_run])
            infer_methods.append(infer_version)
    else:
        versions = {}
        for p in paths:
            config_dict = yaml.safe_load(open(p, "r", encoding="utf-8"))
            state_dict = dict(np.load(p.parent / "state.npz", allow_pickle=True))
            run_name = config_dict["run_name"]
            key = p.parent.parent
            init_dict = {
                "config_path": p,
                "config_dict": config_dict,
                "state_dict_path": p.parent / "state.npz",
                "state_dict": state_dict,
            }
            general_dict = copy.deepcopy(eval_config["global_baseline_settings"])
            init_dict = update_dict(general_dict, init_dict)
            new_run = InferRun(init_dict)
            if key not in versions:
                new_version = Method(run_name, [new_run])
                versions[key] = new_version
            else:
                versions[key].runs.append(new_run)
        infer_methods = list(versions.values())

    # Sort the methods by name
    infer_methods = sorted(infer_methods, key=lambda x: x.name)

    return infer_methods


def construct_image_evaluation_from_config(eval_config):
    if isinstance(eval_config, (str, Path)):
        with open(eval_config, "r", encoding="utf-8") as f:
            eval_config = yaml.safe_load(f)
    assert isinstance(eval_config, dict)

    evaluation = ImageEvaluation()

    run_baselines_on_all = eval_config["run_baselines_on_all"]

    # ==========================================================================
    # Parse the measurements
    # ==========================================================================
    gcnr_points = []
    all_gcnr_points = eval_config["gcnr_points"]
    if all_gcnr_points is None:
        all_gcnr_points = []

    for gcnr_point in all_gcnr_points:
        x, y = gcnr_point["position"]
        inner_radius = float(gcnr_point["inner_radius"])
        outer_radius_start = float(gcnr_point["outer_radius_start"])
        outer_radius_end = float(gcnr_point["outer_radius_end"])
        name = gcnr_point["name"] if "name" in gcnr_point else None
        gcnr_points.append(
            GCNRRingDiskMeasurement(
                x, y, inner_radius, outer_radius_start, outer_radius_end, name=name
            )
        )
    fwhm_points = []
    all_fwhm_points = eval_config["fwhm_points"]
    if all_fwhm_points is None:
        all_fwhm_points = []
    for fwhm_point in all_fwhm_points:
        x, y = fwhm_point
        fwhm_points.append(FWHMMeasurement(x, y))

    # Store the measurements in the evaluation object
    evaluation.gcnr_points = gcnr_points
    evaluation.fwhm_points = fwhm_points

    # ==========================================================================
    # Create the method objects
    # ==========================================================================
    infer_methods = collect_run_config_paths(eval_config)
    evaluation.infer_methods = infer_methods

    methods_to_run_baselines_against = infer_methods
    if not run_baselines_on_all:
        methods_to_run_baselines_against = [infer_methods[0]]

    print(methods_to_run_baselines_against)

    baseline_methods = []

    for baseline_nr, infer_method in enumerate(methods_to_run_baselines_against):

        # We get the first config dict, assuming that all infer runs have the same data path
        config_dict = infer_method.runs[0].param_dict["config_dict"]
        hdf5_path = resolve_data_path(config_dict["path"])

        # ==========================================================================
        # Load the data
        # ==========================================================================
        data_dict = load_hdf5(
            hdf5_path,
            frames=[
                config_dict["frames"],
            ],
            transmits=config_dict["transmits"],
            reduce_probe_to_2d=True,
        )

        target_region_m = np.array(config_dict["target_region_mm"]) * 1e-3

        evaluation.target_region_m = target_region_m
        evaluation.probe_geometry_m = data_dict["probe_geometry"]

        # ==========================================================================
        # Create baseline methods
        # ==========================================================================

        baseline_named_suffix = f"_{infer_method.name}" if run_baselines_on_all else ""
        print(baseline_named_suffix)

        for baseline_name in eval_config["baselines"]:
            if eval_config["baselines"][baseline_name] is None:
                continue
            assert isinstance(eval_config["baselines"][baseline_name], list)
            for baseline_dict in eval_config["baselines"][baseline_name]:
                if baseline_dict is None:
                    baseline_dict = {}
                baseline_config_dict = copy.deepcopy(
                    eval_config["global_baseline_settings"]
                )
                baseline_config_dict = update_dict(baseline_config_dict, baseline_dict)
                data_dict_copy = copy.deepcopy(data_dict)
                data_dict_copy["sound_speed"] = (
                    data_dict_copy["sound_speed"] + config_dict["sound_speed_offset"]
                )
                baseline_config_dict["data_dict"] = data_dict_copy
                baseline_config_dict["target_region_m"] = target_region_m
                if baseline_name == "DAS":
                    run = DASRun(baseline_config_dict)
                    method = Method(baseline_name + baseline_named_suffix, [run])
                elif baseline_name == "DMAS":
                    run = DMASRun(baseline_config_dict)
                    method = Method(baseline_name + baseline_named_suffix, [run])
                elif baseline_name == "MV":
                    run = MVRun(baseline_config_dict)
                    method = Method(baseline_name + baseline_named_suffix, [run])
                elif baseline_name == "RED":
                    run = REDRun(baseline_config_dict)
                    method = Method(baseline_name + baseline_named_suffix, [run])
                else:
                    raise ValueError(f"Invalid baseline method {baseline_name}")

                baseline_methods.append(method)

    evaluation.baseline_methods = baseline_methods
    evaluation.infer_methods = infer_methods

    return evaluation


def replace_scientific_notation(string):
    # Regular expression to match scientific notation (e.g., 1e-3, -2.5E6, etc.)
    pattern = r"[-+]?\d*\.?\d+[eE][-+]?\d+"

    # Function to replace each match with the float conversion
    def convert_to_float(match):
        return str(float(match.group()))

    # Replace all matches in the string with their float values
    return re.sub(pattern, convert_to_float, string)


def evaluate_result(eval_config):
    path = eval_config["path"]

    try:
        dynamic_range = eval_config["dynamic_range"]
    except KeyError:
        dynamic_range = 60
    dynamic_range = abs(dynamic_range)

    evaluation = construct_image_evaluation_from_config(eval_config)
    # mean_methods = []
    # print(evaluation.infer_methods)
    # for method in evaluation.infer_methods:
    #     mean_method = get_mean_method(method)
    #     mean_methods.append(mean_method)
    # evaluation.infer_methods.extend(mean_methods)
    for method in evaluation.methods:
        for run in method.runs:
            run.compute_image()

    histogram_match_image = (
        evaluation.methods[0].runs[0].get_image(apply_log_compression=True)
    )

    # ==========================================================================
    # Compute GCNR
    # ==========================================================================
    # region
    for method in tqdm(
        evaluation.methods, desc=f"{'Computing GCNR on method':<40}", colour="green"
    ):
        for run in tqdm(method.runs, desc=f"{'Run':<40}", leave=False, colour="yellow"):
            evaluation.gcnr_results[method] = []
            for gcnr_point in tqdm(
                evaluation.gcnr_points,
                desc=f"{'Point':<40}",
                leave=False,
                colour="blue",
            ):

                xlims = run.param_dict["pixel_grid"].extent_m[:2]
                zlims = run.param_dict["pixel_grid"].extent_m[2:]

                if QUICK_MODE:
                    gcnr_val = 0.0
                else:
                    image = run.get_image(
                        apply_log_compression=True,
                        histogram_match_image=histogram_match_image,
                    )
                    image = np.clip(image, -dynamic_range, 0)

                    gcnr_x, gcnr_y = gcnr_point.x, gcnr_point.y
                    if isinstance(run, InferRun):
                        try:
                            offset = run.param_dict["state_dict"][
                                "sound_speed_offset_mps"
                            ]
                        except KeyError:
                            log.warning("No sound speed offset found in state dict")
                            offset = 0.0
                        sound_speed = 1540 - offset
                        print(f"sound speed: {sound_speed}")
                        gcnr_x *= sound_speed / 1540
                        gcnr_y *= sound_speed / 1540
                        print(f"rate: {sound_speed / 1540}")
                    gcnr_point.x = gcnr_x
                    gcnr_point.y = gcnr_y

                    gcnr_val = gcnr_compute_disk(
                        image=image,
                        xlims_m=xlims,
                        zlims_m=zlims,
                        disk_pos_m=(gcnr_x, gcnr_y),
                        inner_radius_m=gcnr_point.inner_radius,
                        outer_radius_start_m=gcnr_point.outer_radius_start,
                        outer_radius_end_m=gcnr_point.outer_radius_end,
                    )
                # Store the computed value in the GCNR object
                result = copy.deepcopy(gcnr_point)
                result.value = gcnr_val
                run.gcnr_results.append(result)

    # endregion

    # ==========================================================================
    # Compute FWHM
    # ==========================================================================
    for method in tqdm(
        evaluation.methods,
        desc=f"{'Computing FWHM on method':<40}",
        colour="green",
    ):
        for run in tqdm(method.runs, desc=f"{'Run':<40}", leave=False, colour="yellow"):
            for fwhm_point in tqdm(
                evaluation.fwhm_points,
                desc=f"{'Point':<40}",
                leave=False,
                colour="blue",
            ):
                position = (fwhm_point.x, fwhm_point.y)
                if QUICK_MODE:
                    fwhm_x, fwhm_y, updated_point = (0.0, 0.0, position)
                else:
                    fwhm_x, fwhm_y = fwhm_point.fwhm_x, fwhm_point.fwhm_y
                    if isinstance(run, InferRun):
                        try:
                            offset = run.param_dict["state_dict"][
                                "sound_speed_offset_mps"
                            ]
                        except KeyError:
                            offset = 0.0
                        sound_speed = 1540 - offset
                        position = (
                            fwhm_point.x * sound_speed / 1540,
                            fwhm_point.y * sound_speed / 1540,
                        )
                        print(f"rate: {sound_speed / 1540}")

                    image = run.get_image(
                        apply_log_compression=True,
                        histogram_match_image=histogram_match_image,
                    )
                    fwhm_x, fwhm_y, updated_point = compute_fwhm(
                        image,
                        run.param_dict["pixel_grid"].extent_m,
                        position=position,
                        size=8e-3,
                        in_db=True,
                        return_updated_point=True,
                    )

                result = copy.deepcopy(fwhm_point)
                result.fwhm_x = fwhm_x
                result.fwhm_y = fwhm_y
                result.x = updated_point[0]
                result.y = updated_point[1]
                run.fwhm_results.append(result)

    # ==========================================================================
    # Compute image variance
    # ==========================================================================

    for method in evaluation.infer_methods:
        images = method.get_all_images(apply_log_compression=False)
        standard_deviation = np.std(np.stack(images), axis=0)
        method.standard_deviation = standard_deviation

    return evaluation


# ==============================================================================
# Plotting functions
# ==============================================================================
# def violin(methods, )
