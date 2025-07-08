# region
import copy
import os
import sys

import yaml

# pylint: disable=C0413
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import pickle
import sys
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

from infer import (
    cache_outputs,
    combine_waveform_samples,
    compute_fwhm,
    get_kernel_image,
    get_pixel_grid_from_lims,
    plot_resolution,
    resolve_data_path,
)
from infer.goudarzi import admm_inverse_beamform_compounded


class GCNRRingDiskMeasurement:
    def __init__(
        self, x, y, inner_radius, outer_radius_start, outer_radius_end, value=None
    ):
        self.x = x
        self.y = y
        self.inner_radius = inner_radius
        self.outer_radius_start = outer_radius_start
        self.outer_radius_end = outer_radius_end
        self.value = value

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


class ImageEvaluation:
    def __init__(self, gcnr_points, fwhm_points, sweep_kernel_radius=False):
        self.gcnr_points = gcnr_points
        self.fwhm_points = fwhm_points
        self.sweep_kernel_radius = sweep_kernel_radius
        # This will contain the images
        self.images = {}
        self.images_before_log = {}
        self.pixel_grids = {}
        self.target_region_m = None
        # The methods that were used to generate the images
        self.methods = []

        self.target_region_m = None
        self.probe_geometry_m = None

        self.gcnr_results = {}
        self.fwhm_results = {}

        assert all(isinstance(p, GCNRRingDiskMeasurement) for p in gcnr_points)
        assert all(isinstance(p, FWHMMeasurement) for p in fwhm_points)


def save_evaluation(evaluation, path):
    with open(path, "wb") as f:
        pickle.dump(evaluation, f)


def load_evaluation(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# TODO: Why is this necessary?
RADIUS_SCALING = 1.0

import re


def replace_scientific_notation(string):
    # Regular expression to match scientific notation (e.g., 1e-3, -2.5E6, etc.)
    pattern = r"[-+]?\d*\.?\d+[eE][-+]?\d+"

    # Function to replace each match with the float conversion
    def convert_to_float(match):
        return str(float(match.group()))

    # Replace all matches in the string with their float values
    return re.sub(pattern, convert_to_float, string)


def evaluate_result(config_paths, methods, evaluation: ImageEvaluation):
    if isinstance(config_paths, (str, Path)) and Path(config_paths).is_dir():
        config_paths = list(config_paths.rglob("run_config.yaml"))
        config_paths.sort()
    if not isinstance(config_paths, list):
        config_paths = [config_paths]

    config_paths = [Path(p) for p in config_paths]
    config_dicts = {}
    state_dicts = {}

    for n, config_path in enumerate(config_paths):

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        run_name = config_dict["run_name"]
        run_name = replace_scientific_notation(run_name)
        if run_name in config_dicts:
            id = 1
            while f"{run_name}_{id}" in config_dicts:
                id += 1
            run_name = f"{run_name}_{id}"
        methods.append(run_name)

        config_dicts[run_name] = config_dict

        # ==========================================================================
        #
        # ==========================================================================
        fwhm_results = {}

        # ==========================================================================
        # Load values from the config
        # ==========================================================================
        # region
        if n == 0:
            target_region_m = np.array(config_dict["target_region_mm"]) * 1e-3

            hdf5_path = resolve_data_path(config_dict["path"])
            transmits = np.array(config_dict["transmits"])
            frame = np.array(config_dict["frames"])
            n_ax = config_dict["n_ax"] + config_dict["ax_min"]
            sound_speed_lens = config_dict["lens_sound_speed_mps"]
            lens_thickness = config_dict["lens_thickness_mm"] * 1e-3
            try:
                f_number = config_dict["f_number"]
            except KeyError:
                f_number = 1.5
            log.debug(f"f_number: {f_number}")
            # try:
            #     run_name = config_dict["run_name"]
            # except KeyError:
            #     run_name = "unknown"
            # run_name = replace_scientific_notation(run_name)

            print(f"Processing run {log.yellow(run_name)}")

            x0, x1, z0, z1 = [float(x) for x in target_region_m]
            xlims, zlims = (x0, x1), (z0, z1)

        # endregion
        # ======================================================================================
        # Load solve
        # ======================================================================================
        # region
        state_path = config_path.parent / "state.npz"

        try:
            state_dict = np.load(state_path, allow_pickle=True)
        except FileNotFoundError as e:
            log.error(
                f"Could not find state file {state_path}. Make sure to input the config path from the output directory of the run."
            )
            raise e

        state_dicts[run_name] = state_dict

    # endregion
    # ======================================================================================
    # Enable caching
    # ======================================================================================
    # region
    global beamform_das
    global beamform_mv
    global beamform_dmas
    global admm_inverse_beamform_compounded
    global get_kernel_image
    beamform_das = cache_outputs("temp/cache")(beamform_das)
    beamform_mv = cache_outputs("temp/cache")(beamform_mv)
    beamform_dmas = cache_outputs("temp/cache")(beamform_dmas)
    admm_inverse_beamform_compounded = cache_outputs("temp/cache")(
        admm_inverse_beamform_compounded
    )
    # get_kernel_image = cache_outputs("temp/cache")(get_kernel_image)
    # endregion

    # ======================================================================================
    # Generate inverse image
    # ======================================================================================
    # region

    data_dict = load_hdf5(
        hdf5_path,
        frames=[
            frame,
        ],
        transmits=transmits,
        reduce_probe_to_2d=True,
    )

    evaluation.target_region_m = target_region_m
    evaluation.probe_geometry_m = data_dict["probe_geometry"]

    raw_data = data_dict["raw_data"][:, :n_ax]

    n_tx = raw_data.shape[1]

    wavelength = data_dict["sound_speed"] / data_dict["center_frequency"]
    # endregion

    for method in methods:
        if method in ["DAS", "DMAS", "MV"]:
            # Get the first config dict
            config_dict = config_dicts[list(config_dicts.keys())[0]]
            state_dict = state_dicts[list(state_dicts.keys())[0]]
        else:
            config_dict = config_dicts[method]
            state_dict = state_dicts[method]

        sweep_config_dicts = [config_dict]
        sweep_methods = [method]
        if "INFER" in method and evaluation.sweep_kernel_radius:
            sweep_config_dicts = []
            sweep_methods = []
            for kernel_radius in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                # for kernel_radius in [0.1, 0.15, 0.2]:
                new_config_dict = copy.deepcopy(config_dict)
                new_config_dict["kernel_image_radius_mm"] = kernel_radius
                sweep_config_dicts.append(new_config_dict)
                sweep_methods.append(f"{method}_{kernel_radius}")

        for method, config_dict in zip(sweep_methods, sweep_config_dicts):
            image, pixel_grid = get_image(
                method,
                target_region_m,
                config_dict,
                data_dict,
                state_dict,
            )
            evaluation.images[method] = log_compress(image, normalize=True)
            evaluation.images_before_log[method] = image
            evaluation.pixel_grids[method] = pixel_grid
            evaluation.methods.append(method)

    # ==========================================================================
    # Histgram match
    # ==========================================================================
    refernce_method = evaluation.methods[0]
    reference_image = evaluation.images[refernce_method]

    for method in evaluation.methods:
        evaluation.images[method] = exposure.match_histograms(
            evaluation.images[method], reference_image
        )

    # ==========================================================================
    # Compute GCNR
    # ==========================================================================
    # region

    for method in evaluation.methods:
        evaluation.gcnr_results[method] = []
        for gcnr_point in evaluation.gcnr_points:
            gcnr_val = gcnr_compute_disk(
                image=evaluation.images[method],
                xlims_m=xlims,
                zlims_m=zlims,
                disk_pos_m=(gcnr_point.x, gcnr_point.y),
                inner_radius_m=gcnr_point.inner_radius,
                outer_radius_start_m=gcnr_point.outer_radius_start,
                outer_radius_end_m=gcnr_point.outer_radius_end,
            )
            # Store the computed value in the GCNR object
            result = copy.deepcopy(gcnr_point)
            result.value = gcnr_val
            evaluation.gcnr_results[method].append(result)

    # endregion

    # ==========================================================================
    # Compute FWHM
    # ==========================================================================
    for method in evaluation.methods:
        evaluation.fwhm_results[method] = []
        for i, fwhm_point in enumerate(evaluation.fwhm_points):
            position = (fwhm_point.x, fwhm_point.y)
            fwhm_x, fwhm_y, updated_point = compute_fwhm(
                np.array(evaluation.images[method]),
                evaluation.pixel_grids[method].extent_m,
                position=position,
                size=8e-3,
                in_db=True,
                return_updated_point=True,
            )

            # plot_resolution(
            #     np.array(evaluation.images[method]),
            #     evaluation.pixel_grids[method].extent_m,
            #     position=position,
            #     size=8e-3,
            #     in_db=True,
            # )
            # plt.savefig(f"/infer/results/plots/resolution_plots/{method}_pos={i}.png")
            # plt.close()

            result = copy.deepcopy(fwhm_point)
            result.fwhm_x = fwhm_x
            result.fwhm_y = fwhm_y
            result.x = updated_point[0]
            result.y = updated_point[1]
            evaluation.fwhm_results[method].append(result)

            log.debug(
                f"FWHM x: {log.yellow(fwhm_x*100)}, FWHM y: {log.yellow(fwhm_y*100)}"
            )

    # ==========================================================================
    # Compute image variance
    # ==========================================================================
    infer_methods = set(evaluation.methods) - set(["DAS", "DMAS", "MV"])
    images = []
    for method in infer_methods:
        image_clipped = np.clip(evaluation.images[method], -60, 0)
        images.append(image_clipped)
    variance = np.var(np.stack(images), axis=0)

    evaluation.image_variance = variance

    return evaluation


def get_image(
    method, target_region, config_dict, data_dict, state_dict, state_dict2=None
):
    """Applies a method to the data in the target region and returns the resulting image."""
    method = method.upper()
    log.info(f"Processing method {log.yellow(method)}")

    xlims, zlims = (float(target_region[0]), float(target_region[1])), (
        float(target_region[2]),
        float(target_region[3]),
    )

    for key in config_dict.keys():
        print(key)

    wavelength = data_dict["sound_speed"] / data_dict["center_frequency"]

    n_ax = config_dict["n_ax"] + config_dict["ax_min"]
    n_tx = data_dict["raw_data"].shape[1]
    raw_data = data_dict["raw_data"][:, :n_ax]

    if method in ["DAS", "DMAS", "MV"]:
        pixel_grid = get_pixel_grid_from_lims((xlims, zlims), pixel_size=wavelength / 4)
    else:
        method = "INFER"
        pitch = np.abs(
            data_dict["probe_geometry"][1, 0] - data_dict["probe_geometry"][0, 0]
        )

        fs = data_dict["sampling_frequency"]
        c = data_dict["sound_speed"]
        dx = pitch
        dz = c / (2 * fs)
        pixel_grid = get_pixel_grid_from_lims(
            (xlims, zlims),
            pixel_size=(dx, dz),
        )

    # endregion
    # ======================================================================================
    # Enable caching
    # ======================================================================================
    # region
    global beamform_das
    global beamform_mv
    global beamform_dmas
    global admm_inverse_beamform_compounded
    global get_kernel_image
    beamform_das = cache_outputs("temp/cache")(beamform_das)
    beamform_mv = cache_outputs("temp/cache")(beamform_mv)
    beamform_dmas = cache_outputs("temp/cache")(beamform_dmas)
    admm_inverse_beamform_compounded = cache_outputs("temp/cache")(
        admm_inverse_beamform_compounded
    )
    # get_kernel_image = cache_outputs("temp/cache")(get_kernel_image)
    # endregion

    t_peak = find_t_peak(data_dict["waveform_samples_two_way"], 250e6) * np.ones(n_tx)

    if method == "DAS":
        log.info("Performing DAS beamforming...")
        # Beamform with DAS and remove frame dimension
        image = beamform_das(
            rf_data=raw_data,
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            sampling_frequency=data_dict["sampling_frequency"],
            sound_speed=data_dict["sound_speed"],
            carrier_frequency=data_dict["center_frequency"],
            pixel_positions=pixel_grid.pixel_positions_flat,
            t_peak=t_peak,
            initial_times=data_dict["initial_times"],
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
            sound_speed_lens=config_dict["lens_sound_speed_mps"],
            lens_thickness=config_dict["lens_thickness_mm"] * 1e-3,
            f_number=config_dict["f_number"],
            iq_beamform=True,
            progress_bar=True,
        )[0]
    elif method == "INFER":
        image, grid_inverse = get_kernel_image(
            scatterer_pos_m=state_dict["scat_pos_m"],
            scatterer_amplitudes=np.abs(state_dict["scat_amp"]),
            xlim=xlims,
            zlim=zlims,
            pixel_size=config_dict["kernel_image_pixel_size_mm"] * 1e-3,
            radius=config_dict["kernel_image_radius_mm"] * 1e-3 * RADIUS_SCALING,
            falloff_power=2,
        )
        pixel_grid = grid_inverse
    elif method == "MV":
        image = beamform_mv(
            rf_data=raw_data,
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            sampling_frequency=data_dict["sampling_frequency"],
            sound_speed=data_dict["sound_speed"],
            carrier_frequency=data_dict["center_frequency"],
            pixel_positions=pixel_grid.pixel_positions_flat,
            t_peak=t_peak,
            initial_times=data_dict["initial_times"],
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
            iq_beamform=True,
            subaperture_size=30,
            diagonal_loading=0.000,
            pixel_chunk_size=4096 * 4,
            sound_speed_lens=config_dict["lens_sound_speed_mps"],
            lens_thickness=config_dict["lens_thickness_mm"] * 1e-3,
            f_number=config_dict["f_number"],
            progress_bar=True,
        )[0]
    elif method == "DMAS":
        image = beamform_dmas(
            rf_data=raw_data,
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            sampling_frequency=data_dict["sampling_frequency"],
            sound_speed=data_dict["sound_speed"],
            carrier_frequency=data_dict["center_frequency"],
            pixel_positions=pixel_grid.pixel_positions_flat,
            t_peak=t_peak,
            initial_times=data_dict["initial_times"],
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
            sound_speed_lens=config_dict["lens_sound_speed_mps"],
            lens_thickness=config_dict["lens_thickness_mm"] * 1e-3,
            f_number=config_dict["f_number"],
            # pixel_chunk_size=1024,
            progress_bar=True,
        )[0]
    elif method == "RED":
        tx_waveform_indices = data_dict["tx_waveform_indices"]
        waveform_samples = [
            data_dict["waveform_samples_two_way"][idx] for idx in tx_waveform_indices
        ]
        image = admm_inverse_beamform_compounded(
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
            pixel_grid.pixel_positions,
            nlm_h_parameter=0.8,
            mu=2000,
            method="RED",
            epsilon=8e-3,
            f_number=config_dict["f_number"],
            chunk_size=1024,
        )
    else:
        raise ValueError(f"Invalid method {method}")

    # image = log_compress(image, normalize=True)
    if not method == "INFER":
        image = image.reshape(pixel_grid.shape_2d)

    return image, pixel_grid


def get_run_configs(directory):
    config_paths = list(Path(directory).rglob("run_config.yaml"))
    return config_paths


if __name__ == "__main__":
    # pass
    evaluate_result(
        "/home/vincent/1-projects/infer/results/snellius-nwo/results/2024-week-42/10-17/8195382/week-42/2024-10-17/20241017_105354_infer_run/run_config.yaml",
        [
            "DAS",
        ],
        ImageEvaluation([], []),
    )
