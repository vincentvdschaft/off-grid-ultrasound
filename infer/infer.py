import dataclasses
import pickle
import shutil
import time
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from imagelib import Image
from jax import grad, jit, value_and_grad, vmap
from jaxus import (
    PixelGrid,
    beamform_das,
    find_t_peak,
    get_pixel_grid,
    log,
    log_compress,
    plot_beamformed,
    plot_beamformed_old,
    plot_rf,
    use_dark_style,
)
from tqdm import tqdm

from .simulate import OptVars  # forward_model_multi_sample,
from .simulate import (
    ForwardSettings,
    batched_forward_model,
    execute_batched,
    inverse_reparameterize_opt_vars,
    inverse_reparameterize_scat_amp,
    inverse_reparameterize_scat_pos,
    reparameterize_opt_vars,
    reparameterize_scat_amp,
    reparameterize_scat_pos,
)
from .utils import copy_repo  # plot_overview,; to_mm_int,
from .utils import (
    create_date_folder,
    create_unique_dir,
    create_week_folder,
    get_available_vram,
    get_kernel_image,
    get_source_dir,
    get_vfocus,
    info_writer,
    load_hdf5,
    make_read_only,
    parse_range,
    plot_solution_3d,
    prune_scatterers,
    resolve_data_path,
    save_figure,
    stamp_figure,
    zip_directory,
)
from .waveforms import combine_waveform_samples, get_sampled_multi

original_repr = jax.numpy.ndarray.__repr__


def custom_repr(self):
    return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"


jax.numpy.ndarray.__repr__ = custom_repr


class DataTracker:
    def __init__(self):
        self.data = {}
        self.steps = {}

    def add_data(self, key, value, step):
        if not key in self.data:
            self.data[key] = []
            self.steps[key] = []
        self.data[key].append(value)
        self.steps[key].append(step)

    def keys(self):
        assert np.all(self.data.keys() == self.data.keys())
        return self.data.keys()

    def get_data(self, key):
        try:
            return self.steps[key], self.data[key]
        except KeyError:
            log.warning(f"Key {key} not found in data tracker. Returning zeros.")
            return [np.array([0])], [np.array([0])]

    def dump(self, path: Path):
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            log.warning(
                f"DataTracker dump at path {path} failed with the following exception:\n{e}"
            )

    def load(self, path: Path):
        try:
            with open(path, "rb") as f:
                unpickled_object = pickle.load(f)
                self.data = unpickled_object.data
                self.steps = unpickled_object.steps
        except Exception as e:
            log.warning(
                f"DataTracker load at path {path} failed with the following exception:\n{e}"
            )


def sample_locations_from_image(intensity_map, num_samples=1):
    # Flatten the image to 1D and normalize
    flattened = intensity_map.flatten()
    flattened -= np.min(flattened)
    total_intensity = np.sum(flattened)
    normalized = flattened / total_intensity

    # Compute the CDF
    cdf = np.cumsum(normalized)
    # plt.plot(cdf)
    # plt.show()

    # Sample a random number and find corresponding index
    random_number = np.random.rand(num_samples)
    sampled_rows = np.zeros(num_samples, dtype=np.int32)
    sampled_cols = np.zeros(num_samples, dtype=np.int32)
    # samples = np.random.choice(range(len(flattened)), num_samples, p=normalized)
    for i in range(num_samples):
        idx = np.searchsorted(cdf, random_number[i])
        # idx = samples[i]

        # Convert the index back to 2D coordinates
        rows, cols = intensity_map.shape
        sampled_rows[i] = idx // cols
        sampled_cols[i] = idx % cols

    sampled_rows = np.clip(sampled_rows, 0, rows - 1)
    sampled_cols = np.clip(sampled_cols, 0, cols - 1)

    return sampled_rows, sampled_cols


def f(scat_pos, scat_amp, opt_vars):
    return jnp.sum(jnp.square(scat_amp - 3))


# @jit
def loss_value_and_grad(
    tx,
    ax,
    ch,
    scat_pos_opt,
    scat_amp_opt,
    forward_settings,
    opt_vars_opt,
    true_samples,
    lam=0.0,
    regularization_type="LP_norm",
    regularization_params={"order": 1},
):
    """Computes the gradient of the loss function with respect to the optimization variables."""
    # only compute gradients w.r.t. scat_pos if optimize_scatterer_positions is true
    chosen_argnums = (
        (3, 4, 6) if forward_settings.optimize_scatterer_positions else (4, 6)
    )
    return value_and_grad(forward_model_loss, argnums=chosen_argnums)(
        tx,
        ax,
        ch,
        scat_pos_opt,
        scat_amp_opt,
        forward_settings,
        opt_vars_opt,
        true_samples,
        lam,
        regularization_type,
        regularization_params,
    )
    # return grad(f, argnums=(0, 1, 2))(scat_pos_m, scat_amp, opt_vars)


def opt_vars_regularization(opt_vars_opt, lam=1e-4):
    opt_vars = reparameterize_opt_vars(opt_vars_opt)
    return lam * (
        jnp.mean(jnp.square((opt_vars.gain - 1) / 0.5))
        # + jnp.mean(jnp.square(opt_vars.sound_speed_offset_mps / 100))
        # + jnp.mean(jnp.square(opt_vars.initial_times_shift_s) / 0.3e-6)
        + jnp.mean(jnp.square(opt_vars.angle_scaling - 1) / 0.5)
    )


def streq(str_a, str_b):
    return str_a.upper() == str_b.upper()


def regularization_summary(
    regularization_weight, regularization_type, regularization_params
):
    if regularization_weight == 0:
        return log.red("disabled")
    else:
        return (
            f"{log.green(regularization_type)} with {log.yellow(regularization_params)}"
        )


REGULARIZATION_PARAMS = {
    "LP_norm": {"order": float},
    "cauchy": {"gamma": float},
    "cbrt": {},
    "cauchy_implicit_symlog": {"symlog_epsilon": float, "cauchy_width": float},
    "implicit_symlog": {"symlog_epsilon": float},
}


def validate_regularization_params(regularization_type, regularization_params):
    if regularization_params is None:
        return

    def validate_individual_param(param_key, param_val):
        assert (
            regularization_type in REGULARIZATION_PARAMS.keys()
        ), f"regularization_params for regularization_type={regularization_type} should contain parameter {param_key}"
        assert isinstance(
            param_val, REGULARIZATION_PARAMS[regularization_type][param_key]
        ), f"Parameter {param_key} should be of type {REGULARIZATION_PARAMS[regularization_type][param_key]}"

    [
        validate_individual_param(param_key, param_val)
        for param_key, param_val in regularization_params.items()
    ]


def scat_amp_regularization(scat_amp, regularization_type, regularization_params):
    if streq(regularization_type, "LP_norm"):
        order = regularization_params["order"]
        return jnp.linalg.norm(scat_amp, ord=order) / scat_amp.shape[0]
    elif streq(regularization_type, "cauchy"):
        gamma = regularization_params["gamma"]
        return jnp.sum(jnp.log(1 + (scat_amp / gamma) ** 2)) / scat_amp.shape[0]
    elif streq(regularization_type, "cbrt"):
        return jnp.sum(jnp.abs(jnp.cbrt(scat_amp))) / scat_amp.shape[0]
    elif streq(regularization_type, "implicit_symlog"):
        symlog_epsilon = regularization_params["symlog_epsilon"]
        return jnp.sum(jnp.log(jnp.abs(scat_amp) + symlog_epsilon)) / scat_amp.shape[0]
    # elif streq(regularization_type, "implicit_symlog"):
    #     # this regularizer gives the penalty term for x that comes from using
    #     # a Cauchy prior on z = symlog(x)
    #     symlog_epsilon = regularization_params["symlog_epsilon"]
    #     cauchy_width = regularization_params["cauchy_width"]
    #     abs_sum = jnp.abs(scat_amp) + jnp.abs(symlog_epsilon)
    #     abs_div = jnp.abs(scat_amp / symlog_epsilon)
    #     # numerator = scat_amp * symlog_epsilon**2 * jnp.sign(scat_amp)
    #     numerator = scat_amp * jnp.sign(scat_amp)
    #     # denominator = jnp.pi * symlog_epsilon**2 * jnp.abs(scat_amp) * (abs_sum) * (jnp.sign(scat_amp)**2 * jnp.log(abs_sum / jnp.abs(symlog_epsilon))**2 + 1)
    #     denominator = jnp.pi * symlog_epsilon**2 * abs_div * (abs_div + 1) * cauchy_width * (jnp.log(abs_sum / jnp.abs(symlog_epsilon)) + cauchy_width**2/cauchy_width**2)
    #     # this because ~identical to the vanilla symlog if we subtract 1.
    #     constant_translation = 1
    #     return jnp.sum(-jnp.log(numerator/denominator) - constant_translation) / scat_amp.shape[0]

    else:
        raise UserWarning(
            f"Please specify a regularization type in {REGULARIZATION_PARAMS.keys()}"
        )


# @jit
def forward_model_loss(
    tx,
    ax,
    ch,
    scat_pos_opt,
    scat_amp_opt,
    forward_settings,
    opt_vars_opt,
    true_samples,
    lam=1e-3,
    regularization_type="LP_norm",
    regularization_params={"order": 1},
):
    """Computes the MSE loss between the true and predicted samples."""

    # Compute the predicted samples
    predicted_samples = batched_forward_model(
        tx, ax, ch, scat_pos_opt, scat_amp_opt, forward_settings, opt_vars_opt
    )

    # Compute the MSE loss
    scat_amp = reparameterize_scat_amp(
        scat_amp_opt,
        forward_settings.scat_amp_reparameterization,
        forward_settings.symlog_epsilon,
    )
    # only require regularization to be specified in config in lam > 0
    scat_amp_penalty = (
        0
        if lam == 0
        else lam
        * scat_amp_regularization(scat_amp, regularization_type, regularization_params)
    )
    return (
        jnp.mean(jnp.square(true_samples - predicted_samples))
        + scat_amp_penalty
        + opt_vars_regularization(opt_vars_opt, lam=lam)
    )


def compute_loss_on_rf(
    true_rf_data_all,
    tx_all,
    ax_all,
    ch_all,
    scat_pos_opt,
    scat_amp_opt,
    forward_settings,
    opt_vars_opt,
    prediction_batch_size,
    progress_bar=True,
):
    log.info("Computing current RF data estimation")
    # int(np.log(vram_factor))

    while True:
        try:
            log.debug(f"Batch size: {log.yellow(prediction_batch_size)}")

            predicted_samples = predict_batched(
                tx=tx_all,
                ax=ax_all,
                ch=ch_all,
                scat_pos_opt=scat_pos_opt,
                scat_amp_opt=scat_amp_opt,
                forward_settings=forward_settings,
                opt_vars_opt=opt_vars_opt,
                batch_size=max(1, prediction_batch_size),
                progress_bar=progress_bar,
            )
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                log.warning("OOM error in RF prediction. Reducing batch size...")
                prediction_batch_size = prediction_batch_size // 2
            else:
                raise e
    loss = jnp.mean(jnp.square(true_rf_data_all - predicted_samples))
    return loss, predicted_samples


def predict_batched(
    tx,
    ax,
    ch,
    scat_pos_opt,
    scat_amp_opt,
    forward_settings,
    opt_vars_opt,
    batch_size=512,
    progress_bar=True,
):
    """Predicts the RF samples for a batch of indices."""
    return execute_batched(
        batched_forward_model,
        batch_size=batch_size,
        batched_kwarg_dict={
            "tx": tx,
            "ax": ax,
            "ch": ch,
        },
        kwarg_dict={
            "scat_pos_opt": scat_pos_opt,
            "scat_amp_opt": scat_amp_opt,
            "forward_settings": forward_settings,
            "opt_vars_opt": opt_vars_opt,
        },
        progress_bar=progress_bar,
    )


def get_probe_geometry_tx(probe_geometry_m, tx_apodizations, t0_delays_s):
    """Returns the probe geometry for the transmitters."""
    n_tx = tx_apodizations.shape[0]
    active_element_idx = [np.where(tx_apodizations[tx] > 0)[0] for tx in range(n_tx)]
    tx_apodizations_tx = [
        tx_apodizations[tx, tx_apodizations[tx] > 0] for tx in range(n_tx)
    ]
    t0_delays_tx_s = [t0_delays_s[tx, tx_apodizations[tx] > 0] for tx in range(n_tx)]

    return (
        jnp.stack(active_element_idx).astype(jnp.int32),
        jnp.stack(tx_apodizations_tx).astype(jnp.float32),
        jnp.stack(t0_delays_tx_s).astype(jnp.float32),
    )


def _interpret_config_to_scat_pos(config_dict, wavelength):
    """Interprets the configuration dictionary to get the initial scatterer positions."""
    initial_sampling_locations = config_dict["initial_sampling_locations"]
    # ==========================================================================
    # Try to load the initial state from a file
    # ==========================================================================
    init_file = config_dict["init_file"]
    if not config_dict["init_file"] is None:
        init_file = Path(init_file)
        try:
            init_file = resolve_data_path(init_file)
            state = np.load(init_file)
            scat_pos_m = jnp.array(state["scat_pos_m"])
            scat_amp = jnp.array(state["scat_amp"])
            target_region_m = jnp.array(state["target_region_m"])
            log.info(
                f"{scat_amp.shape[0]} scatterers were loaded from {log.yellow(init_file)}"
            )
            return scat_pos_m, scat_amp, target_region_m
        except FileNotFoundError:
            log.warning(
                f"Could not find the file {init_file}. Using settings from config to reinitialize the scatterers."
            )

    # ==========================================================================
    # Construct a grid from target_region_mm, grid_shape, and grid_spacing_wl
    # ==========================================================================
    assert (
        config_dict["target_region_mm"] is not None
        and (
            config_dict["grid_shape"] is not None
            or config_dict["grid_spacing_wl"] is not None
        )
    ) or (
        config_dict["grid_shape"] is not None and config_dict["grid_spacing_wl"] is None
    )

    grid_shape = config_dict["grid_shape"]

    if config_dict["grid_spacing_wl"] is not None:
        grid_spacing_m = np.array(config_dict["grid_spacing_wl"]) * wavelength

    # --------------------------------------------------------------------------
    # If the target region is provided, compute the grid shape and spacing
    # --------------------------------------------------------------------------
    target_region_m = config_dict["target_region_mm"]
    if target_region_m is not None:
        # Convert the target region to meters
        target_region_m = np.array(config_dict["target_region_mm"]) * 1e-3
        log.info(f"Target region [m]: {target_region_m}")

        # Compute the size of the imaging region in all dimensions
        dim_sizes = []
        for i in range(len(target_region_m) // 2):
            start = target_region_m[2 * i]
            end = target_region_m[2 * i + 1]
            dim_sizes.append(end - start)
        dim_sizes = np.array(dim_sizes)

        if grid_shape is None:
            log.info(
                "Computing grid shape and start points from target region and grid spacing."
            )
            grid_shape = []
            for dim_size, dim_spacing in zip(dim_sizes, grid_spacing_m):
                grid_shape.append(int(dim_size / dim_spacing) + 1)
        else:
            log.info(
                "Computing grid spacing and start points from target region and grid shape."
            )
            grid_spacing_m = []
            for dim_size, dim_shape in zip(dim_sizes, grid_shape):
                grid_spacing_m.append(dim_size / (dim_shape - 1))
        grid_spacing_m = np.array(grid_spacing_m)
        log.debug(f"Grid spacing [m]: {grid_spacing_m}")
        grid_start_points_m = target_region_m[::2]
        grid_center = [False] * len(grid_shape)
    # --------------------------------------------------------------------------
    # If the target_region is not provided, compute it from the grid shape and spacing
    # --------------------------------------------------------------------------
    else:
        log.info(
            "Computing target region from grid shape, grid spacing, and grid start points."
        )

        grid_start_points_m = np.array(config_dict["grid_start_points_mm"]) * 1e-5
        grid_center = config_dict["grid_center"]

    # ==========================================================================
    # Create the pixel grid
    # ==========================================================================
    pixel_grid = get_pixel_grid(
        shape=grid_shape,
        spacing=grid_spacing_m,
        startpoints=grid_start_points_m,
        center=grid_center,
    )
    target_region_m = pixel_grid.extent_m

    # in the "from_image" case positions are re-set later based on the DAS image
    if (
        initial_sampling_locations == "grid"
        or initial_sampling_locations == "from_image"
    ):
        scat_pos_m = pixel_grid.pixel_positions_flat
    elif initial_sampling_locations == "uniform_random":
        scat_pos_m = pixel_grid.uniformly_sample_points_within_extent()
    else:
        scat_pos_m = pixel_grid.pixel_positions_flat

    scat_amp = np.random.randn((pixel_grid.n_points)) * 1e-5
    return scat_pos_m, scat_amp, target_region_m


def _das_beamform_2d(
    initial_scat_pos_m,
    wavelength,
    rf_data,
    probe_geometry_m,
    t0_delays_s,
    initial_times_s,
    sampling_frequency,
    probe_center_frequency_hz,
    sound_speed_mps,
    waveform_samples,
    n_tx,
    tx_apodizations,
    n_ch,
    lens_thickness=1e-3,
    lens_sound_speed_mps=1540,
):
    vmin_x = np.min(initial_scat_pos_m[:, 0], axis=0)
    vmax_x = np.max(initial_scat_pos_m[:, 0], axis=0)
    vmin_z = np.min(initial_scat_pos_m[:, 1], axis=0)
    vmax_z = np.max(initial_scat_pos_m[:, 1], axis=0)
    dx = 0.25 * wavelength

    shape = [int((vmax_x - vmin_x) / dx), int((vmax_z - 1e-3) / dx)]
    spacing = [dx, dx]
    center = (True, False)
    beamform_pixel_grid = get_pixel_grid(
        shape=shape, spacing=spacing, startpoints=[0, 1e-3], center=center
    )
    im_das = beamform_das(
        rf_data=rf_data,
        pixel_positions=beamform_pixel_grid.pixel_positions_flat,
        probe_geometry=probe_geometry_m,
        t0_delays=t0_delays_s,
        initial_times=initial_times_s,
        sampling_frequency=sampling_frequency,
        carrier_frequency=probe_center_frequency_hz,
        sound_speed=sound_speed_mps,
        sound_speed_lens=lens_sound_speed_mps,
        lens_thickness=lens_thickness,
        t_peak=find_t_peak(waveform_samples) * np.ones(n_tx),
        tx_apodizations=tx_apodizations,
        rx_apodization=np.ones(n_ch),
        f_number=0.5,
        iq_beamform=True,
        pixel_chunk_size=1024 * 4,
    ).reshape(beamform_pixel_grid.shape_2d)
    return im_das, beamform_pixel_grid


def render_model(
    ax,
    target_region_m,
    probe_geometry_m,
    scat_pos_current,
    scat_amp_current,
    kernel_image_radius,
    kernel_image_pixel_size,
    include_axes=True,
    title=None,
):
    """
    Renders an image of the model in its current state, as specified by the scatterer
    positions and amplitudes.
    """
    kernel_image, kernel_image_grid = get_kernel_image(
        xlim=list(target_region_m[:2]),
        ylim=0.0,
        zlim=list(target_region_m[-2:]),
        scatterer_pos_m=scat_pos_current,
        scatterer_amplitudes=jnp.abs(scat_amp_current),
        radius=kernel_image_radius,
        falloff_power=2,
        pixel_size=kernel_image_pixel_size,
    )

    # TODO: Update to use imshow_centered
    plot_beamformed_old(
        ax=ax,
        image=log_compress(kernel_image, normalize=True),
        extent_m=kernel_image_grid.extent_m_zflipped,
        vmin=-60,
        vmax=0,
        cmap="gray",
        axis_in_mm=True,
        probe_geometry=probe_geometry_m,
        title=title,
        include_axes=include_axes,
    )

    return kernel_image, kernel_image_grid


def infer_from_file(config_dict, working_dir=None, seed=0, stamp=None):
    """Convenience function to load a file and run the infer function."""

    path = config_dict["path"]

    key = jax.random.PRNGKey(seed)

    try:
        path = resolve_data_path(path)
    except FileNotFoundError:
        log.error(f"Could not find the file {log.yellow(path)}.")
        return

    if (
        (
            config_dict["target_region_mm"] is not None
            and len(config_dict["target_region_mm"]) == 6
        )
        or (
            config_dict["grid_shape"] is not None
            and len(config_dict["grid_shape"]) == 3
        )
        or (
            config_dict["grid_spacing_wl"] is not None
            and len(config_dict["grid_spacing_wl"]) == 3
        )
    ):
        log.info(f"3D imaging {log.green('enabled')}.")
        in_3d = True
    else:
        log.info(f"3D imaging {log.red('disabled')}.")
        in_3d = False

    data_dict = load_hdf5(
        path,
        frames=parse_range(config_dict["frames"]),
        transmits=parse_range(config_dict["transmits"]),
        reduce_probe_to_2d=not in_3d,
    )
    wavelength = data_dict["sound_speed"] / data_dict["center_frequency"]

    initial_scat_pos_m, initial_scat_amp, target_region_m = (
        _interpret_config_to_scat_pos(config_dict, wavelength)
    )

    n_el = data_dict["probe_geometry"].shape[0]
    gain = np.ones(n_el)
    # gain[30] = 0
    # gain[31] = 0
    # gain[60] = 0

    try:
        run_name = config_dict["run_name"]
    except KeyError:
        run_name = "infer_run"

    infer(
        rf_data=data_dict["raw_data"] * gain[None, None, None, :, None],
        probe_geometry_m=data_dict["probe_geometry"],
        probe_center_frequency_hz=data_dict["center_frequency"],
        initial_scat_pos_m=initial_scat_pos_m,
        initial_scat_amp=initial_scat_amp,
        target_region_m=target_region_m,
        sound_speed_mps=data_dict["sound_speed"] + config_dict["sound_speed_offset"],
        sampling_frequency=data_dict["sampling_frequency"],
        element_width_m=data_dict["element_width"],
        tgc_gain_curve=data_dict["tgc_gain_curve"],
        waveform_samples=data_dict["waveform_samples_two_way"],
        t0_delays_s=data_dict["t0_delays"],
        initial_times_s=data_dict["initial_times"],
        tx_apodizations=data_dict["tx_apodizations"],
        tx_waveform_indices=data_dict["tx_waveform_indices"],
        forward_model_type=config_dict["forward_model_type"],
        run_name=run_name,
        polar_angles_rad=data_dict["polar_angles"],
        azimuthal_angles_rad=data_dict["azimuth_angles"],
        focus_distances_m=data_dict["focus_distances"],
        n_iterations=config_dict["n_iterations"],
        plot_interval=config_dict["plot_interval"],
        save_state_interval=config_dict.get("save_state_interval", 100),
        ax_min=config_dict["ax_min"],
        n_ax=config_dict["n_ax"],
        batch_size=config_dict["batch_size"],
        noise_standard_deviation=config_dict["noise_standard_deviation"],
        gradient_accumulation=config_dict["gradient_accumulation"],
        kernel_image_radius=config_dict["kernel_image_radius_mm"] * 1e-3,
        kernel_image_pixel_size=config_dict["kernel_image_pixel_size_mm"] * 1e-3,
        learning_rate=float(config_dict["learning_rate"]),
        n_grad_scat=config_dict["n_grad_scat"],
        scat_amp_reparameterization=config_dict["scat_amp_reparameterization"],
        yz_plane_xval=config_dict["yz_plane_xval_mm"] * 1e-3,
        xz_plane_yval=config_dict["xz_plane_yval_mm"] * 1e-3,
        initial_sampling_locations=config_dict["initial_sampling_locations"],
        # initial_scatterer_amplitudes=config_dict["initial_scatterer_amplitudes"],
        apply_lens_correction=config_dict["apply_lens_correction"],
        lens_thickness=config_dict["lens_thickness_mm"] * 1e-3,
        lens_sound_speed_mps=config_dict["lens_sound_speed_mps"],
        regularization_type=str(config_dict["regularization_type"]),
        regularization_params=dict(config_dict["regularization_params"] or {}),
        regularization_weight=float(config_dict["regularization_weight"]),
        working_dir=working_dir,
        key=key,
        seed=seed,
        optimize_scatterer_positions=config_dict["optimize_scatterer_positions"],
        progress_bars=config_dict["progress_bars"],
        symlog_epsilon=config_dict.get("symlog_epsilon", 0.01),
        enable_wavelength_scaling=config_dict["enable_wavelength_scaling"],
        stamp=stamp,
    )


# Define the Lp norm function
def lp_norm(x, p):
    return jnp.sum(jnp.abs(x) ** p) ** (1 / p)


def infer(
    rf_data: np.array,
    probe_geometry_m: np.array,
    probe_center_frequency_hz,
    initial_scat_pos_m,
    initial_scat_amp,
    target_region_m,
    sound_speed_mps,
    sampling_frequency,
    element_width_m,
    tgc_gain_curve,
    waveform_samples,
    t0_delays_s,
    initial_times_s,
    tx_apodizations,
    tx_waveform_indices,
    lens_thickness=1e-3,
    lens_sound_speed_mps=1000,
    regularization_weight=0,
    regularization_type="LP_norm",
    regularization_params={"order": 1},
    apply_lens_correction=False,
    noise_standard_deviation=0.0,
    focus_distances_m=None,
    polar_angles_rad=None,
    azimuthal_angles_rad=None,
    forward_model_type="wavefront_only_general",
    run_name="infer_run",
    n_iterations=10000,
    plot_interval=1000,
    save_state_interval=2500,
    batch_size=128,
    gradient_accumulation=1,
    learning_rate=1e-3,
    ax_min=0,
    n_ax=-1,
    n_grad_scat=2048,
    kernel_image_radius=0.5e-3,
    kernel_image_pixel_size=0.1e-3,
    scat_amp_reparameterization=False,
    yz_plane_xval=0.0,
    xz_plane_yval=0.0,
    initial_sampling_locations="grid",
    # initial_scatterer_amplitudes="fixed",
    working_dir=None,
    key=None,
    seed=0,
    optimize_scatterer_positions=True,
    progress_bars=True,
    symlog_epsilon=0.01,
    enable_wavelength_scaling=True,
    stamp=None,
):
    """INFER"""
    log.info(f"target_region_m: {target_region_m}")
    log.info(f"kernel_image_pixel_size: {kernel_image_pixel_size}")
    if regularization_weight > 0:
        validate_regularization_params(regularization_type, regularization_params)
    log.info(
        f"Regularization: {regularization_summary(regularization_weight, regularization_type, regularization_params)}"
    )
    log.info("INFER STARTED")
    np.random.seed(seed)
    if key is None:
        key = jax.random.PRNGKey(seed)

    # ==========================================================================
    # Input error checking
    # ==========================================================================
    # initial_scat_pos_m[0] = np.array([7e-3, 13e-3])
    assert np.all(
        isinstance(x, np.ndarray)
        for x in [
            rf_data,
            probe_geometry_m,
            initial_scat_pos_m,
            initial_scat_amp,
            waveform_samples,
            t0_delays_s,
            initial_times_s,
            tx_apodizations,
            tgc_gain_curve,
        ]
    )
    assert rf_data.ndim == 5
    assert probe_geometry_m.ndim == 2
    assert initial_scat_pos_m.ndim == 2 and initial_scat_pos_m.shape[1] in (2, 3)
    assert initial_sampling_locations in ["uniform_random", "from_image", "grid"]

    if np.any(initial_scat_amp < 0) and not scat_amp_reparameterization:
        log.warning("Initial scatterer amplitudes should be positive.")
        initial_scat_amp = np.abs(initial_scat_amp)

    n_ch = probe_geometry_m.shape[0]
    n_tx = tx_apodizations.shape[0]
    if n_ax == -1:
        n_ax = rf_data.shape[2] - ax_min
    else:
        if n_ax + ax_min > rf_data.shape[2]:
            log.warning(
                f"n_ax + ax_min ({n_ax + ax_min}) is greater than the number of samples in the RF data ({rf_data.shape[2]}). "
                f"Setting n_ax to {rf_data.shape[2] - ax_min}."
            )
        n_ax = min(n_ax, rf_data.shape[2] - ax_min)

    in_3d = True if initial_scat_pos_m.shape[1] == 3 else False

    # ==========================================================================
    #
    # ==========================================================================

    # else:
    #     reparameterize_scat_amp = reparameterize_scat_amp
    #     inverse_reparameterize_scat_amp = inverse_reparameterize_scat_amp

    vram_factor = get_available_vram()[0] / 4000

    # element_width_m = 30000 * element_width_m

    # ==========================================================================
    # Create data tracker
    # ==========================================================================
    data_tracker = DataTracker()

    # ==========================================================================
    # Create working directory
    # ==========================================================================
    if working_dir is None:
        working_dir = create_week_folder(Path("results"))
        working_dir = create_date_folder(working_dir)
        working_dir = create_unique_dir(
            parent_directory=working_dir, name=run_name, prepend_date=True
        )
    else:
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
    info_writer.set_path(working_dir / "info.yaml")

    wavelength = sound_speed_mps / probe_center_frequency_hz
    info_writer.write_info("wavelength", wavelength)
    if stamp is None:
        stamp = working_dir.stem
    info_writer.write_info("stamp", stamp)

    # ==========================================================================
    # Copy the source code to the working directory
    # ==========================================================================
    source_dir = get_source_dir()
    copy_repo(
        source_dir=source_dir,
        source_dirs_recursive=[Path(source_dir / "infer"), Path(source_dir / "config")],
        destination_dir=working_dir,
    )
    zip_directory(working_dir / "source_code", working_dir / "source_code.zip")
    # Remove the source code directory
    shutil.rmtree(working_dir / "source_code")

    # ==========================================================================
    # Beamform the data
    # ==========================================================================
    # region

    # TODO: beamforming is done including the first ax_min samples. This might not be the best approach.
    probe_geometry_m_2d = np.stack(
        [probe_geometry_m[:, 0], probe_geometry_m[:, -1]], axis=1
    )
    if in_3d:
        vmin_x = np.min(initial_scat_pos_m[:, 0])
        vmax_x = np.max(initial_scat_pos_m[:, 0])
        vmin_y = np.min(initial_scat_pos_m[:, 1])
        vmax_y = np.max(initial_scat_pos_m[:, 1])
        vmin_z = np.min(initial_scat_pos_m[:, -1])
        vmax_z = np.max(initial_scat_pos_m[:, -1])
        dx = 0.25 * wavelength
        spacing = [dx, dx, dx]
        shape_x = int((vmax_x - vmin_x) / dx)
        shape_x = max(1, shape_x)
        shape = [shape_x, 1, int((vmax_z - vmin_z) / dx)]
        center = (True, True, False)
        startpoints = [0, xz_plane_yval, 1e-3]

        beamform_pixel_grid_yz = get_pixel_grid(
            shape=shape, spacing=spacing, startpoints=startpoints, center=center
        )
        # try:
        im_das = beamform_das(
            rf_data=rf_data,
            pixel_positions=beamform_pixel_grid_yz.pixel_positions_flat,
            probe_geometry=probe_geometry_m,
            t0_delays=t0_delays_s,
            initial_times=initial_times_s,
            sampling_frequency=sampling_frequency,
            carrier_frequency=probe_center_frequency_hz,
            sound_speed=sound_speed_mps,
            lens_sound_speed_mps=1540,
            lens_thickness=1e-3,
            t_peak=find_t_peak(waveform_samples) * np.ones(n_tx),
            tx_apodizations=tx_apodizations,
            rx_apodization=np.ones(n_ch),
            f_number=0.5,
            iq_beamform=True,
            pixel_chunk_size=1024 * 4,
        ).reshape(beamform_pixel_grid_yz.shape_2d)
        # except:
        #     im_das = np.ones(pixel_grid.shape[:2])
        das_normval = np.max(np.abs(im_das))

        fig, ax = plt.subplots()
        plot_beamformed_old(
            ax=ax,
            image=log_compress(im_das / das_normval),
            vmin=-60,
            vmax=0,
            extent_m=beamform_pixel_grid_yz.extent_m_2d,
            probe_geometry=np.stack(
                [probe_geometry_m[:, 0], probe_geometry_m[:, -1]], axis=1
            ),
            title="DAS image",
        )
        plt.savefig(working_dir / "das_image_yz.png", bbox_inches="tight")
        plt.close()

        startpoints = [xz_plane_yval, 0, 1e-3]
        shape_y = int((vmax_y - vmin_y) / dx)
        shape_y = max(1, shape_y)
        shape = [1, shape_y, int((vmax_z - vmin_z) / dx)]

        beamform_pixel_grid_xz = get_pixel_grid(
            shape=shape, spacing=spacing, startpoints=startpoints, center=center
        )
        # try:
        im_das = beamform_das(
            rf_data=rf_data,
            pixel_positions=beamform_pixel_grid_xz.pixel_positions_flat,
            probe_geometry=probe_geometry_m,
            t0_delays=t0_delays_s,
            initial_times=initial_times_s,
            sampling_frequency=sampling_frequency,
            carrier_frequency=probe_center_frequency_hz,
            sound_speed=sound_speed_mps,
            lens_sound_speed_mps=1540,
            lens_thickness=1e-3,
            t_peak=find_t_peak(waveform_samples) * np.ones(n_tx),
            tx_apodizations=tx_apodizations,
            rx_apodization=np.ones(n_ch),
            f_number=0.5,
            iq_beamform=True,
            pixel_chunk_size=1024 * 4,
        ).reshape(beamform_pixel_grid_xz.shape_2d)
        das_normval = np.max(np.abs(im_das))

        fig, ax = plt.subplots()
        plot_beamformed_old(
            ax=ax,
            image=log_compress(im_das / das_normval),
            vmin=-60,
            vmax=0,
            extent_m=beamform_pixel_grid_xz.extent_m_2d,
            probe_geometry=np.stack(
                [probe_geometry_m[:, 1], probe_geometry_m[:, -1]], axis=1
            ),
            title="DAS image",
            xlabel_override="y [mm]",
        )
        plt.savefig(working_dir / "das_image_xz.png", bbox_inches="tight")
        plt.close()
    else:
        im_das, beamform_pixel_grid = _das_beamform_2d(
            initial_scat_pos_m,
            wavelength,
            rf_data,
            probe_geometry_m,
            t0_delays_s,
            initial_times_s,
            sampling_frequency,
            probe_center_frequency_hz,
            sound_speed_mps,
            waveform_samples,
            n_tx,
            tx_apodizations,
            n_ch,
            lens_sound_speed_mps=lens_sound_speed_mps,
            lens_thickness=lens_thickness,
        )
        das_normval = np.max(np.abs(im_das))

        fig, ax = plt.subplots()
        plot_beamformed_old(
            ax=ax,
            image=log_compress(im_das / das_normval),
            vmin=-60,
            vmax=0,
            extent_m=beamform_pixel_grid.extent_m_2d,
            probe_geometry=probe_geometry_m_2d,
            title="DAS image",
        )
        plt.savefig(working_dir / "das_image.png", bbox_inches="tight")
        plt.close()

    # exit()
    # endregion

    # ==========================================================================
    # Construct a function for the transmit waveform
    # ==========================================================================
    waveform_samples = combine_waveform_samples(waveform_samples)
    # waveform_fn = get_sampled_multi(waveform_samples, sampling_frequency=250e6)

    active_element_idx, tx_apodizations_tx, t0_delays_tx_s = get_probe_geometry_tx(
        probe_geometry_m, tx_apodizations, t0_delays_s
    )

    # ==========================================================================
    # DAS grid
    # ==========================================================================
    scat_pos_m = jnp.array(initial_scat_pos_m)
    scat_amp = jnp.array(initial_scat_amp)
    if not in_3d and (
        initial_sampling_locations
        == "from_image"
        # or initial_scatterer_amplitudes == "from_image"
    ):
        n_scat = initial_scat_amp.shape[0]
        log.debug(f"Sampling {n_scat} scatterers from the DAS image...")
        positions = np.zeros((n_scat, 2))
        intensity_map = np.clip(log_compress(im_das, normalize=True), -60, 0)
        rows, cols = sample_locations_from_image(intensity_map, n_scat)

        if initial_sampling_locations == "from_image":
            positions = beamform_pixel_grid.pixel_positions[rows, cols]
            scat_pos_m = jnp.array(positions)

        # if initial_scatterer_amplitudes == "from_image":
        #     scat_amp = intensity_map[rows, cols]

    # fig, ax = plt.subplots()
    # ax.scatter(scat_pos_m[:, 0], scat_pos_m[:, 1], s=0.1)
    # plt.savefig(working_dir / "scatterer_positions.png", bbox_inches="tight")

    # ==========================================================================
    # Compute virtual source positions
    # ==========================================================================

    if forward_model_type == "virtual_source":
        assert focus_distances_m is not None and polar_angles_rad is not None
        if in_3d:
            assert azimuthal_angles_rad is not None
            pos_vfocus_m = get_vfocus(
                focus_distances_m, polar_angles_rad, azimuthal_angles_rad
            )
        else:
            focus_distances_m = np.clip(focus_distances_m, -10, 10)
            pos_vfocus_m = get_vfocus(focus_distances_m, polar_angles_rad)
        initial_times_s += (
            np.min(
                np.linalg.norm(probe_geometry_m[None] - pos_vfocus_m[:, None], axis=-1),
                axis=1,
            )
            / sound_speed_mps
        )
    else:
        pos_vfocus_m = None

    depths = jnp.linalg.norm(scat_pos_m, axis=-1)
    sort_indices = jnp.argsort(depths)
    scat_pos_m = scat_pos_m[sort_indices]
    scat_amp = scat_amp[sort_indices]

    # Compute the slope and intercept for the depths
    slope, intercept = np.polyfit(
        np.arange(scat_pos_m.shape[0]), depths[sort_indices], 1
    )
    # plt.plot(np.arange(scat_pos_m.shape[0]), depths[sort_indices])
    # plt.plot(
    #     np.arange(scat_pos_m.shape[0]),
    #     intercept + slope * np.arange(scat_pos_m.shape[0]),
    # )
    # plt.show()
    # exit()

    if "interpolated" in forward_model_type:
        from infer import cache_outputs

        from .simulate import compute_rx_travel_texture, compute_tx_travel_texture

        compute_rx_travel_texture = cache_outputs("cache")(compute_rx_travel_texture)
        compute_tx_travel_texture = cache_outputs("cache")(compute_tx_travel_texture)

        log.info("Computing travel textures...")
        shape = (1024, 1024)
        target_region_tuple = (
            float(target_region_m[0]),
            float(target_region_m[1]),
            float(target_region_m[2]),
            float(target_region_m[3]),
        )

        rx_travel_texture = compute_rx_travel_texture(
            extent_m=target_region_tuple,
            shape=shape,
            sound_speed_mps=sound_speed_mps,
            lens_sound_speed_mps=lens_sound_speed_mps,
            lens_thickness_m=lens_thickness,
        )

        tx_travel_texture = jnp.stack(
            [
                compute_tx_travel_texture(
                    extent_m=target_region_tuple,
                    shape=shape,
                    sound_speed_mps=sound_speed_mps,
                    lens_sound_speed_mps=lens_sound_speed_mps,
                    lens_thickness_m=lens_thickness,
                    active_element_idx=active_element_idx[i],
                    t0_delays_tx_s=t0_delays_tx_s[i],
                )
                for i in range(n_tx)
            ]
        )
    else:
        rx_travel_texture = None
        tx_travel_texture = None

    # for i in range(n_tx):
    #     fig, ax = plt.subplots()
    #     ax.imshow(
    #         log_compress(tx_travel_texture[i], normalize=True),
    #         extent=target_region_m,
    #         origin="lower",
    #     )
    #     plt.savefig(working_dir / f"tx_travel_texture_{i}.png", bbox_inches="tight")
    #     plt.close()
    # fig, ax = plt.subplots()
    # ax.imshow(rx_travel_texture, extent=target_region_m)
    # plt.savefig(working_dir / "rx.png")
    # plt.close()

    # ==========================================================================
    # Initialize forward settings
    # -------------------------------------------------------------------------#
    # These are the settings that are passed to the forward model that do not
    # change during the optimization.
    # ==========================================================================
    forward_settings = ForwardSettings(
        probe_geometry_m=jnp.array(probe_geometry_m),
        t0_delays_tx_s=jnp.array(t0_delays_tx_s),
        active_element_idx=jnp.array(active_element_idx),
        initial_times_s=jnp.array(initial_times_s),
        sound_speed_mps=jnp.array(sound_speed_mps),
        tgc_gain_curve=jnp.array(tgc_gain_curve),
        sampling_frequency_hz=jnp.array(sampling_frequency),
        center_frequency_hz=jnp.array(probe_center_frequency_hz),
        tw_indices=jnp.array(tx_waveform_indices, dtype=jnp.int32),
        element_width_m=jnp.array(element_width_m),
        pos_vfocus_m=pos_vfocus_m,
        tx_apodizations_tx=jnp.array(tx_apodizations_tx),
        waveform_samples=jnp.array(waveform_samples),
        slope=slope,
        intercept=intercept,
        scat_amp_reparameterization=scat_amp_reparameterization,
        n_grad_scat=n_grad_scat,
        forward_model_type=forward_model_type,
        optimize_scatterer_positions=optimize_scatterer_positions,
        apply_lens_correction=apply_lens_correction,
        lens_sound_speed_mps=lens_sound_speed_mps,
        lens_thickness=lens_thickness,
        tx_travel_texture=tx_travel_texture,
        rx_travel_texture=rx_travel_texture,
        target_region_m=jnp.array(target_region_m),
        symlog_epsilon=symlog_epsilon,
        enable_wavelength_scaling=enable_wavelength_scaling,
    )
    forward_settings.add_hash()

    # ==========================================================================
    # Initialize the optimization variables
    # -------------------------------------------------------------------------#
    # These are the variables that are optimized during the optimization.
    # ==========================================================================
    opt_vars = OptVars(
        sound_speed_offset_mps=jnp.array(0.0),
        gain=jnp.ones(n_ch, dtype=jnp.float32),
        angle_scaling=jnp.array(1.0),
        initial_times_shift_s=jnp.array(0.0),
    )

    scat_pos_opt = inverse_reparameterize_scat_pos(
        scat_pos_m, forward_settings, opt_vars
    )
    # scat_amp = scat_amp.at[scat_amp.shape[0] // 4].set(1.0)
    scat_amp_opt = jnp.clip(
        inverse_reparameterize_scat_amp(
            scat_amp,
            forward_settings.scat_amp_reparameterization,
            forward_settings.symlog_epsilon,
        ),
        -5,
        None,
    )
    log.debug(f"Min scatterer amplitude: {jnp.min(scat_amp_opt)}")
    log.debug(f"Max scatterer amplitude: {jnp.max(scat_amp_opt)}")

    opt_vars_opt = inverse_reparameterize_opt_vars(opt_vars)

    # ==========================================================================
    # Create arrays of the transmit, receive, and channel indices
    # ==========================================================================
    tx_vals = jnp.arange(n_tx, dtype=jnp.int32)
    ax_vals = jnp.arange(n_ax, dtype=jnp.int32) + ax_min
    ch_vals = jnp.arange(n_ch, dtype=jnp.int32)

    rf_shape = (n_tx, n_ax, n_ch)

    tx_all, ax_all, ch_all = jnp.meshgrid(tx_vals, ax_vals, ch_vals, indexing="ij")
    tx_all, ax_all, ch_all = tx_all.flatten(), ax_all.flatten(), ch_all.flatten()

    # Slice out only the relevant part of the RF data to shape
    # (n_tx, n_ax, n_el)
    rf_data = jnp.array(rf_data[0, :, ax_min : ax_min + n_ax, :, 0])
    # Normalize the RF data
    normval = jnp.percentile(jnp.abs(rf_data), 99.99)
    normval = jnp.clip(normval, 1e-6, None)
    rf_data = rf_data / normval
    true_rf_data_all = rf_data.flatten()

    # from .simulate import batched_forward_model

    # bs = 16
    # ax = jnp.zeros(bs, dtype=int)
    # ch = jnp.zeros(bs, dtype=int)
    # tx = jnp.zeros(bs, dtype=int)
    # batched_forward_model(
    #     tx, ax, ch, scat_pos_opt, scat_amp_opt, forward_settings, opt_vars_opt
    # )

    # rf_data = predict_batched(
    #     tx=tx_all,
    #     ax=ax_all,
    #     ch=ch_all,
    #     scat_pos_opt=scat_pos_opt,
    #     scat_amp_opt=scat_amp_opt,
    #     forward_settings=forward_settings,
    #     opt_vars_opt=opt_vars_opt,
    # ).reshape(rf_shape)
    # true_rf_data_all = rf_data.flatten()

    # scat_amp_opt = jnp.array(np.ones_like(initial_scat_amp) * -4.0)

    rf_data_vmax = np.percentile(rf_data, 99.5)

    time_total = 0.0

    # ==========================================================================
    # Initialize scheduler and optimizer
    # ==========================================================================
    scheduler = optax.exponential_decay(
        init_value=learning_rate, transition_steps=n_iterations, decay_rate=0.1e-1
    )
    # scheduler = optax.linear_schedule(
    #     init_value=learning_rate, end_value=0.0, transition_steps=n_iterations
    # )

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip(max_delta=1e3),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    log.info(f"Gradient accumulation: {gradient_accumulation}")
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=gradient_accumulation)

    optimized_params = (
        (scat_pos_m, scat_amp, opt_vars)
        if forward_settings.optimize_scatterer_positions
        else (scat_amp, opt_vars)
    )
    optimizer_state = optimizer.init(params=optimized_params)

    # ==========================================================================
    # Define the update function
    # ==========================================================================
    @partial(jit, static_argnames="forward_settings")
    def update(optimizer_state, opt_vars, forward_settings, key, rf_data, step):
        """Update step for the optimizer."""
        key, key_tx, key_ax, key_ch = jax.random.split(key, num=4)

        # ----------------------------------------------------------------------
        # Sample indices for the batch
        # ----------------------------------batched_forward_model------------------------------------
        # Select a single transmit for the entire batch
        tx = jax.random.randint(key=key_tx, shape=(batch_size,), minval=0, maxval=n_tx)

        # ax_start = jax.random.randint(
        #     key=key_ax, shape=(), minval=ax_min, maxval=ax_min + n_ax - batch_size
        # )
        # ax = jnp.arange(batch_size, dtype=jnp.int32) + ax_start

        # # Sample ax values from the range [ax_min, ax_min + n_ax)
        ax = jax.random.randint(
            key=key_ax, shape=(batch_size,), minval=ax_min, maxval=ax_min + n_ax
        )

        # Sample channel values from the range [0, n_ch)
        ch = jax.random.randint(key=key_ch, shape=(batch_size,), minval=0, maxval=n_ch)

        # ----------------------------------------------------------------------
        # Get the corresponding rf data
        # ----------------------------------------------------------------------
        true_rf_data = rf_data[tx, ax - ax_min, ch].flatten()

        scat_pos_opt, scat_amp_opt, opt_vars_opt = opt_vars

        loss_values, grads = loss_value_and_grad(
            tx=tx,
            ax=ax,
            ch=ch,
            scat_pos_opt=scat_pos_opt,
            scat_amp_opt=scat_amp_opt,
            forward_settings=forward_settings,
            opt_vars_opt=opt_vars_opt,
            true_samples=true_rf_data,
            lam=regularization_weight,
            regularization_type=regularization_type,
            regularization_params=regularization_params,
        )

        grads[-1].sound_speed_offset_mps *= jnp.where(step < 250, 0.0, 1.0)

        updates, optimizer_state = optimizer.update(grads, optimizer_state)

        # Apply the updates
        if not forward_settings.optimize_scatterer_positions:
            new_opt_vars = optax.apply_updates((scat_amp_opt, opt_vars_opt), updates)
            new_opt_vars = (scat_pos_opt, *new_opt_vars)
        else:
            new_opt_vars = optax.apply_updates(
                (scat_pos_opt, scat_amp_opt, opt_vars_opt), updates
            )

        return new_opt_vars, optimizer_state, key, loss_values, grads

    # ==========================================================================
    # Main optimization loop
    # ==========================================================================
    prediction_batch_size = jnp.clip(
        256 // forward_settings.active_element_idx.shape[1], 64, None
    )

    last_loss = -1

    log.info("Starting main loop...")
    log.debug(
        f"scat_amp_min: {jnp.min(reparameterize_scat_amp(scat_amp_opt, forward_settings.scat_amp_reparameterization, forward_settings.symlog_epsilon))}"
    )

    if noise_standard_deviation > 0:
        log.warning("Adding noise to the scatterer positions is TURNED ON!")

    last_time_progress_reported = time.time()

    with tqdm(
        total=n_iterations, desc="Optimizing", unit="step", disable=not progress_bars
    ) as pbar:
        for n in range(n_iterations):
            if not progress_bars:
                if (
                    (n + 1) % 250 == 0
                    or n == 0
                    or time.time() - last_time_progress_reported > 5
                ):
                    last_time_progress_reported = time.time()
                    opt_vars = reparameterize_opt_vars(opt_vars_opt)
                    log.info(
                        f"Step {n + 1} sound_speed_offset: {opt_vars.sound_speed_offset_mps}"
                    )

            # if n % 100 == 99:

            key, ky_noise = jax.random.split(key)
            scat_pos_opt = scat_pos_opt + jax.random.normal(
                ky_noise, scat_pos_opt.shape
            ) * noise_standard_deviation * scheduler(n) / scheduler(0)

            # opt_vars_opt.sound_speed_offset_mps = (
            #     opt_vars_opt.sound_speed_offset_mps
            #     + jax.random.normal(ky_noise, opt_vars_opt.sound_speed_offset_mps.shape)
            #     * noise_standard_deviation
            #     * scheduler(n)
            #     / scheduler(0)
            # )

            # ---------------------------------------------------------------------
            # Make the update step
            # ---------------------------------------------------------------------
            for _ in range(gradient_accumulation):
                (
                    (scat_pos_opt, scat_amp_opt, opt_vars_opt),
                    optimizer_state,
                    key,
                    batch_loss,
                    grads,
                ) = update(
                    optimizer_state,
                    (scat_pos_opt, scat_amp_opt, opt_vars_opt),
                    forward_settings,
                    key,
                    rf_data,
                    n,
                )

            # Track some optimization variables
            data_tracker.add_data("batch_loss", batch_loss, n + 1)
            if forward_settings.optimize_scatterer_positions:
                scat_pos_grad, scat_amp_grad, opt_vars_grad = grads
                data_tracker.add_data(
                    "grad_scat_pos", jnp.linalg.norm(scat_pos_grad), n + 1
                )
                if n % 10 == 0 and False:
                    plt.clf()
                    plt.hist(scat_amp_grad, bins=100)
                    if Path("/infer").exists():
                        figpath = Path(
                            f"/infer/results/grads_implicit_symlog/scat_amp_grad_{n + 1}"
                        )
                    else:
                        figpath = Path(
                            "./results/grads_implicit_symlog/scat_amp_grad_{n+1}"
                        )
                    figpath.mkdir(parents=True, exist_ok=True)
                    plt.savefig(figpath)
            else:
                scat_amp_grad, opt_vars_grad = grads
            data_tracker.add_data(
                "grad_scat_amp", jnp.linalg.norm(scat_amp_grad), n + 1
            )
            [
                data_tracker.add_data(
                    f"grad_{key}",
                    jnp.linalg.norm(opt_vars_grad[key]),
                    n + 1,
                )
                for key in opt_vars.keys()
            ]

            # save state every K steps. Set save_state_interval to -1 to prevent state save.
            if save_state_interval > 0 and (n % 100 == 100 - 1):
                opt_vars = reparameterize_opt_vars(opt_vars_opt)
                data_tracker.add_data(
                    "sound_speed_offset_mps",
                    opt_vars.sound_speed_offset_mps.item(),
                    n + 1,
                )
                data_tracker.add_data(
                    "initial_times_shift_s",
                    opt_vars.initial_times_shift_s.item(),
                    n + 1,
                )
                data_tracker.add_data("gain", np.array(opt_vars.gain), n + 1)
                data_tracker.add_data(
                    "angle_scaling", opt_vars.angle_scaling.item(), n + 1
                )
            # save state every K steps. Set save_state_interval to -1 to prevent state save.
            rf_already_computed = False
            if save_state_interval > 0 and (
                n % save_state_interval == save_state_interval - 1
            ):
                loss, predicted_samples = compute_loss_on_rf(
                    true_rf_data_all,
                    tx_all,
                    ax_all,
                    ch_all,
                    scat_pos_opt,
                    scat_amp_opt,
                    forward_settings,
                    opt_vars_opt,
                    prediction_batch_size,
                    progress_bar=progress_bars,
                )
                rf_already_computed = True
                data_tracker.add_data("loss", loss, n + 1)
                save_state(
                    working_dir / f"state_at_n={n + 1}.npz",
                    scat_pos_m=scat_pos_current,
                    scat_amp=scat_amp_current,
                    target_region_m=target_region_m,
                    forward_settings=forward_settings,
                    opt_vars=opt_vars,
                )
                if not in_3d:
                    rf_residual = jnp.square(true_rf_data_all - predicted_samples)
                    rf_residual = rf_residual.reshape(1, n_tx, n_ax, n_ch, 1)
                    im_das, beamform_pixel_grid = _das_beamform_2d(
                        initial_scat_pos_m,
                        wavelength,
                        rf_residual,
                        probe_geometry_m,
                        t0_delays_s,
                        initial_times_s,
                        sampling_frequency,
                        probe_center_frequency_hz,
                        sound_speed_mps,
                        waveform_samples,
                        n_tx,
                        tx_apodizations,
                        n_ch,
                        lens_sound_speed_mps=lens_sound_speed_mps,
                        lens_thickness=lens_thickness,
                    )
                    np.savez(
                        working_dir / f"beamformed_residual_n={n + 1}.npz",
                        beamformed_residual=im_das,
                    )

            # Set the timer after the first update to not count the initial compilation time
            if n == 0:
                time_start = timer()

            # ----------------------------------------------------------------------
            # Plotting and logging
            # ----------------------------------------------------------------------
            if plot_interval > 0 and (
                (n + 1) % plot_interval == 0 or n == 0 or n == n_iterations - 1
            ):
                time_total += timer() - time_start
                time_start = -1
                if not rf_already_computed:
                    loss, predicted_samples = compute_loss_on_rf(
                        true_rf_data_all,
                        tx_all,
                        ax_all,
                        ch_all,
                        scat_pos_opt,
                        scat_amp_opt,
                        forward_settings,
                        opt_vars_opt,
                        prediction_batch_size,
                        progress_bar=progress_bars,
                    )

                log.info(f"Loss: {log.yellow(loss)}")
                loss_diff = loss - last_loss
                loss_diff_str = (
                    log.green(f"{loss_diff:.2e}")
                    if loss_diff < 0
                    else log.red(f"{loss_diff:.2e}")
                )
                if n > 0:
                    log.info(f"Difference in loss: {loss_diff_str}")
                last_loss = loss

                data_tracker.add_data("loss", loss.item(), n + 1)

                scat_amp_current = reparameterize_scat_amp(
                    scat_amp_opt,
                    forward_settings.scat_amp_reparameterization,
                    forward_settings.symlog_epsilon,
                )
                scat_amp_penalty = regularization_weight * scat_amp_regularization(
                    scat_amp_current, regularization_type, regularization_params
                )
                log.info(f"Scatterer amplitude penalty: {log.yellow(scat_amp_penalty)}")
                data_tracker.add_data("scat_amp_penalty", scat_amp_penalty, n + 1)

                fig, axes = plt.subplots(2, n_tx, figsize=(12, 6))
                axes = axes.reshape(2, n_tx)
                out_dir = Path(working_dir / "rf_data")
                for i in range(n_tx):
                    plot_rf(
                        axes[0, i],
                        rf_data[i],
                        vmin=-rf_data_vmax,
                        vmax=rf_data_vmax,
                        start_sample=ax_min,
                    )
                    plot_rf(
                        axes[1, i],
                        predicted_samples.reshape(rf_shape)[i],
                        vmin=-rf_data_vmax,
                        vmax=rf_data_vmax,
                        start_sample=ax_min,
                    )
                    Image(rf_data[i], extent=(0, n_ax, 0, n_ch)).save(
                        out_dir / f"rf_data_{str(i).zfill(2)}.hdf5"
                    )
                    Image(
                        predicted_samples.reshape(rf_shape)[i],
                        extent=(0, n_ax, 0, n_ch),
                    ).save(
                        out_dir
                        / f"predicted_rf_data_{str(i).zfill(2)}_{str(n + 1).zfill(6)}.hdf5"
                    )

                plt.tight_layout()
                stamp_figure(fig, stamp)
                # plt.show()
                plt.savefig(working_dir / f"step_{n + 1}.png", bbox_inches="tight")
                plt.close()

                # exit()

                scat_pos_current = reparameterize_scat_pos(
                    scat_pos_opt, forward_settings, opt_vars_opt
                )

                save_state(
                    working_dir / "state.npz",
                    scat_pos_m=scat_pos_current,
                    scat_amp=scat_amp_current,
                    target_region_m=target_region_m,
                    forward_settings=forward_settings,
                    opt_vars=opt_vars,
                )

                # opt_vars_current = inverse_reparameterize_opt_vars(opt_vars_opt)
                use_dark_style()
                if not in_3d:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    out_dir = Path(working_dir / "kernel_images")
                    out_dir.mkdir(exist_ok=True, parents=True)

                    kernel_image, kernel_image_grid = render_model(
                        ax,
                        target_region_m,
                        probe_geometry_m,
                        scat_pos_current,
                        scat_amp_current,
                        kernel_image_radius,
                        kernel_image_pixel_size,
                    )
                    Image(kernel_image, extent=target_region_m).save(
                        out_dir / f"solved_image_{str(n + 1).zfill(6)}.hdf5"
                    )
                    plt.tight_layout()
                    stamp_figure(fig, stamp)
                    im_dir = working_dir / "solved_image"
                    im_dir.mkdir(exist_ok=True, parents=True)
                    plt.savefig(
                        im_dir / f"solved_image{n + 1:06d}.png", bbox_inches="tight"
                    )
                    plt.close()

                    # plot_overview(
                    #     kernel_image=log_compress(kernel_image),
                    #     kernel_pixel_grid=kernel_image_grid,
                    #     das_image=log_compress(im_das / das_normval),
                    #     das_pixel_grid=beamform_pixel_grid,
                    #     opt_vars=reparameterize_opt_vars(opt_vars_opt),
                    #     forward_settings=forward_settings,
                    #     save_path=None,
                    # )

                elif False:
                    kernel_image_x, kernel_image_grid = get_kernel_image(
                        scatterer_pos_m=scat_pos_current,
                        scatterer_amplitudes=scat_amp_current,
                        pixel_size=kernel_image_pixel_size,
                        xlim=1e-3,
                        ylim=pixel_grid.ylim,
                        zlim=pixel_grid.zlim,
                        radius=kernel_image_radius,
                    )
                    extent = kernel_image_grid.extent_m_zflipped[2:]
                    # extent = [extent[2], extent[3], extent[4], extent[5]]

                    fig, ax = plt.subplots()
                    plot_beamformed_old(
                        ax=ax,
                        image=log_compress(kernel_image_x[0], normalize=True),
                        extent_m=extent,
                        vmin=-2,
                        vmax=0,
                        cmap="gray",
                        axis_in_mm=True,
                        # probe_geometry=probe_geometry_m,
                        title="x-slice",
                        xlabel="y [mm]",
                        ylabel="z [mm]",
                    )
                    plt.savefig(working_dir / "solved_image_x.png", bbox_inches="tight")
                    # plt.show()
                    plt.close()

                # Plot the data in 3D
                if in_3d:
                    fig = plt.figure()

                    if scat_amp_current.shape[0] > 200:
                        scat_pos_pruned, scat_amp_pruned = prune_scatterers(
                            scat_pos_current,
                            scat_amp_current,
                            threshold_fraction=0.00,
                            n_max=2000,
                        )
                    else:
                        scat_pos_pruned, scat_amp_pruned = scat_pos_current, None

                    ax = plot_solution_3d(
                        fig=fig,
                        scat_pos_m=scat_pos_pruned,
                        scat_amp=scat_amp_pruned,
                        probe_geometry_m=probe_geometry_m,
                        active_element_idx=active_element_idx,
                        tx_apodizations_tx=tx_apodizations_tx,
                    )

                    save_figure(working_dir / "scatterer_positions_fig_3d.pkl", fig)
                    plt.savefig(
                        working_dir / "scatterer_positions.png", bbox_inches="tight"
                    )
                # Plot the tracked data
                if n_iterations >= save_state_interval:
                    fig = plt.figure()
                    gs = fig.add_gridspec(5, 2)

                    th = np.linspace(-np.pi / 2, np.pi / 2, 100)
                    opt_vars = reparameterize_opt_vars(opt_vars_opt)
                    angle_scaling = opt_vars.angle_scaling
                    from .simulate import directivity

                    wavelength_lens = (
                        lens_sound_speed_mps / forward_settings.center_frequency_hz
                    )

                    rp = directivity(
                        th,
                        forward_settings.element_width_m / wavelength_lens,
                    )
                    ax_directivity = fig.add_subplot(gs[0, 1])
                    ax_directivity.plot(th, rp)
                    ax_directivity.set_ylabel("Directivity")
                    ax_directivity.set_xlabel("Angle [rad]")

                    ax_sound_speed_offset = fig.add_subplot(gs[0, 0])
                    ax_initial_times_shift = fig.add_subplot(gs[1, 0])
                    ax_loss = fig.add_subplot(gs[3, 0])
                    ax_gain = fig.add_subplot(gs[2, 0])
                    ax_angle_scaling = fig.add_subplot(gs[4, 0])

                    steps, sound_speed_offset_mps = data_tracker.get_data(
                        "sound_speed_offset_mps"
                    )
                    ax_sound_speed_offset.plot(steps, sound_speed_offset_mps)
                    ax_sound_speed_offset.set_ylabel("Sound speed\noffset [m/s]")
                    ax_sound_speed_offset.set_xlabel("Step")

                    step, initial_times_shift_s = data_tracker.get_data(
                        "initial_times_shift_s"
                    )
                    ax_initial_times_shift.plot(step, initial_times_shift_s)
                    ax_initial_times_shift.set_ylabel("Initial times\nshift [s]")
                    ax_initial_times_shift.set_xlabel("Step")

                    step, gain = data_tracker.get_data("gain")
                    gain = np.stack(gain, axis=1)
                    ax_gain.imshow(gain, aspect="auto", vmin=0, vmax=1)
                    ax_gain.set_ylabel("Gain")

                    step, loss = data_tracker.get_data("loss")

                    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

                    ax_loss.plot(step, loss, color=default_colors[1])
                    ax_loss.set_ylabel("Loss")
                    ax_loss.set_xlabel("Step")

                    step, angle_scaling = data_tracker.get_data("angle_scaling")
                    ax_angle_scaling.plot(step, angle_scaling)
                    ax_angle_scaling.set_ylabel("Angle scaling")
                    ax_angle_scaling.set_xlabel("Step")

                    plt.tight_layout()

                    plt.savefig(
                        working_dir / "optimization_variables.png", bbox_inches="tight"
                    )
                    plt.close()
            if time_start == -1:
                time_start = timer()

            pbar.update(1)

    time_total += timer() - time_start
    info_writer.write_info("Runtime", time_total)

    plt.close()

    data_tracker.dump(working_dir / "data_tracker.pkl")

    make_read_only(working_dir)


def save_state(path, scat_pos_m, scat_amp, target_region_m, forward_settings, opt_vars):
    """Saves the state of the optimization to a file."""
    path = Path(path)

    np.savez(
        path,
        scat_pos_m=scat_pos_m,
        scat_amp=scat_amp,
        target_region_m=target_region_m,
        probe_geometry_m=forward_settings.probe_geometry_m,
        initial_times_s=forward_settings.initial_times_s,
        tgc_gain_curve=forward_settings.tgc_gain_curve,
        sampling_frequency_hz=forward_settings.sampling_frequency_hz,
        tw_indices=forward_settings.tw_indices,
        sound_speed_mps=forward_settings.sound_speed_mps,
        t0_delays_tx_s=forward_settings.t0_delays_tx_s,
        active_element_idx=forward_settings.active_element_idx,
        tx_apodizations_tx=forward_settings.tx_apodizations_tx,
        element_width_m=forward_settings.element_width_m,
        sound_speed_offset_mps=opt_vars.sound_speed_offset_mps,
        gain=opt_vars.gain,
        angle_scaling=opt_vars.angle_scaling,
        initial_times_shift_s=opt_vars.initial_times_shift_s,
    )
