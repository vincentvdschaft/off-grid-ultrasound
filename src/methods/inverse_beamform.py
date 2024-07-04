import os
import shutil
import socket
from datetime import datetime
from math import ceil
from pathlib import Path
import inspect
from typing import List, Tuple, Union

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import grad, jit, lax, random, vmap
from matplotlib.ticker import FuncFormatter
from scipy.signal import butter, filtfilt
from tqdm import trange

from jaxus import CartesianPixelGrid, beamform_das, log_compress, use_dark_style, log
from jaxus.utils.log import yellow
from src.utils import create_unique_dir, envelope_detect, find_phase


def inverse_beamform(
    hdf5_path: Union[str, Path],
    frame: int,
    selected_tx: np.array,
    n_ax: int,
    grid_dx_wl: float = 0.65,
    grid_dz_wl: float = 0.4,
    grid_shape=(256, 256),
    regularization=0,
    redistribute_interval=250,
    ax_min=100,
    z_start=1e-4,
    batch_size=256,
    learning_rate=1e-2,
    plot_interval=2000,
    n_steps=15000,
    gradient_accumulation=1,
    das_dynamic_range=-60,
    use_two_way_waveforms=True,
    wavefront_only=False,
    enable_directivity=True,
    enable_waveform_shaping=True,
    enable_global_t0_shifting=True,
    enable_tgc_compensation=True,
    enable_element_gain=True,
    enable_attenuation_spread=True,
    enable_attenuation_absorption=True,
    enable_sound_speed=True,
    enable_scatterer_redistribution=False,
    enable_lens_correction=True,
    enable_optimize_positions=True,
    run_name="run",
    step_update_interval=50,
    super_folder=None,
):
    """Solves the inverse problem of ultrasound.

    Parameters
    ----------
    hdf5_path : str or Path
        The path to the HDF5 file containing the data and parameters.
    frame : int
        The frame to use.
    selected_tx : np.array
        The selected transmits to include in the optimization.
    n_ax : int
        The number of axial samples to use.
    grid_dx_wl : float, optional
        The grid spacing in wavelengths in the x-direction.
    grid_dz_wl : float, optional
        The grid spacing in wavelengths in the z-direction.
    grid_shape : tuple, optional
        The shape of the grid.
    regularization : float, optional
        The regularization factor. Set to 0 to disable regularization.
    redistribute_interval : int, optional
        The interval at which to redistribute the scatterers.
    ax_min : int, optional
        The minimum axial index to use. The first sample that is included in the
        optimization is ax_min. From there n_ax samples are included.
        [ax_min, ax_min + n_ax-1].
    z_start : float, optional
        The starting depth of the grid.
    batch_size : int, optional
        The number of samples to include in each batch. If gradient_accumulation is
        larger than 1, the effective batch size becomes
        batch_size * gradient_accumulation.
    learning_rate : float, optional
        The learning rate for the optimizer.
    plot_interval : int, optional
        The number of steps after which to simulate the RF data and plot the result.
    n_steps : int, optional
        The number of steps to run the optimization for.
    gradient_accumulation : int, optional
        The number of gradients to accumulate before updating the parameters. The
        effective batch size thus becomes batch_size * gradient_accumulation.
    das_dynamic_range : float, optional
        The dynamic range to use for the DAS beamforming. This is the lower limit of
        the dynamic range in dB. Images will be plotted in the range
        [das_dynamic_range, 0].
    use_two_way_waveforms : bool, optional
        Whether to use the two-way waveforms. These are the waveforms that after
        filtering with the transducer bandwidth twice. If False, the one-way waveforms
        are used.
    wavefront_only : bool, optional
        Whether to only consider the first wavefront that arrives. This is useful for
        reducing the computational and memory requirements of the optimization.
    enable_directivity : bool, optional
        Whether to enable element directivity fitting in the model. Used in the
        ablation study.
    enable_waveform_shaping : bool, optional
        Whether to enable waveform shaping to model frequency dependent attenuation.
        Used in the ablation study.
    enable_global_t0_shifting : bool, optional
        Whether to enable global t0 shifting. Used in the ablation study.
    enable_tgc_compensation : bool, optional
        Whether to enable the modelling of TGC compensation. Used in the ablation study.
    enable_element_gain : bool, optional
        Whether to enable element gain fitting. Used in the ablation study.
    enable_attenuation_spread : bool, optional
        Whether to enable spread attenuation. Used in the ablation study.
    enable_attenuation_absorption : bool, optional
        Whether to enable absorption attenuation. Used in the ablation study.
    enable_sound_speed : bool, optional
        Whether to enable sound speed fitting. Used in the ablation study.
    enable_scatterer_redistribution : bool, optional
        Whether to enable scatterer redistribution. Used in the ablation study.
    enable_lens_correction : bool, optional
        Whether to enable lens correction. Used in the ablation study.
    enable_optimize_positions : bool, optional
        Whether to optimize the scatterer positions.
    run_name : str, optional
        The name of the run. This is used to create a unique directory for the results.
    step_update_interval : int, optional
        The interval at which to print the step number.
    super_folder : str, optional
        The super folder to save the results in. If None, the results are saved in the
        current working directory.

    Returns
    -------
    Path
        The path to the directory containing the results.
    """

    # ==================================================================================
    # Parse input
    # ==================================================================================
    grid_shape = np.array(grid_shape, dtype=np.int32)
    use_dark_style()

    # ==================================================================================
    # Load data from file
    # ==================================================================================
    # region
    with h5py.File(
        hdf5_path,
        "r",
    ) as f:
        # ------------------------------------------------------------------------------
        # Load the RF data
        # ------------------------------------------------------------------------------
        # Get the shape of the raw data
        shape = f["data"]["raw_data"].shape

        # Extract the dimensions
        n_frames, _, n_ax_data, n_el, _ = shape
        n_tx = selected_tx.shape[0]

        # Ensure that the frame and n_ax are within the bounds of the data
        frame = np.min([frame, n_frames - 1])
        n_ax = np.min([n_ax, n_ax_data])

        # Read the RF data
        rf_data = f["data"]["raw_data"][frame, selected_tx, :n_ax, :, 0]

        # ------------------------------------------------------------------------------
        # Load the parameters
        # ------------------------------------------------------------------------------
        probe_center_frequency = f["scan"]["center_frequency"][()]
        tw_center_frequency = f["scan"]["center_frequency"][()] * np.ones((n_tx,))
        sampling_frequency = to_value(f["scan"]["sampling_frequency"][()])
        t0_delays = f["scan"]["t0_delays"][selected_tx]
        tx_apodization = f["scan"]["tx_apodizations"][selected_tx]
        probe_geometry = f["scan"]["probe_geometry"][:]
        probe_geometry = np.stack([probe_geometry[:, 0], probe_geometry[:, 2]], axis=0)
        initial_times = f["scan"]["initial_times"][selected_tx]
        lens_correction = to_value(f["scan"]["lens_correction"][()])

        try:
            sound_speed = to_value(f["scan"]["sound_speed"][:])
        except:
            sound_speed = to_value(f["scan"]["sound_speed"][()])

        try:
            element_width_m = to_value(f["scan"]["element_width"][()])
        except KeyError:
            log.warning("element width not found, using 1.33 wl")
            wavelength = sound_speed / probe_center_frequency
            element_width_m = 1.33 * wavelength
            log.warning(
                f"element width not found, using {element_width_m/wavelength:.2f} wl"
            )

        try:
            tgc_gain_curve = f["scan"]["tgc_gain_curve"][:]
        except KeyError:
            tgc_gain_curve = np.ones((n_ax,))
            log.warning("tgc_gain_curve not found, using all ones")

        try:
            beamformed_image = f["data"]["image"][0]
        except:
            pass
        polar_angles = f["scan"]["polar_angles"][:]

        # Get the number of datasets in f["scan"]["waveforms"]
        if use_two_way_waveforms:
            key = "waveforms_two_way"
        else:
            key = "waveforms_one_way"
        if key in f["scan"]:
            waveforms_key = key
        else:
            waveforms_key = "waveforms"

        n_tw = len(f["scan"][waveforms_key])
        if n_ax is None:
            n_ax = rf_data.shape[1]

        # ------------------------------------------------------------------------------
        # Load the waveforms
        # ------------------------------------------------------------------------------
        waveform_samples_list = []
        for n in range(n_tw):
            name = f"waveform_{n:03d}"
            waveform_samples = f["scan"][waveforms_key][name][:]
            waveform_samples_list.append(waveform_samples)

        tx_waveform_indices = f["scan"]["tx_waveform_indices"][selected_tx]
    # endregion

    # ==================================================================================
    # Preprocess the RF data
    # ==================================================================================
    # region
    # ----------------------------------------------------------------------------------
    # Normalize the RF data
    # ----------------------------------------------------------------------------------
    rf_normalization_factor = np.std(rf_data)
    rf_data = rf_data / rf_normalization_factor

    # endregion
    # ==================================================================================
    # Define derived variables
    # ==================================================================================
    # region

    # Define the reference wavelength for the grid as the center frequency
    grid_center_frequency = probe_center_frequency
    wavelength = sound_speed / grid_center_frequency

    n_el = probe_geometry.shape[1]

    # The assumed element width in wavelengths
    element_width_wl = element_width_m / wavelength

    # Compute the number of scatterers
    n_scat = grid_shape[0] * grid_shape[1]

    # Find the point where the waveform peaks
    t_peak = np.argmax(np.abs(waveform_samples_list[0])) / 250e6

    # Assign a random ID number to the run
    run_id = np.random.randint(0, 1e9)

    # Find the number of elements that are active in each transmit
    n_tx_el = np.sum(tx_apodization[0] > 0)

    for n in range(n_tx):
        if np.sum(tx_apodization[n] > 0) != n_tx_el:
            raise ValueError(
                "All transmits must have the same number of transmitting elements."
            )

    # Find the elements probe_geometry and t0_delays corresponsing to elements that
    # transmit
    probe_geometry_tx = np.zeros((n_tx, 2, n_tx_el))
    t0_delays_tx = np.zeros((n_tx, n_tx_el))
    for tx in range(n_tx):
        probe_geometry_tx[tx] = probe_geometry[:, tx_apodization[tx] > 0]
        t0_delays_tx[tx] = t0_delays[tx, tx_apodization[tx] > 0]

    tx_apod_iszero = np.zeros((n_tx, n_el))
    tx_apod_iszero[tx_apodization == 0] = 1

    # The lens correction in the file is in wavelengths. We need to convert it to
    # seconds.
    base_lens_correction_s = lens_correction * wavelength / sound_speed

    # ----------------------------------------------------------------------------------
    # Define the f-number for DAS beamforming
    # ----------------------------------------------------------------------------------

    # Select a f-number based on the element width. For phased array probes we use a
    # smaller f-number than for linear probes.
    if element_width_wl > 1:
        f_number = 2.5
    else:
        f_number = 0.5

    # The evaluation losses
    eval_steps = []
    eval_loss_no_reg = []
    rf_mse = []
    rf_mse_steps = []

    # endregion

    # ==================================================================================
    # Generate grid and amplitudes
    # ==================================================================================
    # region
    # ----------------------------------------------------------------------------------
    # Define the grid
    # ----------------------------------------------------------------------------------
    # Compute the values of the initial grid
    x_vals = (np.arange(grid_shape[1]) - grid_shape[1] / 2) * grid_dx_wl * wavelength
    z_vals = (np.arange(grid_shape[0])) * grid_dz_wl * wavelength + z_start
    xx, zz = np.meshgrid(x_vals, z_vals)

    grid_xlim = np.min(x_vals), np.max(x_vals)
    grid_zlim = np.min(z_vals), np.max(z_vals)

    scatterer_positions = np.stack([xx.ravel(), zz.ravel()], axis=0)
    # endregion

    # ==================================================================================
    # Define re/un-parameterization functions
    # ==================================================================================
    # region
    def reparameterize_scat_amp(opt_scat_amp):
        return jnp.exp(opt_scat_amp)

    def unparameterize_x(x):
        return jnp.log(x + 1e-9)

    def reparameterize_scat_x(opt_scat_x):
        return opt_scat_x * wavelength * grid_dx_wl

    def unparameterize_scat_x(scat_x):
        return scat_x / (wavelength * grid_dx_wl)

    def reparameterize_scat_z(opt_scat_z):
        return opt_scat_z * wavelength * grid_dz_wl

    def unparameterize_scat_z(scat_z):
        return scat_z / (wavelength * grid_dz_wl)

    def reparameterize_t0(opt_t0):
        return centered_sigmoid(opt_t0) * 2 * 4e-6

    def reparameterize_sens(opt_sens):
        return centered_sigmoid(opt_sens) * 2 * 0.99 + 1.0

    def reparameterize_elgain(opt_elgain):
        return 0.5 + centered_sigmoid(opt_elgain) * 2 * 0.5

    def reparameterize_sound_speed(opt_sound_speed):
        return 1540 + centered_sigmoid(opt_sound_speed) * 2 * 30

    def reparameterize_attenuation(opt_attenuation):
        return jnp.exp(opt_attenuation)

    def reparameterize_wvfm(opt_wvfm):
        return jnp.exp(opt_wvfm)

    def reparameterize_lens_correction(opt_lens_correction):
        return centered_sigmoid(opt_lens_correction) * 2 * 1e-6

    # Construct a dictionary that links the optimization variables to the
    # reparameterization function
    repar_fn = {
        "scat_amp": lambda opt_scat_amp: reparameterize_scat_amp(opt_scat_amp),
        "scat_x": lambda opt_scat_x: reparameterize_scat_x(opt_scat_x),
        "scat_z": lambda opt_scat_z: reparameterize_scat_z(opt_scat_z),
        "sens": lambda opt_sens: reparameterize_sens(opt_sens),
        "elgain": lambda opt_elgain: reparameterize_elgain(opt_elgain),
        "t0": lambda opt_t0: reparameterize_t0(opt_t0),
        "sound_speed": lambda opt_sound_speed: reparameterize_sound_speed(
            opt_sound_speed
        ),
        "attenuation": lambda opt_attenuation: reparameterize_attenuation(
            opt_attenuation
        ),
        "wvfm": lambda opt_wvfm: reparameterize_wvfm(opt_wvfm),
        "lens_correction": lambda opt_lens_correction: reparameterize_lens_correction(
            opt_lens_correction
        ),
    }

    def reparameterize_opt_vars(opt_vars):
        return jax.tree_util.tree_map(lambda x, fn: fn(x), opt_vars, repar_fn)

    # endregion

    # ==================================================================================
    # Initialize optimization variables
    # ==================================================================================
    # region

    # ----------------------------------------------------------------------------------
    # Perturb the gridpoints
    # ----------------------------------------------------------------------------------
    perturbation_x = (np.random.rand(n_scat) - 0.5) * grid_dx_wl * wavelength
    perturbation_z = (np.random.rand(n_scat) - 0.5) * grid_dz_wl * wavelength

    perturbation = np.stack([perturbation_x, perturbation_z], axis=0)

    perturbed_positions = scatterer_positions + perturbation
    # ----------------------------------------------------------------------------------
    # Initialize the optimization variables
    # ----------------------------------------------------------------------------------
    opt_vars = {
        "scat_amp": np.random.rand(scatterer_positions.shape[1]) - 10,
        "scat_x": unparameterize_scat_x(perturbed_positions[0]),
        "scat_z": unparameterize_scat_z(perturbed_positions[1]),
        "sens": np.zeros(4),
        "elgain": np.ones(n_el) * 2,
        "sound_speed": np.array(0.0),
        "t0": np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        "attenuation": np.log(0.75) * np.ones((1,)),
        "wvfm": -5 * np.ones(3),
        "lens_correction": np.zeros(3),
    }

    # Define dict to vmap over x, scat_x, scat_z
    # Has value 0 for x, scat_x, scat_z, and None for all other variables
    opt_vars_vmap_axes = {
        key: 0 if key in ["scat_amp", "scat_x", "scat_z"] else None
        for key in opt_vars.keys()
    }

    # ----------------------------------------------------------------------------------
    # Turn the optimization variables into jax arrays
    # ----------------------------------------------------------------------------------
    for key, value in opt_vars.items():
        opt_vars[key] = jnp.array(value)
    # endregion

    # ==================================================================================
    # Beamform the image
    # ==================================================================================
    # region
    das_pixel_grid = CartesianPixelGrid(
        n_x=grid_shape[1] * 2,
        n_z=grid_shape[0] * 2,
        dx_wl=grid_dx_wl * 0.5,
        dz_wl=grid_dz_wl * 0.5,
        z0=z_start,
        wavelength=wavelength,
    )
    das_extent = das_pixel_grid.extent

    if element_width_wl > 1:
        f_number = 3.5
    else:
        f_number = 0.5

    das_image = beamform_das(
        rf_data[None, ..., None],
        pixel_positions=das_pixel_grid.pixel_positions_flat,
        probe_geometry=probe_geometry.transpose(),
        t0_delays=t0_delays,
        tx_apodizations=tx_apodization,
        initial_times=initial_times,
        sampling_frequency=sampling_frequency,
        carrier_frequency=probe_center_frequency,
        sound_speed=sound_speed,
        sound_speed_lens=1000,
        lens_thickness=1.5e-3,
        t_peak=t_peak * jnp.ones((n_tw,)),
        rx_apodization=jnp.ones((n_tx, n_el)),
        f_number=f_number,
        iq_beamform=True,
    )[0]
    das_image = log_compress(das_image.reshape(das_pixel_grid.shape))

    # Define the normalization offset that will be used for all beamformed results
    # (Also used for the beamformed residuals)
    normalization_offset = np.max(das_image)
    das_image = das_image - normalization_offset

    # endregion
    # ==================================================================================
    # Define waveforms
    # ==================================================================================
    # region
    # Find the maximum length of the waveforms
    max_length = np.max(
        [waveform_samples.shape[0] for waveform_samples in waveform_samples_list]
    )
    # Pre-allocate a matrix for the waveforms and one for the envelopes
    N_WVFM = 20
    waveform_samples = np.zeros((n_tw, N_WVFM, max_length))
    waveform_samples_envelope = np.zeros((n_tw, max_length))
    # ==================================================================================
    # Lowpass filter waveforms
    # ==================================================================================
    waveform = waveform_samples_list[0]
    waveforms_lp = np.zeros((n_tw, N_WVFM, waveform.shape[0]))

    # Pre-allocate a vector for the phases
    waveform_phase = np.zeros((n_tw,))

    # Fill the matrices
    for n, samples in enumerate(waveform_samples_list):

        curr_length = samples.shape[0]
        waveform_samples[n, :, :curr_length] = samples[None]
        waveform_samples_envelope[n, :curr_length] = envelope_detect(
            samples, tw_center_frequency[tx_waveform_indices[n]], sampling_frequency
        )
        waveform_phase[n] = find_phase(
            samples, tw_center_frequency[tx_waveform_indices[n]], sampling_frequency
        )

    for tw in range(n_tw):
        for n in range(N_WVFM):
            alpha = N_WVFM / (n + 1)
            Wn_min = 0.01
            Wn_max = 0.1
            Wn = Wn_min + (Wn_max - Wn_min) * alpha / N_WVFM

            # Create a butterworth lowpass filter
            b, a = butter(N=3, Wn=Wn, btype="low")
            waveform_samples[tw, n] = filtfilt(
                b, a, waveform_samples[tw, n], padlen=128
            )

    waveform_samples = waveform_samples / np.max(np.abs(waveform_samples))

    # ==================================================================================
    # Turn arrays in jax arrays
    # ==================================================================================
    probe_geometry = jnp.array(probe_geometry)
    t0_delays = jnp.array(t0_delays)
    probe_geometry_tx = jnp.array(probe_geometry_tx)
    t0_delays_tx = jnp.array(t0_delays_tx)
    tx_apodization = jnp.array(tx_apodization)
    tx_apod_iszero = jnp.array(tx_apod_iszero)
    center_frequency = jnp.array(tw_center_frequency)
    initial_times = jnp.array(initial_times)
    tx_waveform_indices = jnp.array(tx_waveform_indices)
    polar_angles = jnp.array(polar_angles)
    tgc_gain_curve = jnp.array(tgc_gain_curve)
    waveform_samples = jnp.array(waveform_samples).astype(jnp.float32)
    waveform_samples_envelope = jnp.array(waveform_samples_envelope).astype(jnp.float32)
    waveform_phase = jnp.array(waveform_phase).astype(jnp.float32)

    # endregion
    # ==================================================================================
    # Define forward model and rf function
    # ==================================================================================
    # region
    @jit
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

    @jit
    def centered_sigmoid(x):
        """Sigmoid function centered at 0.5. The output is in the range (-0.5, 0.5)."""
        return sigmoid(x) - 0.5

    def get_sampled(waveform_samples):
        """Returns a function that behaves like the interpolated sampled waveforms.

        Parameters
        ----------
        waveform_samples : jnp.array
            The samples of shape `(n_tx, n_samp)`.
        """
        wvfm_sampling_frequency = 250e6

        assert waveform_samples.ndim == 3
        assert waveform_samples.dtype == jnp.float32

        n_samples = waveform_samples.shape[2]

        def func_sampled(tx, t, wvfm, depth_us):

            tw = tx_waveform_indices[tx]

            # Convert time to samples
            # Scale the time around the point wvfm[0] with the factor wvfm[1]
            float_wvfm_index = wvfm[0] + depth_us * wvfm[1]
            float_wvfm_index = jnp.clip(float_wvfm_index, 0, N_WVFM - 1)
            wvfm_index_min = jnp.floor(float_wvfm_index).astype(jnp.int32)
            wvfm_index_max = jnp.ceil(float_wvfm_index).astype(jnp.int32)

            float_sample = t * wvfm_sampling_frequency

            # Clip the samples to the range of the signal
            float_sample = jnp.clip(float_sample, 0, n_samples - 1)

            # Convert to integer samples around the float sample
            n_min = jnp.floor(float_sample).astype(jnp.int32)
            n_max = jnp.ceil(float_sample).astype(jnp.int32)

            # Get the values of the signal at the integer samples
            sample_min_a = waveform_samples[tw, wvfm_index_min, n_min]
            sample_max_a = waveform_samples[tw, wvfm_index_min, n_max]

            # Interpolate between the two samples
            eps = 1e-12
            interpolated_a = sample_min_a + (sample_max_a - sample_min_a) * (
                float_sample - n_min + eps
            )

            sample_min_b = waveform_samples[tw, wvfm_index_max, n_min]
            sample_max_b = waveform_samples[tw, wvfm_index_max, n_max]

            interpolated_b = sample_min_b + (sample_max_b - sample_min_b) * (
                float_sample - n_min + eps
            )

            # ------------------------------------------------------------------------------
            # Interpolate between the two lowpass filtered waveforms
            # ------------------------------------------------------------------------------
            interpolated = interpolated_a + (interpolated_b - interpolated_a) * (
                float_wvfm_index - wvfm_index_min
            )
            return interpolated

        func_sampled = jit(func_sampled)

        return func_sampled

    waveform = get_sampled(waveform_samples)

    @jit
    def directivity(theta, sens, element_width_wl=0.5):
        return (jnp.sinc(element_width_wl * sens[0] * jnp.sin(theta))) * jnp.cos(theta)

    @jit
    def lens_correction_fn(theta, lens_correction, base_correction):
        return base_correction + lens_correction[0] + lens_correction[1] * theta

    @jit
    def forward(ax, el, tx, opt_vars):
        """The forward model. Computes a single single sample in the RF data based on
        all scatterers and other parameters.

        Parameters
        ----------
        ax : int
            The axial index.
        el : int
            The element index.
        tx : int
            The transmit index.
        opt_vars : dict
            The optimization variables (after reparameterization).

        Returns
        -------
        np.array
            The received signal.
        """
        t0 = opt_vars["t0"]
        scat_x = opt_vars["scat_x"]
        scat_z = opt_vars["scat_z"]
        sens = opt_vars["sens"]
        elgain = opt_vars["elgain"]
        sound_speed = opt_vars["sound_speed"]
        attenuation = opt_vars["attenuation"]
        wvfm = opt_vars["wvfm"]
        scatterer_amplitude = opt_vars["scat_amp"]
        lens_correction = opt_vars["lens_correction"]

        if not enable_sound_speed:
            sound_speed = 1540.0

        if not enable_waveform_shaping:
            wvfm = wvfm * 0.0

        if enable_global_t0_shifting:
            t0_shift_global = t0[0]
        else:
            t0_shift_global = 0.0

        # t0[0] models any additional delay. This might be needed if the sampled waveform
        # some offset of there is some other delay in the system.

        element_position = probe_geometry[:, el]

        # Have the scatterers move with the t0[0] delay. This allows for optimization of
        # the t0[0] delay without invalidating the scatterer positions that er there
        # already.
        # We have 1 scatterer at depth z1 already correct. Another one at z2 is not correct
        # because of an incorrect t0 delay. We can correct this by introducing a t0[0]
        # delay. This however also moves the scatterer at z1. To correct this we move the
        # scatterers with the t0[0] delay.
        scat_z = scat_z - 0.5 * t0_shift_global * sound_speed

        # Scale the scatterer positions with the speed of sound
        scatterer_position = (
            jnp.array([scat_x, scat_z])
            / wavelength
            * sound_speed
            / grid_center_frequency
        )

        # scatterer_position = jnp.array([scat_x, scat_z])

        t_tx_travel = (
            jnp.linalg.norm(probe_geometry_tx[tx] - scatterer_position[:, None], axis=0)
            / sound_speed
        )

        t_tx = t0_delays_tx[tx] + t_tx_travel
        # ------------------------------------------------------------------------------
        # Only consider the first sub-wavefront that arrives
        # ----------------------------------------------------t0[1 + tx]--------------------------
        if wavefront_only:
            t_tx = jnp.min(t_tx)
            t_tx_travel = jnp.min(t_tx_travel)
            # When computing only the wavefront it is not possible to compute individual
            # directivities. Therefore, the directivity is set to 1.0.
            directivity_tx = 1.0
            t_lens_correction_tx = base_lens_correction_s

        else:
            if enable_directivity:
                theta = jnp.arctan2(
                    probe_geometry_tx[tx][0] - scatterer_position[0],
                    scatterer_position[1] - probe_geometry_tx[tx][1],
                )

            if enable_directivity:
                directivity_tx = directivity(theta, sens, element_width_wl)
            else:
                directivity_tx = 1.0

            t_lens_correction_tx = base_lens_correction_s

        # ------------------------------------------------------------------------------
        # Compute the travel time back to an element
        # ------------------------------------------------------------------------------
        t_rx = (
            jnp.linalg.norm(element_position - scatterer_position, axis=0) / sound_speed
        )
        t = t_tx + t_rx

        # ------------------------------------------------------------------------------
        # Compute the angular response of the receiving element
        # ------------------------------------------------------------------------------
        if enable_directivity:
            theta = jnp.arctan2(
                (element_position[0] - scatterer_position[0]),
                (scatterer_position[1] - element_position[1]),
            )

        if enable_directivity:
            directivity_rx = directivity(theta, sens, element_width_wl)
        else:
            directivity_rx = 1.0

        t_lens_correction_rx = base_lens_correction_s

        if enable_lens_correction:
            t = t + t_lens_correction_rx + t_lens_correction_tx
        # ------------------------------------------------------------------------------
        # Compute the time instant corresponding to the axial index
        # ------------------------------------------------------------------------------
        t_ax = ax / sampling_frequency + initial_times[tx]

        # ------------------------------------------------------------------------------
        # Model the attenuation due to spread
        # ------------------------------------------------------------------------------
        if enable_attenuation_spread:
            base_dist_time = 3e-6
            att_tx = base_dist_time / t_tx_travel
            att_rx = base_dist_time / t_rx

            att_tx = jnp.clip(att_tx, 0, 1)
            att_rx = jnp.clip(att_rx, 0, 1)
        else:
            att_tx = 1.0
            att_rx = 1.0

        # ------------------------------------------------------------------------------
        # Model the attenuation due to absorption
        # ------------------------------------------------------------------------------
        if enable_attenuation_absorption:
            alpha_db = (
                attenuation[0]
                * (t_tx_travel + t_rx)
                * sound_speed
                * 100
                * center_frequency[tx]
                / 1e6
            )
            att_absorption = 10 ** (-alpha_db / 20)
        else:
            att_absorption = 1.0

        # ------------------------------------------------------------------------------
        # Enable/disable element gain
        # ------------------------------------------------------------------------------
        if enable_element_gain:
            element_gain = elgain[el]
        else:
            element_gain = 1.0

        # ------------------------------------------------------------------------------
        # Enable/disable TGC compensation
        # ------------------------------------------------------------------------------
        if enable_tgc_compensation:
            tgc_scaling = tgc_gain_curve[ax]
        else:
            tgc_scaling = 1.0

        # Compute the depth in us
        depth_us = (t_tx_travel + t_rx) * 1e6

        y = jnp.sum(
            waveform(tx, t_ax - (t + t0_shift_global), wvfm, depth_us)
            * att_tx
            * att_absorption
            * directivity_tx
        )
        y = (
            y
            * tgc_scaling
            * scatterer_amplitude
            * directivity_rx
            * element_gain
            * att_rx
        )

        return y

    @jit
    def rf_function(ax, el, tx, opt_vars):
        """Computes the forward model vmapped over elements, axial samples, transmits, and
        scatterers."""
        # Vmap over element index
        rf_fn = vmap(
            forward,
            in_axes=(None, 0, None, None),
        )
        # Vmap over axial index
        rf_fn = vmap(
            rf_fn,
            in_axes=(
                0,
                None,
                None,
                None,
            ),
        )
        # Vmap over transmit index
        rf_fn = vmap(
            rf_fn,
            in_axes=(
                None,
                None,
                0,
                None,
            ),
        )
        # Vmap over scatterer index
        rf_fn = vmap(
            rf_fn,
            in_axes=(None, None, None, opt_vars_vmap_axes),
        )

        return rf_fn(ax, el, tx, opt_vars)

    def get_rf_chunked(n_ax, n_tx, opt_vars, chunk_size, chunk_size_el=40):
        if n_ax % chunk_size != 0:
            log.warning(
                "n_ax is not divisible by chunk_size. The last chunk will be smaller."
            )
        # Define the axial and element indices
        ax = jnp.arange(n_ax)
        el = jnp.arange(n_el)
        tx = jnp.arange(n_tx)
        transmit_chunks = []
        for n in range(n_tx):

            n_chunks = ceil(n_ax / chunk_size)
            chunks = []
            for m in range(n_chunks):
                el_chunks = []
                for k in range(ceil(n_el / chunk_size_el)):

                    y = rf_function(
                        ax[m * chunk_size : (m + 1) * chunk_size],
                        el[k * chunk_size_el : (k + 1) * chunk_size_el],
                        tx[n : n + 1],
                        opt_vars,
                    )
                    el_chunks.append(y)
                y = jnp.concatenate(el_chunks, axis=3)
                # Sum over the scatterers
                y = jnp.sum(y, axis=0)
                # Remove transmit dimension
                y = y[0]
                chunks.append(y)
            # Concatenate the chunks along the axial dimension
            y = jnp.concatenate(chunks, axis=0)
            transmit_chunks.append(y)
        y = jnp.stack(transmit_chunks, axis=0)
        return y

    # endregion

    # ==================================================================================
    # Define loss and gradient functions
    # ==================================================================================
    # region

    def forward_all_scatterers(ax, el, tx, opt_vars):
        """Computes the forward model vmapped over scatterers.

        Parameters
        ----------
        ax : int
            The axial index.
        el : int
            The element index.
        tx : int
            The transmit index.
        opt_vars : dict
            The optimization variables.

        Returns
        -------
        np.array
            The dot product of h and x.
        """
        y = vmap(
            forward,
            in_axes=(None, None, None, opt_vars_vmap_axes),
        )(ax, el, tx, opt_vars)
        return jnp.sum(y)

    def sample_loss_fn(ax, el, tx, rf, opt_vars):
        """Computes the loss for a single sample. This function is designed to be vmapped
        over the samples.

        Parameters
        ----------
        ax : int
            The axial index.
        el : int
            The element index.
        tx : int
            The transmit index.
        rf : jnp.array
            The true rf data sample.
        opt_vars : dict
            The optimization variables.

        Returns
        -------
        jnp.array
            The loss.
        """
        squared_error = jnp.square(rf - forward_all_scatterers(ax, el, tx, opt_vars))

        if False:
            # Apply weighting to put more weight on axial samples that are further away from the
            # transducer. This is because these values are smaller and will thus be
            # underrepresented in the gradient.
            squared_error = squared_error * tgc_gain_curve[ax]

        return squared_error

    @jit
    def loss_fn(ax, el, tx, rf, opt_vars):
        loss_vector_fn = vmap(
            sample_loss_fn,
            in_axes=(0, 0, 0, 0, None),
        )
        # Apply reparemeterization to all variables
        reparametrized_opt_vars = reparameterize_opt_vars(opt_vars)

        loss = jnp.mean(loss_vector_fn(ax, el, tx, rf, reparametrized_opt_vars))

        if regularization == 0:
            return loss
        else:
            return loss + regularization * (
                jnp.mean(jnp.abs(reparametrized_opt_vars["scat_amp"]))
            )

    @jit
    def loss_grad(ax, el, tx, rf, opt_vars):
        return grad(loss_fn, argnums=4)(ax, el, tx, rf, opt_vars)

    def loss_fn_chunked(
        ax,
        el,
        tx,
        rf,
        opt_vars,
        chunk_size,
        regularization,
    ):
        n_samples = ax.shape[0]
        n_chunks = n_samples // chunk_size
        loss = 0
        for n in trange(n_chunks):
            loss += loss_fn(
                ax[n * chunk_size : (n + 1) * chunk_size],
                el[n * chunk_size : (n + 1) * chunk_size],
                tx[n * chunk_size : (n + 1) * chunk_size],
                rf[n * chunk_size : (n + 1) * chunk_size],
                opt_vars,
                regularization,
            )
        return loss / n_chunks

    # endregion

    # ==================================================================================
    # Generate RF data
    # ==================================================================================
    log.info("Generating RF data")

    # ==================================================================================
    # Generate the RF data
    # ==================================================================================
    # region
    if np.sum(tx_apodization) < 10:
        BIG_CHUNK = True
    else:
        BIG_CHUNK = False

    def get_rf_from_simulator(opt_vars):
        log.info("Simulating RF data...")
        return get_rf_chunked(
            n_ax,
            n_tx,
            opt_vars,
            chunk_size=64 if wavefront_only or BIG_CHUNK else 1,
        )

    # Get the standard deviation of the rf data excluding the first samples
    rf_data_std = jnp.std(rf_data[:, ax_min:])
    # endregion

    # ==================================================================================
    # Get arrays of (ax, el, tx)
    # ==================================================================================
    # These are used to select the indices of the data for batches of samples.
    # ==================================================================================
    # region
    # Define the possible values for the indices
    ax_vals = np.arange(n_ax - ax_min) + ax_min
    el_cutoff = 0
    el_vals = np.arange(n_el - 2 * el_cutoff) + el_cutoff
    tx_vals = np.arange(n_tx)

    # Create a meshgrid of the indices
    ax, el, tx = np.meshgrid(ax_vals, el_vals, tx_vals)

    # Unravel the indices
    ax = ax.ravel()
    el = el.ravel()
    tx = tx.ravel()

    rf_unraveled = rf_data[tx, ax, el]

    # ==================================================================================
    # Create working directory
    # ==================================================================================
    week_nr = datetime.now().isocalendar()[1]
    date_str = datetime.now().strftime("%Y-%m-%d")
    working_dir = Path("results", "inverse-beamform", f"week-{week_nr}", date_str)
    if super_folder is not None:
        working_dir = working_dir / super_folder

    working_dir = create_unique_dir(working_dir, name=f"{run_name}", prepend_date=True)

    log.info("Working directory:")
    log.info(f"--- > {working_dir}")

    promise_mark(working_dir)

    # Store run id
    with open(working_dir / f"{run_id}.txt", "w", encoding="UTF-8") as f:
        f.write("")

    # Store beamformed image
    _, axes = plt.subplots()
    plot_beamformed(
        ax=axes,
        image=beamformed_image,
        extent_m=das_extent,
        title="Beamformed image",
        vmin=das_dynamic_range,
        probe_geometry=probe_geometry.T,
    )
    plt.close()

    # Copy over all the source files in the call stack
    call_stack_paths = get_call_stack_paths()
    for p in call_stack_paths:
        shutil.copy(p, working_dir / f"source_{p.name}")
    # endregion

    # ==================================================================================
    # Optimize
    # ==================================================================================
    # ----------------------------------------------------------------------------------
    # Define init function
    # ----------------------------------------------------------------------------------
    scheduler = optax.exponential_decay(
        init_value=learning_rate, transition_steps=2000, decay_rate=0.8
    )

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip(max_delta=1e3),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=gradient_accumulation)

    optimizer_state = optimizer.init(opt_vars)

    @jit
    def update(opt_vars, optimizer_state, ax, el, tx, rf):
        grads = loss_grad(ax, el, tx, rf, opt_vars)

        updates, opt_state = optimizer.update(grads, optimizer_state)

        if not enable_optimize_positions:
            grads["scat_x"] = jnp.zeros_like(grads["scat_x"])
            grads["scat_z"] = jnp.zeros_like(grads["scat_z"])

        # Apply the updates
        new_opt_vars = optax.apply_updates(opt_vars, updates)
        return new_opt_vars, opt_state

    # ==================================================================================
    # Optimization loop
    # ==================================================================================
    log.info("Starting optimization")
    for step in range(n_steps):
        if step > 1000:
            mark_run_started(working_dir)
        # ------------------------------------------------------------------------------
        # Evaluation and plotting
        # ------------------------------------------------------------------------------
        # region
        if step % plot_interval == 0 or step == n_steps - 1:

            # ------------------------------------------------------------------------------
            # Add one to the step to have the filenames end at 5000 instead of 4999
            # ------------------------------------------------------------------------------
            if step == n_steps - 1:
                step = step + 1

            # ==============================================================================
            # Generate images, RF data, and losses
            # ==============================================================================
            # region
            # ------------------------------------------------------------------------------
            # Generate RF data based on the current optimization variables
            # ------------------------------------------------------------------------------
            rf_data_hat = get_rf_from_simulator(reparameterize_opt_vars(opt_vars))
            rf_data_hat = np.array(rf_data_hat)
            if rf_data_hat.ndim == 4:
                rf_data_hat = rf_data_hat[..., 0]

            # ------------------------------------------------------------------------------
            # Compute and log losses
            # ------------------------------------------------------------------------------
            rf_error = rf_data_hat - rf_data
            rf_error = rf_error[:, ax_min:]
            rf_mse.append(np.mean(rf_error**2))
            rf_mse_steps.append(step)
            log.info(f"rf data MSE: {yellow(rf_mse[-1])}")

            current_loss = rf_mse[-1]

            # Add the evaluation loss
            eval_loss_no_reg.append(current_loss)
            eval_steps.append(step)

            # ------------------------------------------------------------------------------
            # Perform DAS beamforming on the residual
            # ------------------------------------------------------------------------------

            beamformed_residual = beamform_das(
                rf_data[None, ..., None] - rf_data_hat[None, ..., None],
                pixel_positions=das_pixel_grid.pixel_positions_flat,
                probe_geometry=probe_geometry.transpose(),
                t0_delays=t0_delays,
                tx_apodizations=tx_apodization,
                initial_times=initial_times,
                sampling_frequency=sampling_frequency,
                carrier_frequency=probe_center_frequency,
                sound_speed=sound_speed,
                sound_speed_lens=1000,
                lens_thickness=1.5e-3,
                t_peak=t_peak * jnp.ones((n_tw,)),
                rx_apodization=jnp.ones((n_tx, n_el)),
                f_number=f_number,
                iq_beamform=True,
            )[0]

            beamformed_residual = log_compress(
                beamformed_residual.reshape(das_pixel_grid.shape)
            )

            # ------------------------------------------------------------------------------
            # Turn the obtained scatterers into an image
            # ------------------------------------------------------------------------------
            c = repar_fn["sound_speed"](opt_vars["sound_speed"])
            t0 = repar_fn["t0"](opt_vars["t0"])[0]
            im = get_kernel_image(
                xlim=grid_xlim,
                zlim=grid_zlim,
                scatterer_amplitudes=repar_fn["scat_amp"](opt_vars["scat_amp"]),
                scatterer_x=repar_fn["scat_x"](opt_vars["scat_x"]),
                scatterer_z=repar_fn["scat_z"](opt_vars["scat_z"]) - 0.5 * t0 * c,
                pixel_size=0.15e-3,
                radius=wavelength * 0.65,
                falloff_power=2,
            )
            im = np.array(im)
            im = 20 * np.log10(np.abs(im) + 1e-9)
            # endregion

            # ==============================================================================
            # Call the plotting functions
            # ==============================================================================
            # region
            plot_overview(
                working_dir,
                step,
                beamformed_residual,
                das_image,
                das_extent,
                das_dynamic_range,
                im,
                n_tx,
                grid_shape,
                x_vals,
                z_vals,
                probe_geometry,
                normalization_offset,
                vmin_rf=-rf_data_std,
                rf_data=rf_data,
                rf_data_hat=rf_data_hat,
                repar_fn=repar_fn,
                opt_vars=opt_vars,
                run_id=run_id,
            )
            plot_loss(working_dir, eval_loss_no_reg, eval_steps)
            plot_rf_error(working_dir, step, rf_data, rf_data_hat, ax_min)
            plot_elgain(
                working_dir,
                step,
                elgain=repar_fn["elgain"](opt_vars["elgain"]),
            )
            plot_waveforms(
                working_dir,
                step,
                waveform_fn=waveform,
                wvfm=repar_fn["wvfm"](opt_vars["wvfm"]),
                waveform_samples=waveform_samples,
                n_tx=n_tx,
                tx_waveform_indices=tx_waveform_indices,
            )
            plot_directivity(
                working_dir,
                step,
                sens=repar_fn["sens"](opt_vars["sens"]),
                element_width_wl=element_width_wl,
                directivity=directivity,
                directivity_rx_fn=directivity,
                directivity_tx_fn=directivity,
            )

            # --------------------------------------------------------------------------
            # Save the optimization variables
            # --------------------------------------------------------------------------
            var_checkpoint_dir = working_dir / "opt_vars"
            var_checkpoint_dir.mkdir(exist_ok=True)
            np.savez(
                var_checkpoint_dir / f"vars_{str(step).zfill(6)}.npz",
                **reparameterize_opt_vars(opt_vars),
                grid_xlim=grid_xlim,
                grid_zlim=grid_zlim,
                path=str(hdf5_path),
                selected_tx=selected_tx,
                frame=frame,
                im=im,
                n_ax=n_ax,
                rf_data_hat=rf_data_hat,
                ax_min=ax_min,
            )
            # endregion
            # endregion

        # ------------------------------------------------------------------------------
        # Perturb the gridpoints
        # ------------------------------------------------------------------------------
        # region
        if (
            enable_scatterer_redistribution
            and step % redistribute_interval == redistribute_interval - 1
        ):
            log.info("Redistributing scatterers")

            scat_amps = repar_fn["scat_amp"](opt_vars["scat_amp"])
            mean_amp = jnp.mean(scat_amps)

            relative_amp = scat_amps / mean_amp

            # Find the bottom 5% of scatterers
            mask_low_amp = relative_amp < 1e-1
            log.info(f"Redistributing {jnp.sum(mask_low_amp)} scatterers")

            scat_x = opt_vars["scat_x"]
            scat_z = opt_vars["scat_z"]

            key1, key2 = random.split(random.PRNGKey(step))
            # Generate random perturbations based on opt_scat_amp
            opt_scat_x_new = unparameterize_scat_x(
                random.uniform(key1, scat_x.shape, jnp.float32, x_vals[0], x_vals[-1])
            )
            opt_scat_z_new = unparameterize_scat_z(
                random.uniform(key2, scat_z.shape, jnp.float32, z_vals[0], z_vals[-1])
            )

            opt_vars["scat_x"] = scat_x.at[mask_low_amp].set(
                opt_scat_x_new[mask_low_amp]
            )
            opt_vars["scat_z"] = scat_z.at[mask_low_amp].set(
                opt_scat_z_new[mask_low_amp]
            )

        # endregion

        # ------------------------------------------------------------------------------
        # Compute the gradients
        # ------------------------------------------------------------------------------
        # region

        # ------------------------------------------------------------------------------
        # Get a random subset of the samples
        # ------------------------------------------------------------------------------
        for _ in range(gradient_accumulation):
            indices = np.random.choice(
                ax.shape[0],
                size=batch_size // 2,
                replace=False,
            )
            ax_selected = ax[indices]
            ax_selected = np.clip(ax_selected, ax_min, None)
            el_selected = el[indices]
            tx_selected = tx[indices]
            rf_selected = rf_unraveled[indices]

            opt_vars, optimizer_state = update(
                opt_vars,
                optimizer_state,
                ax_selected,
                el_selected,
                tx_selected,
                rf_selected,
            )

        if step_update_interval > 0 and step % step_update_interval == 0:
            log.info(f"step: {step}")

        # endregion

    return working_dir


# ==================================================================================
# Define functions
# ==================================================================================
# region
# ----------------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------------
def plot_rf(
    ax,
    rf_data,
    start_sample=0,
    extent_m=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    aspect="auto",
    axis_in_mm=True,
    title=None,
):
    """Plots RF data to an axis.

    Parameters
    ----------
    ax : Plt.Axes
        The axis to plot to.
    rf_data : np.ndarray
        The RF data to plot.
    start_sample : int, default=0
        The sample number to start plotting from.
    extent_m : list, default=None
        The extent of the plot in meters. If None, the extent is set to the number of
        elements and samples.
    cmap : str, default="viridis"
        The colormap to use.
    vmin : float, default=None
        The minimum value of the colormap. If None, the minimum value is set to the 0.5
        percentile of the data.
    vmax : float, default=None
        The maximum value of the colormap. If None, the maximum value is set to the 99.5
        percentile of the data.
    aspect : str, default="auto"
        The aspect ratio of the plot.
    axis_in_mm : bool, default=True
        Whether to plot the x-axis in mm.
    """
    formatter = FuncFormatter(lambda x, _: f"{int(x)}")
    if extent_m is not None:
        if axis_in_mm:
            xlabel = "x [mm]"
            zlabel = "z [mm]"
        else:
            xlabel = "x [m]"
            zlabel = "z [m]"
        kwargs = {"extent": extent_m}

    else:
        kwargs = {"aspect": aspect}
        xlabel = "element [-]"
        zlabel = "sample [-]"

    if vmin is None and vmax is None:
        vmin = np.percentile(rf_data, 0.5)
        vmax = np.percentile(rf_data, 99.5)

    # Plot the RF data to the axis
    ax.imshow(
        rf_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    # Set the formatter for the major ticker on both axes
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(zlabel)
    # Set the yticks to start at the start_sample
    n_ax = rf_data.shape[0]
    if n_ax >= 500:
        # Divide the axis into 4 parts and round to multiples of 100
        step = int(np.floor((n_ax / 4) / 100) * 100)
    else:
        # Divide the axis into 4 parts and round to multiples of 50
        step = int(np.floor((n_ax / 4) / 50) * 50)
    step = max(step, 10)
    ax.set_yticks(np.arange(0, n_ax, step) + start_sample)
    ax.set_xticks(np.linspace(0, rf_data.shape[1], 4))

    if title is not None:
        ax.set_title(title)


def plot_beamformed(
    ax,
    image,
    extent_m,
    vmin=-60,
    vmax=0,
    cmap="gray",
    axis_in_mm=True,
    probe_geometry=None,
    title=None,
):
    """Plots a beamformed image to an axis.

    Parameters
    ----------
    ax : Plt.Axes
        The axis to plot to.
    image : np.ndarray
        The image to plot.
    extent_m : list
        The extent of the image in meters.
    vmin : float, default=-60
        The minimum value of the colormap.
    vmax : float, default=0
        The maximum value of the colormap.
    cmap : str, default="gray"
        The colormap to use.
    axis_in_mm : bool, default=True
        Whether to plot the x-axis in mm.
    probe_geometry : np.ndarray, default=None
        The geometry of the probe. If not None, the probe geometry is plotted.
    title : str, default=None
        The title of the plot.
    """

    if axis_in_mm:
        xlabel = "x [mm]"
        zlabel = "z [mm]"
        formatter = FuncFormatter(lambda x, _: f"{round(1000*x)}")
    else:
        xlabel = "x [m]"
        zlabel = "z [m]"
        formatter = FuncFormatter(lambda x, _: f"{x:.3f}")

    ax.imshow(
        image,
        extent=extent_m,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
        interpolation="none",
    )

    # Set the formatter for the major ticker on both axes
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(zlabel)

    if probe_geometry is not None:

        ax.plot(
            [probe_geometry[0, 0], probe_geometry[-1, 0]],
            [probe_geometry[0, 1], probe_geometry[-1, 1]],
            "-|",
            markersize=6,
            color="#AA0000",
            linewidth=1,
        )

    if title is not None:
        ax.set_title(title)


def get_kernel_image(
    xlim,
    zlim,
    scatterer_x,
    scatterer_z,
    scatterer_amplitudes,
    pixel_size,
    radius=1e-3,
    falloff_power=3,
):
    def pixel_brightness(pix_x, pix_z, scat_x, scat_z, scat_amp):
        r = jnp.sqrt((pix_x - scat_x) ** 2 + (pix_z - scat_z) ** 2) / radius
        return np.sum(scat_amp * jnp.exp(np.log(0.1) * jnp.abs(r**falloff_power)))

    pixel_brightness = jit(vmap(pixel_brightness, in_axes=(0, 0, None, None, None)))

    n_x = int((xlim[1] - xlim[0]) / pixel_size)
    n_z = int((zlim[1] - zlim[0]) / pixel_size)
    x = jnp.linspace(xlim[0], xlim[1], n_x)
    z = jnp.linspace(zlim[0], zlim[1], n_z)
    X, Z = jnp.meshgrid(x, z)
    X = X.ravel()
    Z = Z.ravel()

    num_pixels = X.size
    chunk_size = 1024
    n_pixels_rounded_up = num_pixels + chunk_size - (num_pixels % chunk_size)

    intensities = []
    for i in range(0, n_pixels_rounded_up, chunk_size):
        intensities.append(
            pixel_brightness(
                X[i : i + chunk_size],
                Z[i : i + chunk_size],
                scatterer_x,
                scatterer_z,
                scatterer_amplitudes,
            )
        )
    image = jnp.concatenate(intensities)
    image = jnp.reshape(image, (n_z, n_x))

    return image


def load_opt_vars(path):
    opt_vars = np.load(path)
    opt_scat_amp = opt_vars["opt_scat_amp"]
    opt_scat_x = opt_vars["opt_scat_x"]
    opt_scat_z = opt_vars["opt_scat_z"]
    opt_sens = opt_vars["opt_sens"]
    opt_t0 = opt_vars["opt_t0"]
    opt_elgain = opt_vars["opt_elgain"]
    opt_sound_speed = opt_vars["opt_sound_speed"]
    opt_attenuation = opt_vars["opt_attenuation"]
    opt_wvfm = opt_vars["opt_wvfm"]
    return (
        opt_scat_amp,
        opt_scat_x,
        opt_scat_z,
        opt_sens,
        opt_t0,
        opt_elgain,
        opt_sound_speed,
        opt_attenuation,
        opt_wvfm,
    )


def mark_run_started(working_dir):
    """Creates a file `started.txt` to indicate that the run has started."""
    with open(working_dir / "started.txt", "w", encoding="UTF-8") as f:
        f.write(
            "This file indicates that the run has been going for a significant amount "
            "of time."
        )


def promise_mark(working_dir):
    """Creates a file `will_mark.txt` to indicate that this run will call
    `mark_run_started`. This way we can avoid delete the folder if it contains
    the `will_mark.txt` file, but not the `started.txt` file."""
    with open(working_dir / "will_mark.txt", "w", encoding="UTF-8") as f:
        f.write(
            "This file indicates that the run will create a `started.txt` file to "
            "indicate that the run has been going for a significant amount of time."
            "If this file (`will_mark.txt`) is present, but the `started.txt` file is "
            "not, the folder can be safely deleted."
        )


# ----------------------------------------------------------------------------------
# Define plotting functions
# ----------------------------------------------------------------------------------
def plot_directivity(
    working_dir,
    step,
    sens,
    element_width_wl,
    directivity,
    directivity_rx_fn,
    directivity_tx_fn,
):
    """Creates a plot of the current directivity profile of the transducer elements."""
    fig, ax = plt.subplots(1, 1)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 128)
    beampattern_rx = directivity_rx_fn(theta, sens, element_width_wl)
    beampattern_tx = directivity_tx_fn(theta, sens, element_width_wl)
    beampattern_original = directivity(theta, jnp.ones(4), element_width_wl)
    ax.plot(theta * 180 / np.pi, beampattern_rx, label="current directivity rx")
    ax.plot(theta * 180 / np.pi, beampattern_tx, label="current directivity tx")
    ax.plot(theta * 180 / np.pi, beampattern_original, "--w", label="original")

    ax.set_xlabel("angle [deg]")
    ax.set_ylabel("sensitivity")
    ax.set_ylim([0, 1.1])
    ax.legend()
    plt.tight_layout()
    beampattern_dir = working_dir / "directivity"
    beampattern_dir.mkdir(exist_ok=True)
    plt.savefig(beampattern_dir / f"directivity_{str(step).zfill(6)}.png", dpi=256)
    plt.close()


def plot_elgain(working_dir, step, elgain):
    """Creates a stem plot of the element gains."""
    fig, ax = plt.subplots()
    ax.stem(elgain)
    ax.set_xlabel("element index")
    ax.set_ylabel("sensitivity")
    plt.tight_layout()
    elgain_dir = working_dir / "element_gain"
    elgain_dir.mkdir(exist_ok=True)
    plt.savefig(elgain_dir / f"gain_{str(step).zfill(6)}.png", dpi=256)
    plt.close()


def plot_waveforms(
    working_dir, step, waveform_fn, wvfm, waveform_samples, n_tx, tx_waveform_indices
):

    # Create empty set
    unique_tw_tx = set()
    for tx in range(n_tx):
        tw = int(tx_waveform_indices[tx].item())
        if tw in unique_tw_tx:
            continue
        unique_tw_tx.add(tx)

    fig, axes = plt.subplots(len(unique_tw_tx), 1, figsize=(4, 2))
    t = np.arange(waveform_samples.shape[2] * 2) / 250e6
    if len(unique_tw_tx) == 1:
        axes = np.array([axes])

    for tx in unique_tw_tx:
        tw = tx_waveform_indices[tx]
        y_original = waveform_fn(tw, t, np.zeros_like(wvfm), 15e-3)
        y = waveform_fn(tx, t, wvfm, 15)
        y2 = waveform_fn(tx, t, wvfm, 30)
        y3 = waveform_fn(tx, t, wvfm, 60)

        axes[tx].plot(t * 1e6, y_original, "--w", label="original")
        axes[tx].plot(t * 1e6, y, label="stretched 15us")
        axes[tx].plot(t * 1e6, y2, label="stretched 30us")
        axes[tx].plot(t * 1e6, y3, label="stretched 60us")
        axes[tx].set_xlabel("time [us]")
        axes[tx].set_ylabel("amplitude")
        axes[tx].set_ylim([-1.5, 1.5])
        axes[tx].legend(loc="upper right")
        axes[tx].set_title(f"waveform {tw} (tx:{tx}): {1 + wvfm[3+tw]:.2f}")
    fig.suptitle(f"wvfm {wvfm[:3]}")
    plt.tight_layout()
    waveform_dir = working_dir / "waveforms"
    waveform_dir.mkdir(exist_ok=True)
    plt.savefig(waveform_dir / f"waveform_{str(step).zfill(6)}.png", dpi=256)
    plt.close()


def plot_overview(
    working_dir,
    step,
    beamformed_residual,
    das_image,
    das_extent,
    das_dynamic_range,
    inverse_image,
    n_tx,
    grid_shape,
    x_vals,
    z_vals,
    probe_geometry,
    normalization_offset,
    vmin_rf,
    rf_data,
    rf_data_hat,
    repar_fn,
    opt_vars,
    run_id,
):
    """Creates an overview plot of the current state of the optimization with the
    beamformed image, the estimated- and true RF data, and the residual image.

    Parameters
    ----------
    working_dir : Path
        The directory to save the image to.
    step : int
        The step number.
    beamformed_residual : np.ndarray
        The residual image after beamforming.
    das_image : np.ndarray
        The DAS beamformed image.
    das_extent : list
        The extent of the DAS beamformed image in meters.
    das_dynamic_range : float
        The dynamic range of the DAS beamformed image. The range will be
        [das_dynamic_range, 0]dB.
    inverse_image : np.ndarray
        The inverse image of the RF data.
    n_tx : int
        The number of transmits.
    grid_shape : np.ndarray
        The shape of the grid.
    x_vals, z_vals : np.ndarray
        The x and z axis values of the grid.
    probe_geometry : np.ndarray
        The geometry of the probe in meters of shape (n_el, 2).
    normalization_offset : float
        The normalization offset of the beamformed image. This is used to keep the
        normalization the same acress steps and across images.
    vmin_rf : float
        The minimum value of the RF data colormap.
    rf_data : np.ndarray
        The true RF data.
    rf_data_hat : np.ndarray
        The RF data estimated by the forward model.
    repar_fn : dict
        The reparameterization functions.
    opt_vars : dict
        The optimization variables.
    run_id : str
        The run id to be displayed in the title.
    """
    n_rf_plots = np.min([n_tx, 3])

    # Create gridspec
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, n_rf_plots + 3)

    # Add subplots to the gridspec
    ax_image = fig.add_subplot(gs[:, 1:3])
    ax_residual = fig.add_subplot(gs[1, 0])
    ax_beamformed = fig.add_subplot(gs[0, 0])

    # Create subplots for the estimated and true rf data
    ax_rf_hat = []
    ax_rf_true = []
    for n in range(n_rf_plots):
        ax_rf_hat.append(fig.add_subplot(gs[0, n + 3]))
        ax_rf_true.append(fig.add_subplot(gs[1, n + 3]))

    # ------------------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------------------
    db_min = das_dynamic_range
    normalization_factor = 1
    ax_image.set_title(f"x_hat [{das_dynamic_range}, 0]dB")
    im_shape = (grid_shape / 2.5).astype(np.int32)
    im = inverse_image

    extent = [x_vals[0], x_vals[-1], z_vals[-1], z_vals[0]]
    plot_beamformed(
        ax_image, im, extent_m=extent, vmin=db_min, probe_geometry=probe_geometry.T
    )

    plot_beamformed(
        ax_residual,
        beamformed_residual - normalization_offset,
        extent_m=das_extent,
        vmin=db_min,
        probe_geometry=probe_geometry.T,
        title="residual",
    )
    plot_beamformed(
        ax_beamformed,
        das_image,
        extent_m=das_extent,
        vmin=db_min,
        probe_geometry=probe_geometry.T,
        title="DAS",
    )

    # ------------------------------------------------------------------------------
    # Estimated Rf data
    # ------------------------------------------------------------------------------
    for n in range(n_rf_plots):
        plot_rf(
            ax_rf_hat[n],
            rf_data_hat[n],
            vmin=vmin_rf,
            vmax=-vmin_rf,
            aspect="auto",
            title=f"estimated rf data tx {n}",
        )
        plot_rf(
            ax_rf_true[n],
            rf_data[n],
            vmin=vmin_rf,
            vmax=-vmin_rf,
            aspect="auto",
            title=f"true rf data tx {n}",
        )
    # ------------------------------------------------------------------------------
    # Finish up
    # ------------------------------------------------------------------------------
    fig.suptitle(
        f"Step {str(step).zfill(6)}"
        f"\nsound speed: {repar_fn['sound_speed'](opt_vars['sound_speed']):.2f} [m/s]"
        f"\n{repar_fn['attenuation'](opt_vars['attenuation'])[0]:.2e} [dB/cm/MHz]"
        f"\nrun id: {run_id}"
    )
    all_axes = np.array([ax_image, ax_residual, ax_beamformed, *ax_rf_true, *ax_rf_hat])

    plt.tight_layout()

    # ------------------------------------------------------------------------------
    # Save the figure
    # ------------------------------------------------------------------------------
    save_dir = working_dir / "overview"
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / f"step_{str(step).zfill(6)}.png", dpi=256)
    # plt.show()
    plt.close()


def plot_loss(working_dir, eval_loss, steps):
    """Creates a line plot of the loss."""
    fig, ax = plt.subplots(1, 1)
    if np.max(eval_loss) / np.min(eval_loss) > 20:
        ax.semilogy(steps, eval_loss, "r-o")
    else:
        ax.plot(steps, eval_loss, "r-o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    # Ensure the y-axis starts at 0
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1]])

    plt.tight_layout()
    plt.savefig(working_dir / f"loss.png", dpi=256)
    # plt.show()
    plt.close()


def plot_rf_error(working_dir, step, rf_data, rf_data_hat, ax_min=150):
    n_tx = rf_data.shape[0]
    fig, axes = plt.subplots(1, n_tx, figsize=(7, 5))
    if n_tx == 1:
        axes = np.array([axes])

    error = np.abs(rf_data - rf_data_hat)
    vmax = np.std(rf_data[:, ax_min:]) * 4

    for n in range(n_tx):
        plot_rf(
            axes[n],
            error[n],
            vmin=0,
            vmax=vmax,
            title=f"absolute error tx {n}\nscale [0, {vmax:.4f}]",
        )

    plt.tight_layout()
    error_dir = working_dir / "rf_error"
    error_dir.mkdir(exist_ok=True)
    plt.savefig(error_dir / f"rf_error_{str(step).zfill(6)}.png", dpi=256)
    plt.close()


def to_value(x):
    """Converts a hdf5 group of size 1 to a scalar."""
    return x.item() if x.size == 1 else x


# endregion
# ==================================================================================
# Configure paths
# ==================================================================================
# region
def find_data_root(start_dir: Path, depth: int = 3):
    """Move up in the directory structure until a folder with the name
    `VERASONICS_ROOT.txt` is found. This folder is assumed to be the root of the
    Verasonics data. If the folder is not found, an error is raised."""

    assert 0 <= depth < 10, "depth must be between 0 and 100."

    path = Path(start_dir)
    root_marker = "VERASONICS_ROOT.txt"

    current_depth = 0

    while True:
        # Find all files with the name `VERASONICS_ROOT.txt` in the current directory
        # tree
        found_matches = list(path.rglob(root_marker))

        # If an occurrence is found, return the path
        if len(found_matches) > 0:
            return Path(found_matches[0].parent)

        # If we are already at the root of the file system or we have exhausted the max,
        # depth, raise an error
        if path.parent == path or current_depth >= depth:
            raise RuntimeError(f"{root_marker} not found in the directory structure.")

        # Move up one level in the directory tree
        path = path.parent

        # Increment the current depth
        current_depth += 1


def get_call_stack_paths():
    """Collects all unique file paths in the call stack."""
    stack = inspect.stack()
    paths = set()
    for frame in stack:
        paths.add(frame.filename)

    # Turn paths to Path objects
    paths = [Path(p) for p in paths]

    return paths
