import socket
from functools import partial
from math import ceil
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import jit, vmap
from jaxus import compute_lensed_travel_time
from jaxus.utils import log
from scipy.optimize import minimize
from scipy.sparse import csr_array

# Import nlm denoiser
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm

from .utils import cache_outputs


def admm_inverse_beamform(
    rf_data: np.ndarray,
    probe_geometry: np.ndarray,
    t0_delays: np.ndarray,
    tx_apodization: np.ndarray,
    initial_time: float,
    t_peak: float,
    sound_speed: Union[float, int],
    waveform_samples: np.ndarray,
    sampling_frequency: Union[float, int],
    carrier_frequency: Union[float, int],
    pixel_positions: np.ndarray,
    grid_shape: tuple,
    red_iterations: int = 1,
    nlm_h_parameter: float = 0.05,
    ax_min: int = 0,
    f_number: float = 0.5,
    beta: Union[float, int] = 1000,
    mu: Union[float, int] = 5,
    max_n_iterations: int = 100,
    method: str = "admm",
    epsilon: float = 1e-5,
    chunk_size: int = 1024,
    return_phi: bool = False,
):
    """Performs denoising based inverse beamforming using the ADMM algorithm as
    described in the paper 'Inverse Problem of Ultrasound Beamforming with
    Denoising-Based Regularized Solutions' by Goudarzi et al. on multiple
    transmits and compounds the results.

    This function implements three different approaches, set by the `method` parameter:
    - ADMM - Alternating Direction Method of Multipliers
    - RED - Regularization by Denoising
            from the paper: "The little engine that could: Regularization by denoising
                             (RED)"
    - PNP - Plug-and-Play denoising.
            from the paper: "Plug-and-play priors for model based reconstruction"

    Parameters
    ----------
    rf_data : np.ndarray
        The raw RF data in shape `(n_ax, n_el)`. All samples are
        used in the optimization.
    probe_geometry : np.ndarray
        The geometry of the probe in shape `(n_el, 2)`.
    t0_delays : np.ndarray
        The t0 delays in shape `(n_el,)`.
    initial_time : np.ndarray
        The time at which axial sample 0 was recorded by all elements.
    sound_speed : Union[float, int]
        The speed of sound in m/s.
    waveform_samples : np.ndarray
        The waveform samples in shape `(n_samples,)`.
    sampling_frequency : Union[float, int]
        The sampling frequency in Hz.
    carrier_frequency : Union[float, int]
        The carrier frequency in Hz.
    pixel_positions : np.ndarray
        The positions of the pixels in shape `(n_x, n_z, 2)`.
    red_iterations : int, default=1
        The number of iterations to perform in the RED algorithm. This parameter is
        only used if the `method` parameter is set to 'red'.
    nlm_h_parameter : float, default=0.05
        The h parameter in the non-local means denoising algorithm. This parameter
        controls the amount of smoothing applied to the image in the denoising step.
        (Only used if the method is set to 'red' or 'pnp').
    ax_min: int, default=0
        The starting index of the axial samples. This parameter can be used to
        discard the first few samples in the RF data. Note that the `rf_data`
        parameter should still contain the full RF data.
    f_number : float, default=0.5
        The f-number to apply in computing the forward model matrix.
    beta : Union[float, int], default=1000
        The beta parameter in the ADMM algorithm. This parameter is used to balance
        the data fidelity term and the regularization term.
    mu : Union[float, int], default=5
        The mu parameter in the ADMM algorithm. This parameter is used to balance
        the regularization term and the constraint term. The paper recommends small
        values for mu when using the ADMM method, and large values for the RED method.
        (Not used in the PNP method.)
    max_n_iterations : int, default=100
        The maximum number of iterations to perform in the ADMM algorithm.
    method : str, default='admm'
        The method to use for the inverse beamforming. Choose from 'admm', 'pnp', or
        'red'.
    epsilon : float, default=1e-5
        The convergence criterion. The algorithm stops when the relative change in cost
        is less than this value.
    chunk_size : int, default=1024
        The number of samples to compute at once when constructing the Phi matrix. Set
        to a lower value if you run out of memory.
    return_phi : bool, default=False
        If True, the function returns the Phi matrix in addition to the compounded
        image.
    """

    log.info("Starting ADMM inverse beamforming")

    # ==================================================================================
    # Input checking
    # ==================================================================================
    assert isinstance(t_peak, (int, float)), "t_peak must be an integer or a float."
    # assert pixel_positions.ndim == 2, "pixel_positions must be a 2D array."
    assert (
        pixel_positions.shape[-1] == 2
    ), "pixel_positions must have shape (n_x, n_z, 2)."
    assert rf_data.ndim == 2, "rf_data must be a 2D array of shape (n_ax, n_el)."
    n_ax, n_el = rf_data.shape

    assert probe_geometry.ndim == 2, "probe_geometry must be a 2D array."
    assert probe_geometry.shape == (
        n_el,
        2,
    ), "probe_geometry must have shape (n_el, 2)."
    assert t0_delays.shape == (n_el,), "t0_delays must have shape (n_el,)."
    assert waveform_samples.ndim == 1, "waveform_samples must be a 1D array."
    # ----------------------------------------------------------------------------------
    # Ensure that all these parameters are positive numbers
    # ----------------------------------------------------------------------------------
    numbers = (
        ("sound_speed", sound_speed),
        ("sampling_frequency", sampling_frequency),
        ("carrier_frequency", carrier_frequency),
        ("t_peak", t_peak),
        ("mu", mu),
        ("beta", beta),
        ("max_n_iterations", max_n_iterations),
        ("epsilon", epsilon),
        ("red_iterations", red_iterations),
        ("chunk_size", chunk_size),
    )
    for name, value in numbers:
        assert (
            isinstance(value, (int, float)) and value > 0
        ), f"{name} must be a positive number."

    # Turn the method to lowercase string
    method = str(method).lower()
    assert method in [
        "admm",
        "pnp",
        "red",
    ], f"Invalid method {method}. Choose 'admm', 'pnp', or 'red'."

    # TODO: Add more input checking

    n_ax, n_el = rf_data.shape
    n_x, n_z = int(grid_shape[0]), int(grid_shape[1])

    # ======================================================================================
    # Derived parameters
    # ======================================================================================

    # ==================================================================================
    # Compute hash
    # ==================================================================================
    # Compute a hash from the function inputs to uniquely identify the Phi matrix
    # All elements must be turned into immutable types (e.g. tuples) to be uniquely
    # hashable.
    phi_hash = hash(
        (
            n_ax,
            n_el,
            hash(pixel_positions.shape),
            hash(tuple(np.array(pixel_positions)[:100].flatten())),
            hash(tuple(np.array(probe_geometry).flatten())),
            hash(tuple(np.array(t0_delays.flatten()))),
            hash(tuple(np.array(tx_apodization.flatten()))),
            hash(float(initial_time)),
            hash(float(sound_speed)),
            hash(tuple(np.array(waveform_samples.flatten()))),
            hash(float(sampling_frequency)),
            hash(float(carrier_frequency)),
        )
    )

    # ==================================================================================
    # Turn values into jax arrays
    # ==================================================================================
    # region
    probe_geometry = jnp.array(probe_geometry, dtype=jnp.float32)
    t0_delays = jnp.array(t0_delays, dtype=jnp.float32)
    tx_apodization = jnp.array(tx_apodization, dtype=jnp.float32)
    initial_time = jnp.array(initial_time, dtype=jnp.float32)
    sound_speed = jnp.array(sound_speed, dtype=jnp.float32)
    waveform_samples = jnp.array(waveform_samples, dtype=jnp.float32)
    sampling_frequency = jnp.array(sampling_frequency, dtype=jnp.float32)
    carrier_frequency = jnp.array(carrier_frequency, dtype=jnp.float32)
    pixel_positions = jnp.array(pixel_positions, dtype=jnp.float32)
    rf_data = jnp.array(rf_data, dtype=jnp.float32)
    tx_apodization_iszero = jnp.array(tx_apodization == 0, dtype=jnp.float32)

    # endregion

    # ==================================================================================
    # Define functions
    # ==================================================================================
    # region

    # --------------------------------------------------------------------------------------
    # Define functions to compute the forwFard model matrix PHI
    # --------------------------------------------------------------------------------------
    # region

    # endregion
    # endregion

    # ==================================================================================
    # Compute or load the Phi matrix
    # ==================================================================================
    _construct_phi_cached = cache_outputs("temp/cache")(_construct_phi)
    # _construct_phi_cached = _construct_phi

    phi = _construct_phi_cached(
        n_ax,
        n_el,
        jnp.array(pixel_positions),
        sampling_frequency,
        initial_time,
        probe_geometry,
        t0_delays,
        sound_speed,
        tx_apodization_iszero,
        f_number,
        t_peak,
        chunk_size=1024,
        ax_min=ax_min,
    )

    # ==================================================================================
    # Run the ADMM algorithm
    # ==================================================================================

    last_cost = np.inf

    # ----------------------------------------------------------------------------------
    # Initialize variables
    # ----------------------------------------------------------------------------------
    u = np.random.randn(pixel_positions.shape[0]) * 1e-3
    v = np.random.randn(pixel_positions.shape[0]) * 1e-3
    lam = np.zeros(pixel_positions.shape[0])
    y = rf_data[ax_min:].flatten()

    for n in range(max_n_iterations):

        # ------------------------------------------------------------------------------
        # Update u
        # ------------------------------------------------------------------------------
        # Run the L-BFGS optimization to minimize the u variable
        u = minimize(
            method="L-BFGS-B",
            fun=partial(_u_func, v=v, lam=lam, phi=phi, y=y, beta=beta),
            x0=u,
            jac=partial(_gradient, v=v, lam=lam, phi=phi, y=y, beta=beta),
            options={"maxiter": 100, "disp": False},
            tol=1e-7,
        ).x

        if np.any(np.isnan(u)):
            log.error("NaNs in u.")
            break

        # ------------------------------------------------------------------------------
        # Update v
        # ------------------------------------------------------------------------------
        # The way to update v depends on the method used. The paper describes three
        # methods: ADMM, PnP, and RED.
        # ADMM - Soft thresholding
        # PnP - Non-local means denoising
        # RED - Iterative non-local means denoising based on update rule
        # ------------------------------------------------------------------------------
        if method == "admm":
            v = _softthresh(u + lam / beta, mu / beta)
        elif method == "pnp":
            # Define the input to the NLM denoiser
            denoise_input = v + lam / beta

            sigma = estimate_sigma(denoise_input.reshape(n_x, n_z))

            # Perform the denoising
            v = denoise_nl_means(
                denoise_input.reshape(n_x, n_z),
                h=nlm_h_parameter * sigma,
                fast_mode=False,
            ).flatten()

        elif method == "red":
            # Initialize
            z = v.reshape(n_x, n_z)
            for _ in range(red_iterations):
                sigma = estimate_sigma(z.reshape(n_x, n_z))
                # See equation (15) in the paper
                z = (
                    mu
                    / (mu + beta)
                    * denoise_nl_means(
                        z, h=nlm_h_parameter * sigma, fast_mode=False, patch_distance=21
                    )
                    + beta / (mu + beta) * u.reshape(n_x, n_z)
                    + 1 / (mu + beta) * lam.reshape(n_x, n_z)
                )
            v = z.flatten()

        # ------------------------------------------------------------------------------
        # Update lambda
        # ------------------------------------------------------------------------------
        lam = lam + beta * (u - v)

        # ------------------------------------------------------------------------------
        # Report progress
        # ------------------------------------------------------------------------------
        # Compute the cost function
        current_cost = _cost_function(u, v, lam, phi, y, beta, mu)
        # Compute the criterion as the relative change in cost
        criterion = (jnp.abs(last_cost - current_cost) + 1e-9) / current_cost
        # Update the last cost
        last_cost = current_cost

        # Format the cost values as colored strings
        iteration_str = log.yellow(format(n, "<4d"))
        current_criterion_str = log.red(format(criterion, "<16.2e"))
        ufunc_loss_str = log.red(format(_u_func(u, v, lam, phi, y, beta), "<16.2f"))
        cost_str = log.red(format(current_cost, "<16.2f"))

        if criterion < epsilon and n > 2:
            break

        # Report the progress
        log.info(
            f"Iteration {iteration_str}\t"
            f"ufunc loss: {ufunc_loss_str}\t"
            f"criterion: {current_criterion_str}\t"
            f"cost: {cost_str}"
        )
    if return_phi:
        return v, phi
    else:
        return v


def admm_inverse_beamform_compounded(
    rf_data: np.ndarray,
    probe_geometry: np.ndarray,
    t0_delays: np.ndarray,
    tx_apodizations: np.ndarray,
    initial_times: np.ndarray,
    t_peak: np.ndarray,
    sound_speed: Union[float, int],
    waveform_samples: np.ndarray,
    sampling_frequency: Union[float, int],
    carrier_frequency: Union[float, int],
    pixel_positions: np.ndarray,
    grid_shape: tuple,
    red_iterations: int = 1,
    nlm_h_parameter: float = 0.05,
    ax_min: int = 0,
    f_number: float = 0.5,
    beta: Union[float, int] = 1000,
    mu: Union[float, int] = 5,
    max_n_iterations: int = 100,
    method: str = "admm",
    epsilon: float = 1e-5,
    chunk_size: int = 1024,
    return_phi: bool = False,
    **kwargs,
):
    """Performs denoising based inverse beamforming using the ADMM algorithm as
    described in the paper 'Inverse Problem of Ultrasound Beamforming with
    Denoising-Based Regularized Solutions' by Goudarzi et al. on multiple
    transmits and compounds the results.

    This function implements three different approaches, set by the `method` parameter:
    - ADMM - Alternating Direction Method of Multipliers
    - RED - Regularization by Denoising
            from the paper: "The little engine that could: Regularization by denoising
                             (RED)"
    - PNP - Plug-and-Play denoising.
            from the paper: "Plug-and-play priors for model based reconstruction"

    Parameters
    ----------
    rf_data : np.ndarray
        The raw RF data in shape `(n_tx, n_ax, n_el)`. All samples are
        used in the optimization.
    probe_geometry : np.ndarray
        The geometry of the probe in shape `(n_el, 2)`.
    t0_delays : np.ndarray
        The t0 delays in shape `(n_tx, n_el)`.
    initial_times : np.ndarray
        The time at which axial sample 0 was recorded by all elements for each transmit.
        of shape `(n_tx,)`.
    sound_speed : Union[float, int]
        The speed of sound in m/s.
    waveform_samples : np.ndarray
        The waveform samples in shape `(n_tx, n_samples,)`.
    sampling_frequency : Union[float, int]
        The sampling frequency in Hz.
    carrier_frequency : Union[float, int]
        The carrier frequency in Hz.
    pixel_positions : np.ndarray
        The positions of the pixels in shape `(n_x, n_z, 2)`.
    red_iterations : int, default=1
        The number of iterations to perform in the RED algorithm. The paper notes that 1
        is usually enough. This parameter is only used if the `method` parameter is set
        to 'red'.
    nlm_h_parameter : float, default=0.05
        The h parameter in the non-local means denoising algorithm. This parameter
        controls the amount of smoothing applied to the image in the denoising step.
        (Only used if the method is set to 'red' or 'pnp').
    ax_min: int, default=0
        The starting index of the axial samples. This parameter can be used to
        discard the first few samples in the RF data. Note that the `rf_data`
        parameter should still contain the first samples.
    f_number : float, default=0.5
        The f-number to apply in computing the forward model matrix.
    beta : Union[float, int], default=1000
        The beta parameter in the ADMM algorithm. This parameter is used to balance
        the data fidelity term and the regularization term.
    mu : Union[float, int], default=5
        The mu parameter in the ADMM algorithm. This parameter is used to balance
        the regularization term and the constraint term. The paper recommends small
        values for mu when using the ADMM method, and large values for the RED method.
        (Not used in the PNP method.)
    max_n_iterations : int, default=100
        The maximum number of iterations to perform in the ADMM algorithm.
    method : str, default='admm'
        The method to use for the inverse beamforming. Choose from 'admm', 'pnp', or
        'red' (not case sensitive). The three methods are:
        ADMM - Alternating Direction Method of Multipliers;
        RED - Regularization by Denoising;
        PNP - Plug-and-Play;
    epsilon : float, default=1e-5
        The convergence criterion. The algorithm stops when the relative change in cost
        is less than this value.
    chunk_size : int, default=1024
        The number of samples to compute at once when constructing the Phi matrix. Set
        to a lower value if you run out of memory.
    return_phi : bool, default=False
        If True, the function returns the Phi matrix in addition to the compounded
        image.

    """

    # Check if the rf_data is a 3D array
    assert isinstance(rf_data, (np.ndarray, jnp.ndarray))

    if rf_data.ndim == 5:
        rf_data = rf_data[0]
        rf_data = rf_data[..., 0]
    n_tx, n_ax, n_el = rf_data.shape

    images = []

    if return_phi:
        phis = []

    # pixel_positions = pixel_positions.reshape(_deduce_pixel_grid_shape(pixel_positions))

    # Cache the admm_inverse_beamform function to save computation time
    admm_inverse_beamform_cached = cache_outputs("temp/cache/admm_inverse_beamform")(
        admm_inverse_beamform
    )
    # admm_inverse_beamform_cached = admm_inverse_beamform
    # admm_inverse_beamform_cached = admm_inverse_beamform

    for tx in range(n_tx):
        output = admm_inverse_beamform_cached(
            rf_data=rf_data[tx],
            probe_geometry=probe_geometry,
            t0_delays=t0_delays[tx],
            tx_apodization=tx_apodizations[tx],
            initial_time=float(initial_times[tx]),
            t_peak=float(t_peak[tx]),
            sound_speed=sound_speed,
            waveform_samples=waveform_samples[tx],
            sampling_frequency=sampling_frequency,
            carrier_frequency=carrier_frequency,
            pixel_positions=pixel_positions,
            grid_shape=grid_shape,
            red_iterations=red_iterations,
            nlm_h_parameter=nlm_h_parameter,
            ax_min=ax_min,
            f_number=f_number,
            beta=beta,
            mu=mu,
            max_n_iterations=max_n_iterations,
            method=method,
            epsilon=epsilon,
            chunk_size=chunk_size,
            return_phi=return_phi,
        )
        if return_phi:
            image, phi = output
            phis.append(phi)
        else:
            image = output

        images.append(image)

    # Stack the images along the first axis
    images = np.stack(images, axis=0)

    # Compound the images
    compounded_image = np.mean(images, axis=0)

    if return_phi:
        return compounded_image, phis
    else:
        return compounded_image


def _deduce_pixel_grid_shape(pixel_positions):
    """Deduce the shape of the pixel grid from the pixel positions."""
    if pixel_positions.ndim == 3:
        return pixel_positions.shape
    elif pixel_positions.ndim == 2:
        n_x = np.unique(pixel_positions[:, 0]).size
        n_z = np.unique(pixel_positions[:, 1]).size
        return n_x, n_z, 2
    else:
        raise ValueError("Invalid shape for pixel_positions.")


def _softthresh(x, mu):
    """Soft thresholding function.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    mu : float
        The threshold value.

    Returns
    -------
    np.ndarray
        The soft thresholded array.
    """
    return np.maximum(np.abs(x) - mu, 0) * np.sign(x)


def _u_func(u, v, lam, phi, y, beta):
    """Computes the minimization function to minimize the u variable in the ADMM
    algorithm."""
    return (
        1 / 2 * np.linalg.norm(y - phi @ u) ** 2
        + beta / 2 * np.linalg.norm(u - v + lam / beta) ** 2
    )


def _gradient(u, v, lam, phi, y, beta):
    """Computes the gradient of the minimization function with respect to the u
    variable explicitly. This speeds up the optimization by a lot, because the
    LBFGS solver does not have to compute the gradient numerically."""
    return -phi.T @ (y - phi @ u) + beta * (u - v + lam / beta)


def _cost_function(u, v, lam, phi, y, beta, mu):
    """Computes the overall cost function that is being minimized in the ADMM
    algorithm."""
    return _u_func(u, v, lam, phi, y, beta) + mu * np.sum(np.abs(v))


def compute_tof_difference(
    ax,
    el,
    pixel_positions,
    sampling_frequency,
    initial_time,
    probe_geometry,
    t0_delays,
    sound_speed,
    tx_apodization_iszero,
    f_number,
    t_peak,
    lens_thickness=1e-3,
    lens_sound_speed=1000,
):
    """Compute the difference between the time of flight (TOF) of the pixel and the
    time the sample was recorded.

    Parameters
    ----------
    ax : int
        Sample index.
    el : int
        Element index.
    pixel_positions : jnp.ndarray
        The position of the pixel in shape `(2,)`.

    Returns
    -------
    jnp.ndarray
        The difference between the TOF of the pixel and the time the sample was
        recorded."""

    # Compute the time the sample was recorded
    t_ax = ax / sampling_frequency + initial_time

    # Compute the distance from the pixel to each element of shape (n_el,)
    # dist_to_elements = jnp.linalg.norm(pixel_positions[None] - probe_geometry, axis=1)

    # Compute the transmit the minimum time of arrival of all firing elements
    # Add a large number to the elements that are not firing to ensure they are not
    # selected as the minimum

    lensed_travel_time = vmap(
        compute_lensed_travel_time, in_axes=(0, None, None, None, None, None)
    )(
        probe_geometry,
        pixel_positions,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        1,
    )

    t_tx = jnp.min(t0_delays + lensed_travel_time + tx_apodization_iszero) + t_peak

    # Compute the tof from the pixel to the receiving element
    t_rx = lensed_travel_time[el]

    # Compute the absolute difference between the time of flight and the time the sample
    # was recorded
    return jnp.abs(t_tx + t_rx - t_ax)


@jit
def _compute_phi_row(
    ax,
    el,
    pixel_positions,
    sampling_frequency,
    initial_time,
    probe_geometry,
    t0_delays,
    sound_speed,
    tx_apodization_iszero,
    f_number,
    t_peak,
):
    """Computes the time of flight (TOF) difference for all pixels contributing to a
    single sample."""
    vmap_fn = jax.vmap(
        compute_tof_difference,
        in_axes=(None, None, 0, None, None, None, None, None, None, None, None),
    )
    tof_differences = vmap_fn(
        ax,
        el,
        pixel_positions,
        sampling_frequency,
        initial_time,
        probe_geometry,
        t0_delays,
        sound_speed,
        tx_apodization_iszero,
        f_number,
        t_peak,
    )

    # # Clip the TOF differences to the range [0, 1/sampling_frequency]. (See equation (3) in the paper.)
    tof_differences = jnp.clip(tof_differences, 0, 1 / sampling_frequency)

    # Compute f-number
    dx = jnp.abs(pixel_positions[:, 0] - probe_geometry[el, 0][None])
    dz = jnp.abs(pixel_positions[:, 1] - probe_geometry[el, 1][None])

    # Implement the f-number and hanning window rx apodization
    factor = jnp.where(dz > f_number * dx, 1.0, 0.0).astype(jnp.float32) * (
        0.5 + 0.5 * jnp.cos(jnp.pi * f_number * dx / jnp.clip(dz, 1e-3, None))
    )

    # # Find the maximum value to normalize by. (See equation (3) in the paper.)
    t_max = jnp.max(tof_differences)

    # # Normalize the TOF differences. (See equation (3) in the paper.)
    tof_differences = tof_differences / t_max

    return factor * (1 - tof_differences)


def _construct_phi(
    n_ax,
    n_el,
    pixel_positions,
    sampling_frequency,
    initial_time,
    probe_geometry,
    t0_delays,
    sound_speed,
    tx_apodization_iszero,
    f_number,
    t_peak,
    chunk_size=1024,
    ax_min=0,
):
    """Construct the forward model matrix PHI as a sparse matrix. The function iterates
    over chunks of RF samples and computes computes the corresponding rows of the PHI
    matrix. From these rows only the non-zero elements are kept and stored in a sparse
    matrix.

    Parameters
    ----------
    n_ax : int
        The number of samples in the RF data.
    n_el : int
        The number of elements in the probe.
    pixel_positions : jnp.ndarray
        The positions of the pixels in shape `(n_pixels, 2)`.
    chunk_size : int, optional
        The number of samples to compute at once. Defaults to 1024.

    Returns
    -------
    csr_array
        The sparse forward model matrix PHI.
    """

    # Create arrays of all unique values for the sample index and element index
    ax_vals = jnp.arange(n_ax - ax_min) + ax_min
    el_vals = jnp.arange(n_el)

    # Create a meshgrid of all combinations of sample and element indices
    ax_grid, el_grid = jnp.meshgrid(ax_vals, el_vals, indexing="ij")

    # Create flat arrays of input values
    ax_vals = ax_grid.flatten()
    el_vals = el_grid.flatten()

    # vmap the function to compute multiple rows at once
    compute_phi_chunk = jit(
        vmap(
            _compute_phi_row,
            in_axes=(0, 0, None, None, None, None, None, None, None, None, None),
        )
    )

    # Get the number fo samples and chunks
    n_samples = ax_vals.shape[0]
    n_chunks = ceil(n_samples / chunk_size)

    # Initialize lists to store the rows, columns, and values of the sparse matrix
    submatrices = []

    description = "Building Phi matrix..."

    for chunk_index in tqdm(range(n_chunks), colour="red", desc=description):

        # Determine the start and end indices of the chunk
        index0 = chunk_index * chunk_size
        index1 = min((chunk_index + 1) * chunk_size, n_samples)

        # Compute the chunk
        full_row = compute_phi_chunk(
            ax_vals[index0:index1],
            el_vals[index0:index1],
            pixel_positions,
            sampling_frequency,
            initial_time,
            probe_geometry,
            t0_delays,
            sound_speed,
            tx_apodization_iszero,
            f_number,
            t_peak,
        )

        if jnp.any(jnp.isnan(full_row)):
            raise ValueError("NaNs in full_row.")

        new_csr_array = csr_array(
            full_row, shape=(index1 - index0, pixel_positions.shape[0])
        )

        submatrices.append(new_csr_array)

    phi = scipy.sparse.vstack(submatrices)
    return phi
