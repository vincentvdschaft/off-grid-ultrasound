""""""

import numpy as np
import jax.numpy as jnp
from jaxus import rf2iq
from jax import jit, vmap
from pathlib import Path


def envelope_detect(signal, center_frequency, sampling_frequency):
    """Detects the envelope of a signal.

    Parameters
    ----------
    signal : np.ndarray
        The input signal.
    center_frequency : float
        The carrier frequency.
    sampling_frequency : float
        The sampling frequency.

    Returns
    -------
    np.ndarray
        The envelope of the signal.
    """
    assert isinstance(signal, np.ndarray), "signal must be a numpy array"
    assert signal.ndim == 1, "signal must be 1D"
    assert isinstance(
        sampling_frequency, (int, float)
    ), "sampling_frequency must be a number"

    signal_iq = rf2iq(signal[None, None, :, None, None], 1e6, sampling_frequency)[
        0, 0, :, 0
    ]

    envelope = np.abs(signal_iq)

    return envelope


def find_phase(samples, center_frequency, sampling_frequency):
    """Computer the phase of the signal component at the center frequency."""
    n = samples.size
    t = np.arange(n) / sampling_frequency
    y_pos = samples * np.exp(-1j * 2 * np.pi * center_frequency * t)
    y_neg = samples * np.exp(1j * 2 * np.pi * center_frequency * t)
    return np.angle(np.sum(y_pos) + np.sum(y_neg))


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


def waveform_samples_block(waveform_samples: list, tx_waveform_indices: np.ndarray):
    """Turns a list of waveform arrays and a numpy array of transmit indices into a
    single waveform array of shape (n_tx, n_samples). The number of samples is the
    maximum number of samples in the waveform samples. The waveform_tx_indices array
    specifies which waveform is placed per row in the output waveform array.

    Parameters
    ----------
    waveform_samples : list
        A list of numpy arrays containing the waveform samples.
    tx_waveform_indices : np.ndarray
        A 1D numpy array containing the transmit indices for each waveform of shape
        `(n_tx,)`.

    Returns
    -------
    np.ndarray
        The output waveform array of shape `(n_tx, n_samples)`.
    """
    assert isinstance(waveform_samples, list), "waveform_samples must be a list"
    assert (
        isinstance(tx_waveform_indices, np.ndarray) and tx_waveform_indices.ndim == 1
    ), "waveform_tx_indices must be a 1D numpy array"
    assert all(
        isinstance(waveform, np.ndarray) and waveform.ndim == 1
        for waveform in waveform_samples
    ), "waveform_samples must contain 1D numpy arrays"

    # Get the number of transmit indices
    n_tx = tx_waveform_indices.size

    # Get the number of samples in each waveform
    n_samples = max([waveform.size for waveform in waveform_samples])

    # Initialize the output waveform array
    waveform = np.zeros((n_tx, n_samples))

    # Place the waveforms at the correct indices
    for i, tx_index in enumerate(tx_waveform_indices):
        waveform[i, : waveform_samples[tx_index].size] = waveform_samples[tx_index]

    return waveform


def get_latest_opt_vars(path):
    """Returns the path of the last file from the opt_vars folder, sorted
    alphabetically."""

    # Make the path absolute
    path = Path(path).resolve()

    # Look in the opt_vars folder
    opt_vars_folder = path / "opt_vars"

    # Check if the folder exists
    if not opt_vars_folder.exists():
        raise FileNotFoundError(f"Folder {opt_vars_folder} does not exist.")

    # Sort the files in the folder
    files = sorted(opt_vars_folder.glob("*.npz"))

    # Check if any files were found
    if not files:
        raise FileNotFoundError(f"No files found in {opt_vars_folder}.")

    # Return the last file
    return files[-1]


def get_grid(n_x, n_z, xlims, zlims):
    """Create a grid of pixel positions.

    Parameters
    ----------
    n_x : int
        The number of pixels in the x-direction.
    n_z : int
        The number of pixels in the z-direction.
    xlims : list, tuple, or np.ndarray
        The limits of the x-axis.
    zlims : list, tuple, or np.ndarray
        The limits of the z-axis.

    Returns
    -------
    np.ndarray
        The pixel positions of shape `(2, n_x * n_z)`.
    """
    assert isinstance(n_x, int), "n_x must be an integer"
    assert isinstance(n_z, int), "n_z must be an integer"
    assert isinstance(
        xlims, (list, tuple, np.ndarray)
    ), "xlims must be a list, tuple, or numpy array"
    assert isinstance(
        zlims, (list, tuple, np.ndarray)
    ), "zlims must be a list, tuple, or numpy array"
    assert len(xlims) == 2, "xlims must have length 2"
    assert len(zlims) == 2, "zlims must have length 2"
    assert xlims[0] < xlims[1], "xlims[0] must be less than xlims[1]"
    assert zlims[0] < zlims[1], "zlims[0] must be less than zlims[1]"
    assert n_x > 0, "n_x must be greater than 0"
    assert n_z > 0, "n_z must be greater than 0"

    # Create a grid of pixel positions
    x_vals = np.linspace(xlims[0], xlims[1], n_x)
    z_vals = np.linspace(zlims[0], zlims[1], n_z)
    xx, zz = np.meshgrid(x_vals, z_vals)
    pixel_pos = np.stack((xx.flatten(), zz.flatten()), axis=0)
    return pixel_pos
