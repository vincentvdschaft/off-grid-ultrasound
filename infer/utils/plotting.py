import pickle
from math import ceil
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, vmap
from jaxus import plot_beamformed, plot_rf
from tqdm import tqdm

from .grids import get_pixel_grid_from_lims
from .splatting import splat_gaussians_2d


def plot_optimizer_rf(ax, rf_data, ax_min, start_sample=0):
    """
    Plots the RF data to an axis with a line indicating the minimum of the axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot to.
    rf_data : np.ndarray
        The RF data to plot of shape (n_ax, n_ch) starting at sample `start_sample`.
    ax_min : float
        A horizontal line is drawn at this value.
    start_sample : int, optional
        The sample index corresponding to the first row of `rf_data`. Defaults to 0.
    """
    assert isinstance(ax, plt.Axes), "The axis must be a matplotlib axis."
    assert isinstance(rf_data, np.ndarray), "The RF data must be a numpy array."
    assert rf_data.ndim == 2, "The RF data must be 2D."

    plot_rf(ax, rf_data, start_sample=start_sample)

    # Draw a horizontal line at the minimum of the axis
    ax.axhline(ax_min, color="red", linestyle="--", label="Minimum of the axis")


def get_kernel_image(
    scatterer_pos_m,
    scatterer_amplitudes,
    pixel_size,
    xlim=(0.0, 1.0),
    ylim=None,
    zlim=(0.0, 1.0),
    radius=1e-3,
    falloff_power=3,
):
    """
    Generates an image from the scatterer positions and amplitudes using a kernel based
    approach. The function creates a grid of pixels and splats the scatterer amplitudes
    onto the pixels using a kernel.

    Parameters
    ----------
    xlim, ylim, zlim : tuple
        The limits of the image in meters. One of these can be set to a single value to
        generate a 2D image.
    scatterer_pos_m : array
        The positions of the scatterers in meters of shape (n_scat, n_dim).
    scatterer_amplitudes :   array
        The amplitudes of the scatterers.
    pixel_size : float
        The size of the pixels in the image.
    radius : float
        The radius of the kernel.
    falloff_power : float
        The power of the falloff of the kernel.

    Returns
    -------
    output : np.array
        The generated image.
    """
    assert (
        scatterer_pos_m.ndim == 2
    ), "The scatterer positions array must be of shape (n_scat, n_dim)."
    assert scatterer_amplitudes.ndim == 1, "The scatterer amplitudes array must be 1D."
    assert scatterer_pos_m.shape[1] in (
        2,
        3,
    ), "The scatterer positions must be 2D or 3D."

    assert not (
        scatterer_pos_m.shape[1] == 3 and ylim is None
    ), "The y limits must be set for 3D images."

    assert isinstance(
        xlim, (tuple, list, int, float)
    ), "The x limits must be a tuple, list, int or float."
    assert (
        isinstance(ylim, (tuple, list, int, float)) or ylim is None
    ), "The y limits must be a tuple, list, int, float, or None."
    assert isinstance(
        zlim, (tuple, list, int, float)
    ), "The z limits must be a tuple, list, int or float."

    # Turn the limits into tuples if they are not already
    xlim = (xlim, xlim) if isinstance(xlim, (int, float)) else xlim
    if ylim is not None:
        ylim = (ylim, ylim) if isinstance(ylim, (int, float)) else ylim
    zlim = (zlim, zlim) if isinstance(zlim, (int, float)) else zlim

    # Ensure that the limits are in the correct order
    xlim = (min(xlim), max(xlim))
    if ylim is not None:
        ylim = (min(ylim), max(ylim))
    zlim = (min(zlim), max(zlim))

    # Create a pixel grid from the limits
    n_dim = scatterer_pos_m.shape[1]
    if n_dim == 2:
        pixel_grid = get_pixel_grid_from_lims((xlim, zlim), pixel_size)
        # image = splat_gaussians_2d(
        #     pixel_grid.shape_2d,
        #     scatterer_pos_m,
        #     scatterer_amplitudes,
        #     tuple(pixel_grid.extent_m),
        #     radius=radius,
        #     batch_size=1024,
        # )
        image = splat_pixels(
            pixel_pos=pixel_grid.pixel_positions,
            scat_pos=scatterer_pos_m,
            scat_amp=scatterer_amplitudes,
            radius=radius,
            falloff_power=falloff_power,
        )
    else:
        pixel_grid = get_pixel_grid_from_lims((xlim, ylim, zlim), pixel_size)

        # Do the splatting
        image = splat_pixels(
            pixel_pos=pixel_grid.pixel_positions,
            scat_pos=scatterer_pos_m,
            scat_amp=scatterer_amplitudes,
            radius=radius,
            falloff_power=falloff_power,
        )

    return image, pixel_grid


def splat_pixels(pixel_pos, scat_pos, scat_amp, radius, falloff_power=2):
    """Splats the scatterer amplitudes onto the pixels using a kernel based approach.

    Parameters
    ----------
    pixel_pos : np.ndarray
        The positions of the pixels of shape (..., 2) or (..., 3).
    scat_pos : np.ndarray
        The positions of the scatterers of shape (..., 2) or (..., 3).
    scat_amp : np.ndarray
        The amplitudes of the scatterers of shape (...,).
    radius : float
        The radius of the kernel.
    falloff_power : float
        The power of the falloff of the kernel.

    Returns
    -------
    image : np.ndarray
        The generated image of the same shape as `pixel_pos` except for the last dimension.
    """

    output_shape = pixel_pos.shape[:-1]
    pixel_pos_flat = pixel_pos.reshape(-1, pixel_pos.shape[-1])

    assert (
        pixel_pos.shape[-1] == scat_pos.shape[-1]
    ), "The pixel and scatterer positions must have the same number of dimensions."

    def fn(x):
        plateau_width = 0.45
        slope_width = 0.85
        low_width = 0.5
        x = jnp.abs(x)
        y = jnp.array(0.0)

        plateau_min = 0.2

        x_plateau = x / plateau_width
        y_plateau = 1 - x_plateau**2 * plateau_min

        slope_min = 0.25
        start = plateau_width
        x_slope = (x - start) / slope_width
        a = jnp.log10(slope_min)

        y_slope = jnp.exp(-((x_slope))) * 0.4

        x_low = (x - plateau_width - slope_width) / low_width
        # y_low = slope_min * jnp.power(10, -x_low * 0.5) * 0.2
        y_low = 0.2

        y = jnp.where(x <= plateau_width, y_plateau, y)
        y = jnp.where(
            jnp.logical_and(x > plateau_width, x <= plateau_width + slope_width),
            y_slope,
            y,
        )
        y = jnp.where(
            jnp.logical_and(
                x > plateau_width + slope_width,
                x <= plateau_width + slope_width + low_width,
            ),
            y_low,
            y,
        )
        return y

    def pixel_brightness(pix_pos, scat_pos, scat_amp):
        """Computes the brightness of a single pixel in response to all scatterers."""
        r = jnp.linalg.norm(pix_pos[None] - scat_pos, axis=-1) / radius

        # return jnp.min(r)
        return np.sum(scat_amp * jnp.exp(-0.5 * (r**falloff_power)))

    # Vectorize the pixel brightness function to manage multiple pixels
    pixel_brightness = jit(vmap(pixel_brightness, in_axes=(0, None, None)))

    # Compute the number number of pixels rounded up to the nearest chunk size
    num_pixels = pixel_pos_flat.shape[0]
    chunk_size = 1024 * 32
    n_pixels_rounded_up = num_pixels + chunk_size - (num_pixels % chunk_size)

    # Iterate over the chunks of pixels and compute the pixel intensities
    intensities = []
    for i in tqdm(range(0, n_pixels_rounded_up, chunk_size), desc="Generating image"):
        intensities.append(
            pixel_brightness(
                pixel_pos_flat[i : i + chunk_size],
                scat_pos,
                scat_amp,
            )
        )

    # Concatenate the pixel intensities and reshape the image to its original shape
    image = jnp.concatenate(intensities)
    image = jnp.reshape(image, output_shape)

    return image


def save_figure(path, fig):
    """
    Saves a figure to a .pkl file.

    Parameters
    ----------
    path : str
        The path to save the figure to.
    fig : plt.Figure
        The figure to save.
    """
    path = Path(path)

    assert isinstance(fig, plt.Figure), "The figure must be a matplotlib figure."
    assert path.suffix == ".pkl", "The figure must be saved as a .pkl file."

    # Save the figure using pickle
    with open(path, "wb") as file:
        pickle.dump(fig, file)


def show_saved_figure(path, save_path=None, show=True):
    """
    Shows a saved figure.

    Parameters
    ----------
    path : str
        The path to the saved figure.
    save_path : str, optional
        The path to save the figure to. Defaults to None.
    """
    path = Path(path)
    save_path = Path(save_path) if save_path is not None else None

    assert path.suffix == ".pkl", "The figure must be saved as a .pkl file."

    # Load the figure using pickle
    with open(path, "rb") as file:
        fig = pickle.load(file)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        fig.show()
        plt.show()
    return fig


def to_mm_int(x, _):
    return f"{int(x*1e3)}"


def plot_solution_3d(
    fig,
    scat_pos_m,
    scat_amp,
    probe_geometry_m,
    active_element_idx,
    tx_apodizations_tx,
    add_subplot_arg=111,
    vmin=None,
    vmax=None,
):
    """Creates a 3D plot of the scatterer positions and amplitudes, the probe geometry
    and the transmit apodizations."""

    # Plot the data in 3D
    ax = fig.add_subplot(add_subplot_arg, projection="3d", computed_zorder=False)
    ax.view_init(elev=20, azim=20)
    ax.scatter(
        scat_pos_m[:, 0],
        scat_pos_m[:, 1],
        scat_pos_m[:, 2],
        c=scat_amp if scat_amp is not None else "w",
        cmap="gray",
        s=0.5,
    )

    ax.scatter(
        probe_geometry_m[:, 0],
        probe_geometry_m[:, 1],
        probe_geometry_m[:, 2],
        c=np.arange(probe_geometry_m.shape[0]),
        cmap="viridis",
        s=2,
    )

    # Plot the transmit apodizations
    if np.sum(tx_apodizations_tx > 0) >= np.sum(probe_geometry_m > 0):
        tx_apodizations_tx = None
    if tx_apodizations_tx is not None:

        ax.scatter(
            active_element_idx.reshape(-1, 3)[:, 0],
            active_element_idx.reshape(-1, 3)[:, 1],
            tx_apodizations_tx.reshape(-1, 1) * 0.3e-3,
            c="g",
            s=2,
            vmin=vmin,
            vmax=vmax,
        )

    # Disable gray bottom and walls of the plot
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    mm_formatter = plt.FuncFormatter(to_mm_int)
    ax.xaxis.set_major_formatter(mm_formatter)
    ax.yaxis.set_major_formatter(mm_formatter)
    ax.zaxis.set_major_formatter(mm_formatter)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    # Turn off the grid
    ax.grid(False)

    # Enable all spines
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Set view
    ax.view_init(elev=20, azim=20)

    return ax


def imshow_centered(ax, arr, extent, *args, indexing="xy", **kwargs):
    """
    Identical to `ax.imshow` but it ensures that coordinates are at the center of the
    pixels by extending the extent of the image by half a pixel size on both sides.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot to.
    arr : np.ndarray
        The array to plot.
    extent : tuple
        The extent of the image.
    *args, **kwargs
        Additional arguments to pass to `ax.imshow`.
    """
    assert isinstance(ax, plt.Axes), "The axis must be a matplotlib axis."
    # assert isinstance(arr, np.ndarray), "The array must be a numpy array."
    extent = tuple(extent)
    assert len(extent) == 4, "The extent must be a tuple of length 4."

    xlim = (min(extent[0], extent[1]), max(extent[0], extent[1]))
    ylim = (min(extent[2], extent[3]), max(extent[2], extent[3]))

    # Compute the pixel size
    dx = (xlim[1] - xlim[0]) / arr.shape[1]
    dy = (ylim[1] - ylim[0]) / arr.shape[0]

    # Compute the new extent
    new_extent = (
        xlim[0] - dx / 2,
        xlim[1] + dx / 2,
        ylim[0] - dy / 2,
        ylim[1] + dy / 2,
    )

    if extent[0] > extent[1]:
        new_extent = (new_extent[1], new_extent[0], new_extent[2], new_extent[3])

    if extent[2] > extent[3]:
        new_extent = (new_extent[0], new_extent[1], new_extent[3], new_extent[2])

    if indexing == "ij":
        arr = arr.T

    # Plot the image
    ax.imshow(arr, extent=new_extent, *args, **kwargs)


def prune_scatterers(
    scatterer_positions, scatterer_amplitudes, threshold_fraction, n_max
):
    """
    Prunes the scatterers based on their amplitudes.

    Parameters
    ----------
    scatterer_positions : np.ndarray
        The positions of the scatterers.
    scatterer_amplitudes : np.ndarray
        The amplitudes of the scatterers.
    threshold_fraction : float
        The fraction of the maximum amplitude to keep.
    n_max : int
        The maximum number of scatterers to keep.

    Returns
    -------
    pruned_positions : np.ndarray
        The pruned positions of the scatterers.
    pruned_amplitudes : np.ndarray
        The pruned amplitudes of the scatterers.
    """

    assert (
        isinstance(threshold_fraction, float) and 0 <= threshold_fraction <= 1
    ), "The threshold fraction must be a float between 0 and 1."
    assert (
        isinstance(n_max, int) and n_max > 0
    ), "The maximum number must be a positive integer."

    maxval = np.percentile(scatterer_amplitudes, 99.99)

    scatterer_amplitudes = np.abs(scatterer_amplitudes) / maxval

    mask = scatterer_amplitudes > threshold_fraction
    if np.sum(mask) > 0:
        pruned_positions = scatterer_positions[mask]
        pruned_amplitudes = scatterer_amplitudes[mask]

    if pruned_amplitudes.shape[0] > n_max:
        idx = np.argsort(pruned_amplitudes)
        pruned_positions = pruned_positions[idx]
        pruned_amplitudes = pruned_amplitudes[idx]
        pruned_positions = pruned_positions[-n_max:]
        pruned_amplitudes = pruned_amplitudes[-n_max:]

    return pruned_positions, pruned_amplitudes


def plot_overview(
    kernel_image,
    kernel_pixel_grid,
    das_image,
    das_pixel_grid,
    opt_vars,
    forward_settings,
    save_path=None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax_das = axes[0]
    ax_kernel = axes[1]

    plot_beamformed_old(ax_das, das_image, das_pixel_grid.extent_m_zflipped)
    plot_beamformed_old(ax_kernel, kernel_image, kernel_pixel_grid.extent_m_zflipped)

    gain = opt_vars.gain
    probe_geometry = forward_settings.probe_geometry_m
    ax_kernel.stem(probe_geometry[:, 0], gain * (-2e-3), markerfmt="ro", linefmt="r-")

    return fig


#
# def plot_loss_curve(loss_curve)


def stamp_figure(fig, str):
    """Stamps a string on a figure in the bottom left corner."""
    fig.text(0.01, 0.01, str, fontsize=8, color="gray", ha="left", va="bottom")
    return fig
