from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from jaxus import use_dark_style
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit


@dataclass
class FWHMCurve:
    # The position where the FWHM was computed of shape (2,).
    target_position: np.ndarray
    # The positions in 2D space where the curve was interpolated of shape (n, 2).
    curve_positions: np.ndarray
    # The values of the curve at the curve_positions of shape (n,).
    curve_vals: np.ndarray
    # The detected center point of the curve
    center_index: int
    # The start and end indices of the FWHM along the curve of shape (2,).
    fwhm_indices: np.ndarray


def gaussian(x, amp, cen, wid):
    """Define a Gaussian function for fitting."""
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))


def compute_fwhm(
    image,
    extent,
    position,
    size,
    in_db=False,
    return_updated_point=False,
    return_curves=False,
):
    """
    Computes the full width at half maximum (FWHM) of an image.
    The function targets a point in the image and then interpolates
    horizontally and vertically to find the FWHM in both directions.

    Parameters
    ----------
    image : np.array
        The image to analyze of shape (n_x, n_y).
    extent : tuple
        The extent of the image in meters (xmin, xmax, ymin, ymax).
    position : tuple of floats
        The position of the point to analyze (x, y).
    size : float
        The size of the horizontal and vertical lines.
    in_db : bool
        Whether the image is in dB. If set to True the fwhm is found to the
        -20 dB point instead of the half maximum.

    Returns
    -------
    fwhm_x, fwhm_y : float
        The FWHM in the x and y directions.
    curve_s : list
        The interpolated horizontal and vertical lines.
    """

    # Check if the position is within the extent
    if not (
        extent[0] <= position[0] <= extent[1] and extent[2] <= position[1] <= extent[3]
    ):
        raise ValueError("Position is not within the extent.")

    assert size > 0, "Size must be positive."
    assert image.ndim == 2, "Image must be 2D."

    xmin, xmax, ymin, ymax = extent
    x = np.linspace(xmin, xmax, image.shape[0])
    y = np.linspace(ymin, ymax, image.shape[1])

    # 2D interpolation over the entire image
    interp = RegularGridInterpolator((x, y), image, bounds_error=False, fill_value=0)

    position = np.array(position)

    position = _find_point(position, interp)

    # Get the coordinates of the horizontal line
    horizontal_line_x = np.linspace(
        position[0] - size / 2, position[0] + size / 2, 2048
    )
    horizontal_line_y = np.full_like(horizontal_line_x, position[1])

    # Interpolate the horizontal line
    horizontal_line_values = interp((horizontal_line_x, horizontal_line_y))

    # Get the coordinates of the vertical line
    vertical_line_y = np.linspace(position[1] - size / 2, position[1] + size / 2, 2048)
    vertical_line_x = np.full_like(vertical_line_y, position[0])

    # Interpolate the vertical line
    vertical_line_values = interp((vertical_line_x, vertical_line_y))

    # Compute the FWHM of the horizontal and vertical lines
    fwhm_x, bounds_x = _curve_compute_fwhm(horizontal_line_values, size, in_db)
    fwhm_y, bounds_y = _curve_compute_fwhm(vertical_line_values, size, in_db)

    output = (fwhm_x, fwhm_y)

    if return_updated_point:
        output = output + (position,)

    if not return_curves:
        return output

    curves = []

    curve = FWHMCurve(
        target_position=position,
        curve_positions=np.column_stack((horizontal_line_x, horizontal_line_y)),
        curve_vals=horizontal_line_values,
        center_index=np.argmax(horizontal_line_values),
        fwhm_indices=bounds_x,
    )
    curves.append(curve)

    curve = FWHMCurve(
        target_position=position,
        curve_positions=np.column_stack((vertical_line_x, vertical_line_y)),
        curve_vals=vertical_line_values,
        center_index=np.argmax(vertical_line_values),
        fwhm_indices=bounds_y,
    )
    curves.append(curve)

    return output + (curves,)


def _find_point(position, interpolator, max_diff=0.6e-3):
    x_vals = np.linspace(position[0] - max_diff, position[0] + max_diff, 100)
    y_vals = np.linspace(position[1] - max_diff, position[1] + max_diff, 100)
    max_val = -1e6

    xx, yy = np.meshgrid(x_vals, y_vals)

    im = interpolator((xx, yy))

    row, col = np.unravel_index(np.argmax(im), im.shape)

    x = x_vals[col]
    y = y_vals[row]
    return np.array([x, y])


def _curve_compute_fwhm(values, range_width, in_db=False):
    """
    Compute the FWHM of a curve by finding the maximum value and then
    walking to the left and right until the value is half of the maximum.
    The width in pixels is then scaled to the range width.

    Parameters
    ----------
    values : np.array
        The samples of the curve (1D array).
    range_width : float
        The width of the range to scale the FWHM to.
    in_db : bool
        Whether the curve is in dB. If set to True the fwhm is found to the
        -20 dB point instead of the half maximum.

    Returns
    -------
    fwhm : float
        The FWHM of the curve.
    bounds : tuple
        The bounds of the FWHM.
    """
    max_index = np.argmax(values)
    max_value = values[max_index]
    if not in_db:
        half_max = max_value / 2
    else:
        half_max = max_value - 20

    margin = 6
    print("margin")
    # Walk to the left
    left_index = max_index
    repeats = 0
    while left_index > 0 and repeats <= margin:
        left_index -= 1
        if values[left_index] < half_max:
            repeats += 1
        else:
            repeats = 0

    # Walk to the right
    right_index = max_index
    repeats = 0
    while right_index < len(values) - 1 and repeats <= margin:
        right_index += 1
        if values[right_index] < half_max:
            repeats += 1
        else:
            repeats = 0

    fwhm = (right_index - margin) - (left_index + margin)
    n_samples = len(values)

    bounds = (left_index, right_index)

    return fwhm / n_samples * range_width, bounds


def plot_resolution(im, extent, position, size, in_db=False):
    from jaxus import plot_beamformed, use_dark_style

    (fwhm_x, fwhm_y, curves) = compute_fwhm(
        im, extent, position=position, size=size, in_db=True, return_curves=True
    )

    curve_x, curve_y = curves

    use_dark_style()
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.2], height_ratios=[1, 0.2])
    ax_im = fig.add_subplot(gs[0, 0])
    ax_curve_x = fig.add_subplot(gs[1, 0])
    ax_curve_y = fig.add_subplot(gs[0, 1])

    plot_beamformed_old(ax_im, im, extent_m=extent, cmap="gray")
    ax_im.plot(
        curve_x.target_position[0], curve_x.target_position[1], "+", markersize=4
    )

    curve_x_axis = (
        np.linspace(-size / 2, size / 2, curve_x.curve_vals.shape[0]) + position[0]
    )
    curve_y_axis = (
        np.linspace(-size / 2, size / 2, curve_y.curve_vals.shape[0]) + position[1]
    )
    ax_curve_x.plot(curve_x_axis, curve_x.curve_vals)
    ax_curve_y.plot(curve_y.curve_vals, curve_y_axis)
    ax_curve_x.set_xlim([ax_im.get_xlim()[0], ax_im.get_xlim()[1]])
    ax_curve_y.set_ylim([ax_im.get_ylim()[0], ax_im.get_ylim()[1]])

    ind0, ind1 = curve_x.fwhm_indices
    ax_curve_x.plot(curve_x_axis[ind0:ind1], curve_x.curve_vals[ind0:ind1])
    ax_curve_x.set_title(f"FWHM x: {fwhm_x * 1e3:.2f} mm")

    ind0, ind1 = curve_y.fwhm_indices
    ax_curve_y.plot(curve_y.curve_vals[ind0:ind1], curve_y_axis[ind0:ind1])
    ax_curve_y.set_title(f"FWHM y: {fwhm_y * 1e3:.2f} mm")

    if in_db:
        ax_curve_x.set_ylim([-60, 0])
        ax_curve_y.set_xlim([-60, 0])
    else:
        ax_curve_x.set_ylim([0, 1])
        ax_curve_y.set_xlim([0, 1])

    def onclick(event):
        # Check if the click is within the axes
        if event.inaxes is not None:
            x, z = event.xdata, event.ydata
            print(f"Clicked at ({x*1e3:.3f}e-3, {z*1e3:.3f}e-3)")
        else:
            print("Clicked outside axes bounds but inside plot window")

    def on_escape(event):
        if event.key == "escape":
            # Close all figures
            plt.close("all")

    # Connect the onclick function to the figure
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    cid = fig.canvas.mpl_connect("key_press_event", on_escape)

    plt.tight_layout()
    return fwhm_x, fwhm_y, curves


if __name__ == "__main__":
    from pathlib import Path

    import cv2

    path = Path("scat.png")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    position = (1e-3, 3.6e-3)

    extent = (0, image.shape[1], 0, image.shape[0])
    extent = (-10e-3, 10e-3, 0, 10e-3)

    # # Example usage
    # extent = (-1, 1, -1, 1)
    # x_vals = np.linspace(extent[0], extent[1], 100)
    # y_vals = np.linspace(extent[2], extent[3], 100)

    # # Create a 2D Gaussian
    # position = (0.1, 0.1)
    # X, Y = np.meshgrid(x_vals, y_vals)
    # image = np.exp(-((X - position[0]) ** 2 + (Y - position[1]) ** 2) / 0.1)

    size = 4e-3

    fwhm_x, fwhm_y, curve_x, curve_y, bounds_x, bounds_y = compute_fwhm(
        image, extent, position, size
    )
    curve_x_axis = np.linspace(-size / 2, size / 2, curve_x.shape[0]) + position[0]
    curve_y_axis = np.linspace(-size / 2, size / 2, curve_y.shape[0]) + position[1]

    fig, axes = plt.subplots(2, 2, width_ratios=[1, 0.3], height_ratios=[1, 0.3])
    ax_im = axes[0, 0]
    ax_curve_x = axes[1, 0]
    ax_curve_y = axes[0, 1]
    extent = (extent[0], extent[1], extent[3], extent[2])
    ax_im.imshow(image, extent=extent, cmap="gray")
    ax_im.plot(position[0], position[1], marker="+", color="red")
    ax_curve_x.plot(curve_x_axis, curve_x)

    indices = np.logical_and(
        curve_x_axis - position[0] > bounds_x[0],
        curve_x_axis - position[0] < bounds_x[1],
    )

    ax_curve_x.plot(
        curve_x_axis[indices],
        curve_x[indices],
    )
    ax_curve_x.set_xlim(extent[0], extent[1])
    ax_curve_x.set_title(f"FWHM x: {fwhm_x:.2e}")

    ax_curve_y.plot(curve_y, curve_y_axis)

    indices = np.logical_and(
        curve_y_axis - position[1] > bounds_y[0],
        curve_y_axis - position[1] < bounds_y[1],
    )

    ax_curve_y.plot(
        curve_y[indices],
        curve_y_axis[indices],
    )
    ax_curve_y.set_ylim(extent[2], extent[3])
    ax_curve_y.set_title(f"FWHM y: {fwhm_y:.2e}")

    ax_formatter_mm = lambda x, _: f"{int(x * 1e3)}"
    ax_im.xaxis.set_major_formatter(plt.FuncFormatter(ax_formatter_mm))
    ax_im.yaxis.set_major_formatter(plt.FuncFormatter(ax_formatter_mm))
    ax_curve_x.xaxis.set_major_formatter(plt.FuncFormatter(ax_formatter_mm))
    ax_curve_y.yaxis.set_major_formatter(plt.FuncFormatter(ax_formatter_mm))

    axes[1, 1].axis("off")
    plt.show()
