from math import ceil
import jax
from jax import lax, jit, vmap
import jax.numpy as jnp
import numpy as np
from functools import partial
from tqdm import tqdm


@jax.jit
def add_patch(array, patch, row, col):
    """
    Adds a patch to the array at position (r, c).

    Parameters:
    array (jax.numpy.ndarray): The larger array.
    patch (jax.numpy.ndarray): The patch to add.
    row (int): The row index where the patch's top-left corner should be placed.
    col (int): The column index where the patch's top-left corner should be placed.

    Returns:
    jax.numpy.ndarray: A new array with the patch added.
    """
    original_patch = lax.dynamic_slice(array, (row, col), patch.shape)
    new_patch = patch + original_patch

    # Use dynamic_update_slice to update the array
    updated_array = lax.dynamic_update_slice(array, new_patch, (row, col))

    return updated_array


@partial(
    jax.jit,
    static_argnames=(
        "radius",
        "extent",
    ),
)
def splat_gaussian(image, position, amplitude, extent, radius=0.5e-3):
    """
    Adds a gaussian at 'position' parameterised by 'radius' and 'amplitude' to 'image'.
    This is an efficient implementation that adds only a small 'patch' to the image,
    covering the region with non-negligible values from this Gaussian.
    e.g., image might by 512x512, but a Gaussian with radius 5 might be well represented
        by a patch of size 10x10.
    """
    n_x, n_y = image.shape
    x, y = position
    xlim = (extent[0], extent[1])
    ylim = (extent[2], extent[3])

    # Compute the spacing between pixels
    dx = (xlim[1] - xlim[0]) / n_x
    dy = (ylim[1] - ylim[0]) / n_y

    patch_size_x = 10 * int(round(radius / dx)) + 1
    patch_size_y = 10 * int(round(radius / dy)) + 1
    patch_size = jnp.array([patch_size_x, patch_size_y])

    # Create a Gaussian patch
    position_in_samples = jnp.array([(x - xlim[0]) / dx, (y - ylim[0]) / dy])

    patch_start = jnp.floor(position_in_samples - patch_size / 2)

    patch_start = jnp.clip(patch_start, 0, jnp.array([n_x, n_y]) - patch_size)

    x_vals = jnp.arange(patch_size_x) + patch_start[0]
    y_vals = jnp.arange(patch_size_y) + patch_start[1]

    x_vals = x_vals * dx + xlim[0] - x
    y_vals = y_vals * dy + ylim[0] - y

    xx, yy = jnp.meshgrid(
        x_vals,
        y_vals,
        indexing="xy",
    )

    patch = jnp.exp(-((xx**2) + (yy**2)) / (2 * radius**2)) * amplitude

    return add_patch(
        image,
        patch,
        patch_start[0].astype(int),
        patch_start[1].astype(int),
    )


@partial(
    jax.jit,
    static_argnames=(
        "radius",
        "extent",
    ),
)
def splat_multiple_gaussians(image, positions, amplitudes, extent, radius=0.5e-3):
    """Add multiple Gaussian splats to an image."""
    image = vmap(
        partial(splat_gaussian, radius=radius, extent=extent),
        in_axes=(None, 0, 0),
    )(image, positions, amplitudes)
    return jnp.sum(image, axis=0) / positions.shape[0]


def splat_gaussians_2d(
    image_shape, positions, amplitudes, extent, radius=0.5e-3, batch_size=1024
):
    """Batched version of splat_multiple_gaussians."""
    n_points = positions.shape[0]
    n_batches = ceil(n_points / batch_size)

    image = jnp.zeros(image_shape)

    for n in tqdm(range(n_batches), desc="Splatting Gaussians 🎨"):
        start = n * batch_size
        end = min((n + 1) * batch_size, n_points)
        image = splat_multiple_gaussians(
            image, positions[start:end], amplitudes[start:end], extent, radius
        )

    return image
