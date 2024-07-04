"""This module contains some functions that are called by the modules in paper_results
to ensure consistency in the plots that are generated."""
import os
from pathlib import Path

from jaxus.beamforming import CartesianPixelGrid
import jaxus.utils.log as log
import traceback


def get_grid(width, height, sampling_frequency, sound_speed, wavelength):

    dz_m = sound_speed / (2 * sampling_frequency)
    dx_m = 0.5 * wavelength

    dx_wl = dx_m / wavelength
    dz_wl = dz_m / wavelength

    n_x = int(width / dx_m) + 1
    n_z = int(height / dz_m) + 1

    log.info(f"Creating pixel grid with {n_z} x {n_x} pixels")

    pixel_grid = CartesianPixelGrid(
        n_x=n_x,
        n_z=n_z,
        dx_wl=dx_wl,
        dz_wl=dz_wl,
        z0=1e-3,
        wavelength=wavelength,
    )

    return pixel_grid


def plot_error(run_name, exception):
    print("-" * 80 + "\nTRACEBACK\n" + "-" * 80 + "\n")
    traceback.print_exc()
    print("\n" + "-" * 80 + "\n")
    log.error(f"{exception} - Failed to plot {run_name}")

    # Check if the exception was a keyboard interrupt
    if isinstance(exception, KeyboardInterrupt):
        log.error("Keyboard interrupt detected, exiting...")
        exit(1)


def plot_header(run_name):
    log.info("-" * 80)
    log.info(f"Plotting {run_name}")
    log.info("-" * 80)


def run_header(run_name):
    log.info("-" * 80)
    log.info(f"Running {run_name}")
    log.info("-" * 80)


def get_data_root(default=None):
    """Gets the data root from the VERASONICS_DATA_ROOT environment variable. If the
    environment variable is not set, an error is raised unless a default value is
    provided.

    Parameters
    ----------
    default : str, default=None
        The default value to use if the environment variable is not set.

    Returns
    -------
    Path
        The path to the data root.
    """
    data_root = os.getenv("VERASONICS_DATA_ROOT", default)

    if data_root is None and default is None:
        raise EnvironmentError("VERASONICS_DATA_ROOT environment variable not set")

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root {data_root} does not exist")

    return Path(data_root)