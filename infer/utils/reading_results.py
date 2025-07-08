"""This module contains some functions to read from results output directories."""

from pathlib import Path

import numpy as np
import yaml
from jaxus import log


def result_get_state(path: str):
    """Reads the state from a result directory.

    Parameters
    ----------
    path : str
        The path to the result directory.

    Returns
    -------
    dict
        The state dictionary.
    """
    state = np.load(path / "state.npz")
    return state


def result_get_source_hdf5_file(path: str, remove_verasonics_root: bool = True):
    """Reads the source HDF5 file from a result directory.

    Parameters
    ----------
    path : str
        The path to the result directory.

    Returns
    -------
    str
        The path to the source HDF5 file.
    """
    config_path = path / "run_config.yaml"

    try:
        with open(config_path, "r", encoding="UTF-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The config file {config_path} does not exist.")

    try:
        hdf5_path = config["path"]
    except KeyError:
        raise KeyError("The config file does not contain a 'path' key.")

    if remove_verasonics_root:
        hdf5_path = remove_verasonics_data_root(hdf5_path)

    return hdf5_path


def remove_verasonics_data_root(path: str):
    """Removes the Verasonics data root from a path.

    Parameters
    ----------
    path : str
        The path to remove the Verasonics data root from.

    Returns
    -------
    str
        The path with the Verasonics data root removed.
    """
    path = Path(path)
    for i, part in enumerate(path.parts):
        if part.lower() == "verasonics":
            return Path(*path.parts[i + 1 :])

    log.warning("Could not find Verasonics data root in path.")
    return path
