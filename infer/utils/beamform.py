from typing import List, Union
import numpy as np
from jaxus import (
    beamform_das,
    load_hdf5,
    get_pixel_grid,
    find_t_peak,
    sort_extent,
    beamform_dmas,
    beamform_mv,
)
import jax.numpy as jnp


def beamform_file(
    hdf5_path,
    frame: int,
    transmits: list,
    target_region_m: Union[List[float], np.ndarray],
    fnumber: float = 1.5,
    beamformer="DAS",
    **overwrite_kwargs,
):
    """
    Beamform a single frame from a file.

    Parameters
    ----------
    hdf5_path : str
        The path to the HDF5 file containing the data_dict.
    frame : int
        The frame to
    transmits : list
        The transmits to use.
    target_region_m : list
        The target region in meters [xmin, xmax, zmin, zmax].
    fnumber : float, optional
        The f-number of the beamformer, by default 1.5.
    beamformer : str, optional
        The beamformer to use, by default "DAS".

    Returns
    -------
    im_das : np.ndarray
        The beamformed image.
    """
    SUPPORTED_BEAMFORMERS = ("DAS", "DMAS", "MV")
    assert isinstance(beamformer, str), "Beamformer must be a string."
    beamformer = beamformer.upper()
    assert (
        beamformer in SUPPORTED_BEAMFORMERS
    ), f"Beamformer {beamformer} not supported. Choose from {SUPPORTED_BEAMFORMERS}."

    data_dict = load_hdf5(
        hdf5_path, frames=[frame], transmits=transmits, reduce_probe_to_2d=True
    )

    target_region_m = sort_extent(target_region_m)

    wavelength = data_dict["sound_speed"] / data_dict["center_frequency"]
    n_ax = data_dict["raw_data"].shape[2]
    dx_wl = 0.125
    dz_wl = 0.125

    dx = target_region_m[1] - target_region_m[0]
    dz = target_region_m[3] - target_region_m[2]

    n_x = int(dx / (dx_wl * wavelength)) + 1
    n_z = int(dz / (dz_wl * wavelength)) + 1

    startpoints = (target_region_m[0], target_region_m[2])

    shape = (n_x, n_z)
    spacing = (dx_wl * wavelength, dz_wl * wavelength)
    center = (False, False)
    pixel_grid = get_pixel_grid(
        shape=shape,
        spacing=spacing,
        startpoints=startpoints,
        center=center,
    )

    t_peak = find_t_peak(data_dict["waveform_samples_two_way"][0][:]) * jnp.ones(1) * 2

    if beamformer == "DAS":
        kwargs = dict(
            rf_data=data_dict["raw_data"],
            pixel_positions=pixel_grid.pixel_positions_flat,
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            initial_times=data_dict["initial_times"],
            sampling_frequency=data_dict["sampling_frequency"],
            carrier_frequency=data_dict["center_frequency"],
            sound_speed=data_dict["sound_speed"],
            sound_speed_lens=1540,
            lens_thickness=1.5e-3,
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=jnp.ones(data_dict["tx_apodizations"].shape[1]),
            f_number=fnumber,
            t_peak=t_peak,
            iq_beamform=True,
            progress_bar=True,
            pixel_chunk_size=2**21,
        )
        kwargs = _overwrite_kwargs(overwrite_kwargs, kwargs)
        im_beamformed = beamform_das(**kwargs)
    elif beamformer == "DMAS":
        kwargs = dict(
            rf_data=data_dict["raw_data"],
            pixel_positions=pixel_grid.pixel_positions_flat,
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            initial_times=data_dict["initial_times"],
            sampling_frequency=data_dict["sampling_frequency"],
            carrier_frequency=data_dict["center_frequency"],
            sound_speed=data_dict["sound_speed"],
            sound_speed_lens=1540,
            lens_thickness=1.5e-3,
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=jnp.ones(data_dict["tx_apodizations"].shape[1]),
            f_number=fnumber,
            t_peak=t_peak,
            progress_bar=True,
            pixel_chunk_size=2**21,
        )
        kwargs = _overwrite_kwargs(overwrite_kwargs, kwargs)
        im_beamformed = beamform_dmas(
            **kwargs,
        )
    elif beamformer == "MV":
        kwargs = dict(
            rf_data=data_dict["raw_data"],
            pixel_positions=pixel_grid.pixel_positions_flat,
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            initial_times=data_dict["initial_times"],
            sampling_frequency=data_dict["sampling_frequency"],
            carrier_frequency=data_dict["center_frequency"],
            sound_speed=data_dict["sound_speed"],
            sound_speed_lens=1540,
            lens_thickness=1.5e-3,
            tx_apodizations=data_dict["tx_apodizations"],
            rx_apodization=jnp.ones(data_dict["tx_apodizations"].shape[1]),
            f_number=fnumber,
            t_peak=t_peak,
            subaperture_size=10,
            diagonal_loading=1e-3,
            iq_beamform=True,
            progress_bar=True,
            pixel_chunk_size=2**12,
        )
        kwargs = _overwrite_kwargs(overwrite_kwargs, kwargs)
        im_beamformed = beamform_mv(**kwargs)

    im_beamformed = im_beamformed.reshape(pixel_grid.shape)

    return im_beamformed


def _overwrite_kwargs(overwrite_kwargs, kwargs):
    if overwrite_kwargs is not None:
        for key, value in overwrite_kwargs.items():
            kwargs[key] = value

    return kwargs
