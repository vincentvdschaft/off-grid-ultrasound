"""Data loading utilities."""

from typing import List, Tuple, Union
import h5py
import numpy as np


def load_hdf5(
    path,
    frames: Union[Tuple, List, np.ndarray],
    transmits: Union[Tuple, List, np.ndarray],
    reduce_probe_to_2d: bool = False,
):
    """
    Loads a USBMD dataset into a python dictionary.

    Parameters
    ----------
    path : str
        The path to the USBMD dataset.
    frames : list
        The frames to load (list of indices).
    transmits : list
        The transmits to load (list of indices).
    reduce_probe_to_2d : bool
        Whether to reduce the probe geometry to 2D, omitting the y-coordinate.

    Returns
    -------
    loaded_data : dict
        The loaded data.
    """

    frames = np.array(frames)
    transmits = np.array(transmits)

    with h5py.File(path, "r") as dataset:
        data = {}
        raw_data = dataset["data"]["raw_data"][frames]
        raw_data = raw_data[:, transmits]
        data["raw_data"] = raw_data.astype(np.float32)

        t0_delays = dataset["scan"]["t0_delays"][transmits]
        data["t0_delays"] = t0_delays.astype(np.float32)

        tx_apodizations = dataset["scan"]["tx_apodizations"][transmits]
        data["tx_apodizations"] = tx_apodizations.astype(np.float32)

        initial_times = dataset["scan"]["initial_times"][transmits]
        data["initial_times"] = initial_times.astype(np.float32)

        sampling_frequency = dataset["scan"]["sampling_frequency"][()]
        data["sampling_frequency"] = float(sampling_frequency)

        center_frequency = dataset["scan"]["center_frequency"][()]
        data["center_frequency"] = float(center_frequency)

        sound_speed = dataset["scan"]["sound_speed"][()]
        data["sound_speed"] = float(sound_speed)

        probe_geometry = dataset["scan"]["probe_geometry"][()]
        if reduce_probe_to_2d:
            probe_geometry = probe_geometry[:, np.array([0, 2])]
        data["probe_geometry"] = probe_geometry.astype(np.float32)

        if "element_width" in dataset["scan"]:
            element_width = dataset["scan"]["element_width"][()]
            data["element_width"] = float(element_width)

        if "bandwidth" in dataset["scan"]:
            bandwidth = dataset["scan"]["bandwidth"][()]
            data["bandwidth"] = (float(bandwidth[0]), float(bandwidth[1]))

        waveform_samples_one_way = []
        waveform_samples_two_way = []

        if "waveforms_one_way" in dataset["scan"]:
            for key in dataset["scan"]["waveforms_one_way"].keys():
                samples = dataset["scan"]["waveforms_one_way"][key][()].astype(
                    np.float32
                )
                waveform_samples_one_way.append(samples)
        data["waveform_samples_one_way"] = waveform_samples_one_way

        if "waveforms_two_way" in dataset["scan"]:
            for key in dataset["scan"]["waveforms_two_way"].keys():
                samples = dataset["scan"]["waveforms_two_way"][key][()].astype(
                    np.float32
                )
                waveform_samples_two_way.append(samples)

        data["waveform_samples_two_way"] = waveform_samples_two_way

        tx_waveform_indices = dataset["scan"]["tx_waveform_indices"][transmits]
        data["tx_waveform_indices"] = tx_waveform_indices

        if "tgc_gain_curve" in dataset["scan"]:
            tgc_gain_curve = dataset["scan"]["tgc_gain_curve"][:]
            data["tgc_gain_curve"] = tgc_gain_curve.flatten().astype(np.float32)

        if "polar_angles" in dataset["scan"]:
            polar_angles = dataset["scan"]["polar_angles"][transmits]
            data["polar_angles"] = polar_angles.flatten().astype(np.float32)

        if "azimuth_angles" in dataset["scan"]:
            azimuth_angles = dataset["scan"]["azimuth_angles"][transmits]
            data["azimuth_angles"] = azimuth_angles.flatten().astype(np.float32)

        if "focus_distances" in dataset["scan"]:
            focus_distances = dataset["scan"]["focus_distances"][transmits]
            data["focus_distances"] = focus_distances.flatten().astype(np.float32)

    return data
