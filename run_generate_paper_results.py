"""This script repeatedly calls the inverse_beamform function to generate the results
for the paper. To run the baselines and generate the figures run the script
`generate_plots.py` after this script has finished running.

The results from this script are copied over to `results/ready_for_plotting` to be
loaded in by `generate_plots.py`.
"""

from src.methods import inverse_beamform, plot_error, run_header
import numpy as np
from pathlib import Path
import shutil
import subprocess
import jaxus.utils.log as log
from src.utils import cache_outputs

inverse_beamform = cache_outputs("cache/inverse_beamform")(inverse_beamform)


activated_runs = [
    "cardiac",
    "cirs_2_single_element_transmits",
    "cirs_plane_wave_wavefront_only",
    "cirs_plane_wave_not_wavefront_only",
    "carotid_synthetic_aperture",
    "carotid_synthetic_aperture2",
    "carotid_plane_wave",
]

# ======================================================================================
# Create the results folder
# ======================================================================================
results_folder = Path("results/ready_for_plotting")
results_folder.mkdir(parents=True, exist_ok=True)

# Add a readme to the results folder
with open(results_folder / "readme.txt", "w") as f:
    f.write(
        "This folder contains the results generated with the script"
        "`generate_paper_results.py`."
    )


# ======================================================================================
# S5-1 - CIRS
# ======================================================================================
# ---> The settings used for all S5-1 CIRS results
# ======================================================================================
# region

grid_shape = (256 + 128, 256)
n_ax = 1024 + 256 + 64
n_steps = 20
grid_dx_wl = 0.6
grid_dz_wl = 0.4

# ======================================================================================
# S5-1 - CIRS - cardiac
# ======================================================================================
# region
run_name = "cardiac"
if run_name in activated_runs:
    run_header(run_name)
    try:
        selected_tx = (np.arange(5) / 5 * 80).astype(int)
        working_dir = inverse_beamform(
            hdf5_path=r"data/S5-1_cardiac.hdf5",
            frame=0,
            selected_tx=np.array(
                [
                    0,
                    4,
                    5,
                ]
            ),
            n_ax=n_ax + 128,
            grid_shape=(256, 256),
            run_name=run_name,
            n_steps=n_steps,
            batch_size=32,
            gradient_accumulation=1,
            grid_dx_wl=0.6,
            grid_dz_wl=0.6,
            ax_min=128,
            z_start=2e-3,
            enable_global_t0_shifting=True,
            enable_sound_speed=True,
            enable_waveform_shaping=True,
            wavefront_only=False,
        )

        output_path = results_folder / run_name

        # Remove the target folder if it exists already
        shutil.rmtree(output_path, ignore_errors=True)

        # Copy the results to the results folder
        shutil.copytree(working_dir, output_path)
    except Exception as e:
        plot_error(run_name, e)
# endregion


# ======================================================================================
# S5-1 - CIRS - cirs_2_single_element_transmits
# ======================================================================================
# region
run_name = "cirs_2_single_element_transmits"
if run_name in activated_runs:
    run_header(run_name)
    try:
        selected_tx = (np.arange(5) / 5 * 80).astype(int)
        working_dir = inverse_beamform(
            hdf5_path=r"data/S5-1_phantom.hdf5",
            frame=0,
            selected_tx=np.array(
                [
                    10,
                    60,
                ]
            ),
            n_ax=n_ax,
            grid_shape=grid_shape,
            run_name=run_name,
            n_steps=n_steps,
            batch_size=512,
            gradient_accumulation=1,
            grid_dx_wl=grid_dx_wl,
            grid_dz_wl=grid_dz_wl,
            ax_min=128,
            z_start=2e-3,
            enable_global_t0_shifting=True,
            enable_sound_speed=True,
            enable_waveform_shaping=True,
        )

        output_path = results_folder / run_name

        # Remove the target folder if it exists already
        shutil.rmtree(output_path, ignore_errors=True)

        # Copy the results to the results folder
        shutil.copytree(working_dir, output_path)
    except Exception as e:
        plot_error(run_name, e)
# endregion


# ======================================================================================
# S5-1 - CIRS - cirs_plane_wave_wavefront_only
# ======================================================================================
# region
# --------------------------------------------------------------------------------------
# Wavefront only
# --------------------------------------------------------------------------------------
# region
run_name = "cirs_plane_wave_wavefront_only"
if run_name in activated_runs:
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=r"data/S5-1_phantom.hdf5",
            frame=0,
            selected_tx=np.array(
                [
                    90,
                ]
            ),
            n_ax=n_ax,
            grid_shape=grid_shape,
            run_name=run_name,
            n_steps=n_steps,
            wavefront_only=True,
            batch_size=128,
            gradient_accumulation=4,
            grid_dx_wl=grid_dx_wl,
            grid_dz_wl=grid_dz_wl,
        )

        output_path = results_folder / run_name

        # Remove the target folder if it exists already
        shutil.rmtree(output_path, ignore_errors=True)

        # Copy the results to the results folder
        shutil.copytree(working_dir, output_path)
    except Exception as e:
        plot_error(run_name, e)
# endregion


# --------------------------------------------------------------------------------------
# S5-1 - CIRS - cirs_plane_wave_not_wavefront_only
# --------------------------------------------------------------------------------------
# THIS ONE TAKES A LONG TIME TO RUN
# --------------------------------------------------------------------------------------
# region
run_name = "cirs_plane_wave_not_wavefront_only"
if run_name in activated_runs:
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=r"data/S5-1_phantom.hdf5",
            frame=0,
            selected_tx=np.array(
                [
                    90,
                ]
            ),
            n_ax=n_ax,
            grid_shape=grid_shape,
            run_name=run_name,
            n_steps=n_steps,
            wavefront_only=False,
            batch_size=32,
            gradient_accumulation=16,
            grid_dx_wl=grid_dx_wl,
            grid_dz_wl=grid_dz_wl,
        )

        output_path = results_folder / run_name

        # Remove the target folder if it exists already
        shutil.rmtree(output_path, ignore_errors=True)

        # Copy the results to the results folder
        shutil.copytree(working_dir, output_path)
    except Exception as e:
        plot_error(run_name, e)
# endregion


# ======================================================================================
# Carotid
# ======================================================================================
# region
grid_shape = (256 + 64, 256)
n_ax = 1024 + 64

offset = 16
n_tx = 5
selected_tx = np.arange(n_tx) / (n_tx - 1) * (127 - 2 * offset) + offset
selected_tx = selected_tx.astype(int)

# ======================================================================================
# carotid_synthetic_aperture
# ======================================================================================
# region
run_name = "carotid_synthetic_aperture"
if run_name in activated_runs:
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=r"data/L11-5v_carotid_cross_0.hdf5",
            frame=0,
            selected_tx=(np.arange(5) / 4 * 127).astype(int),
            n_ax=n_ax,
            grid_shape=grid_shape,
            run_name=run_name,
            n_steps=n_steps,
            grid_dx_wl=0.75,
            grid_dz_wl=0.4,
            batch_size=512,
            gradient_accumulation=2,
        )

        output_path = results_folder / run_name

        # Remove the target folder if it exists already
        shutil.rmtree(output_path, ignore_errors=True)

        # Copy the results to the results folder
        shutil.copytree(working_dir, output_path)
    except Exception as e:
        plot_error(run_name, e)
# endregion


# ======================================================================================
# carotid_synthetic_aperture2
# ======================================================================================
# region
run_name = "carotid_synthetic_aperture2"
if run_name in activated_runs:
    run_header(run_name)
    offset = 16
    n_tx = 5
    selected_tx = np.arange(n_tx) / (n_tx - 1) * (127 - 2 * offset) + offset
    selected_tx = selected_tx.astype(int)
    try:
        working_dir = inverse_beamform(
            hdf5_path=r"data/L11-5v_carotid_cross_1.hdf5",
            frame=0,
            selected_tx=selected_tx,
            n_ax=n_ax,
            grid_shape=grid_shape,
            run_name=run_name,
            n_steps=n_steps,
            grid_dx_wl=0.75,
            grid_dz_wl=0.4,
            batch_size=512,
            gradient_accumulation=2,
        )

        output_path = results_folder / run_name

        # Remove the target folder if it exists already
        shutil.rmtree(output_path, ignore_errors=True)

        # Copy the results to the results folder
        shutil.copytree(working_dir, output_path)
    except Exception as e:
        plot_error(run_name, e)
# endregion


# ======================================================================================
# carotid_plane_wave
# ======================================================================================
# region
run_name = "carotid_plane_wave"
if run_name in activated_runs:
    run_header(run_name)
    try:
        selected_tx = np.array([132, 138, 144])
        working_dir = inverse_beamform(
            hdf5_path=r"data/L11-5v_carotid_cross_1.hdf5",
            frame=0,
            selected_tx=selected_tx,
            n_ax=n_ax,
            grid_shape=grid_shape,
            run_name=run_name,
            n_steps=n_steps,
            grid_dx_wl=0.75,
            grid_dz_wl=0.4,
            batch_size=32,
            gradient_accumulation=8,
            wavefront_only=False,
        )

        output_path = results_folder / run_name

        # Remove the target folder if it exists already
        shutil.rmtree(output_path, ignore_errors=True)

        # Copy the results to the results folder
        shutil.copytree(working_dir, output_path)
    except Exception as e:
        plot_error(run_name, e)
# endregion

# endregion


# ======================================================================================
# Run the plotting script
# ======================================================================================
subprocess.run(["python run_generate_plots.py"], shell=True)
