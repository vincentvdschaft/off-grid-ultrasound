"""This script allows one to run inverse_beamform with certain features
ablated, as specified via the argument --ablation

The results from this script are copied over to `results/ready_for_plotting` to be
loaded in by `generate_plots.py`.
"""

from src import inverse_beamform, cache_outputs, plot_error, run_header
import numpy as np
from pathlib import Path
import shutil
import subprocess
import jaxus.utils.log as log
import argparse
from distutils.util import strtobool

inverse_beamform = cache_outputs("cache/inverse_beamform")(inverse_beamform)


POSSIBLE_ABLATIONS = [
    "directivity",
    "element_gain",
    "attenuation_spread",
    "attenuation_absorption",
    "waveform_shaping",
    "global_t0_shifting",
    "tgc_compensation",
    "none",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study: specify which model feature you'd like to ablate."
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        help=f"Which feature to ablate. Has to be one of {POSSIBLE_ABLATIONS}",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"The data_root containing the RF data .hdf5 file.",
    )
    parser.add_argument(
        "--use_high_attenuation_acquisition",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help=f"Whether or not to use the acquisition of the high-attenuation region of the CIRS phantom.",
    )
    args = parser.parse_args()
    assert (
        args.ablation is None or args.ablation in POSSIBLE_ABLATIONS
    ), f"'ablation' value has to be one of {POSSIBLE_ABLATIONS}"
    assert (
        args.data_root is not None
    ), "You must specify a data root containing `data/S5-1_phantom.hdf5`."
    return args


# ======================================================================================
# Create the results folder
# ======================================================================================
results_folder = Path("results/ready_for_plotting")
results_folder.mkdir(parents=True, exist_ok=True)

args = parse_args()
ablation = args.ablation or "none"
data_root = Path(args.data_root)
data_path = data_root / r"data/S5-1_phantom.hdf5"

if args.use_high_attenuation_acquisition:
    data_path = data_root / r"data/S5-1_phantom_highattenuation.hdf5"


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
n_steps = 20000
grid_dx_wl = 0.6
grid_dz_wl = 0.4

RUN_NAME_ROOT = "cirs_plane_wave_not_wavefront_only"
run_name = RUN_NAME_ROOT + f"-ablation={ablation}"

# --------------------------------------------------------------------------------------
# S5-1 - CIRS - cirs_plane_wave_not_wavefront_only
# --------------------------------------------------------------------------------------
# ABLATION = NONE
# --------------------------------------------------------------------------------------
# region
if ablation == "none":
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=data_path,
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

# --------------------------------------------------------------------------------------
# S5-1 - CIRS - cirs_plane_wave_not_wavefront_only
# --------------------------------------------------------------------------------------
# ABLATION = DIRECTIVIIY
# --------------------------------------------------------------------------------------
# region
if ablation == "directivity":
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=data_path,
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
            enable_directivity=False,
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
# ABLATION = ELEMENT GAIN
# --------------------------------------------------------------------------------------
# region
if ablation == "element_gain":
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=data_path,
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
            enable_element_gain=False,
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
# ABLATION = ATTENUATION FROM SPREAD
# --------------------------------------------------------------------------------------
# region
if ablation == "attenuation_spread":
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=data_path,
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
            enable_attenuation_spread=False,
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
# ABLATION = ATTENUATION FROM ABSORPTION
# --------------------------------------------------------------------------------------
# region
if ablation == "attenuation_absorption":
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=data_path,
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
            enable_attenuation_absorption=False,
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
# ABLATION = WAVEFORM DEFORMATION DUE TO FREQUENCY DEPENDENT ATTENUATION
# --------------------------------------------------------------------------------------
# region
if ablation == "waveform_shaping":
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=data_path,
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
            enable_waveform_shaping=False,
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
# ABLATION = INITIAL TIME OFFSET
# --------------------------------------------------------------------------------------
# region
if ablation == "global_t0_shifting":
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=data_path,
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
            enable_global_t0_shifting=False,
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
# ABLATION = TIME GAIN COMPENSATION
# --------------------------------------------------------------------------------------
# region
if ablation == "tgc_compensation":
    run_header(run_name)
    try:
        working_dir = inverse_beamform(
            hdf5_path=data_path,
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
            enable_tgc_compensation=False,
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
# Run the plotting script
# ======================================================================================
subprocess.run(["python generate_plots.py"], shell=True)
