"""This script loads in the results of an inverse solver and compares the results to
all the baseline beamformers."""

# region
import matplotlib.pyplot as plt

from pathlib import Path
import traceback
from src import (
    plot_side_by_side,
    plot_side_by_side2,
    plot_gcnr,
    plot_reconstruction,
    plot_header,
    plot_error,
)
from src.utils import get_latest_opt_vars
from jaxus import use_light_style

use_light_style()
# Define the directory where the results were stored by the
# ``generate_paper_results.py`` script
results_dir = Path("results/ready_for_plotting")

output_dir = Path("results/figures/generate_plots_script")
# Create the output directory if it does not exist
output_dir.mkdir(parents=True, exist_ok=True)


activated_runs = [
    "cardiac",
    "cirs_2_single_element_transmits",
    "cirs_plane_wave_wavefront_only",
    "cirs_plane_wave_not_wavefront_only",
    "carotid_synthetic_aperture",
    "carotid_synthetic_aperture2",
    "carotid_plane_wave",
    "cirs_planewave",
    "reconstruction_error",
]

skip_goudarzi = True
f_number_s51 = 0.5
f_number_l115v = 2.5

# endregion

# ======================================================================================
# cirs_synthetic_aperture
# ======================================================================================
# region
run_name = "cirs_2_single_element_transmits"
run_directory = results_dir / run_name
data_path = get_latest_opt_vars(run_directory)
if run_name in activated_runs:
    try:
        plot_header(run_name)
        plot_side_by_side(
            data_path=data_path,
            output_path=Path(output_dir, f"{run_name}.pdf"),
            db_min=-70,
            inverse_image_radius=0.45e-3,
            skip_goudarzi=skip_goudarzi,
            f_number=f_number_s51,
        )
    except Exception as e:
        plot_error(run_name, e)

# endregion

# ======================================================================================
# cirs_plane_wave_not_wavefront_only
# ======================================================================================
# region
run_name = "cirs_plane_wave_not_wavefront_only"
run_directory = results_dir / run_name
data_path = get_latest_opt_vars(run_directory)

if run_name in activated_runs:
    try:
        plot_header(run_name)
        plot_side_by_side(
            data_path=data_path,
            output_path=Path(output_dir, f"{run_name}.pdf"),
            inverse_image_radius=0.4e-3,
            skip_goudarzi=skip_goudarzi,
            f_number=f_number_s51,
        )
    except Exception as e:
        plot_error(run_name, e)

# endregion

# ======================================================================================
# cirs_wavefront_only
# ======================================================================================
# region
run_name = "cirs_plane_wave_wavefront_only"
run_directory = results_dir / run_name
data_path = get_latest_opt_vars(run_directory)

run_name_2 = "cirs_plane_wave_not_wavefront_only"


if "cirs_planewave" in activated_runs:
    try:
        plot_header(run_name)
        plot_side_by_side2(
            data_path1=data_path,
            title1="INFER (wavefront only)",
            data_path2=get_latest_opt_vars(results_dir / run_name_2),
            title2="INFER (full model)",
            output_path=Path(output_dir, "cirs_planewave.pdf"),
            db_min=-70,
            inverse_image_radius=0.45e-3,
            # inverse_image_pixel_size=0.12e-3,
            skip_goudarzi=skip_goudarzi,
            f_number=f_number_s51,
            # show_plot=True,
        )
    except Exception as e:
        plot_error(run_name, e)

# endregion

# ======================================================================================
# Plot reconstruction
# ======================================================================================
# region
run_name = "reconstruction_error"
run_directory = results_dir / "cirs_plane_wave_not_wavefront_only"
data_path = get_latest_opt_vars(run_directory)

if run_name in activated_runs:
    try:
        plot_header(run_name)
        plot_reconstruction(
            data_path=data_path,
            output_path=Path(output_dir, f"{run_name}.pdf"),
            skip_goudarzi=skip_goudarzi,
            f_number=f_number_s51,
            # show_plot=True,
        )
    except Exception as e:
        plot_error(run_name, e)

# endregion

# ======================================================================================
# Plot carotid reconstruction with GCNR
# ======================================================================================
run_name = "carotid_synthetic_aperture"


if run_name in activated_runs:
    fig, axes = plt.subplots(3, 5, figsize=(7.5, 3.5))
    try:
        plot_header(run_name)
        run_directory = results_dir / run_name
        data_path = get_latest_opt_vars(run_directory)
        plot_gcnr(
            data_path=data_path,
            output_path=Path(output_dir, f"{run_name}.pdf"),
            inverse_image_pixel_size=0.15e-3,
            skip_goudarzi=skip_goudarzi,
            f_number=f_number_l115v,
            fig_and_axes=(fig, axes[0]),
            hold=True,
            titles=True,
            xlabels=False,
            row_name="",
        )
    except Exception as e:
        plot_error(run_name, e)

    try:
        plot_header(run_name)
        run_name = "carotid_synthetic_aperture2"
        run_directory = results_dir / run_name
        data_path = get_latest_opt_vars(run_directory)
        plot_gcnr(
            data_path=data_path,
            output_path=Path(output_dir, f"{run_name}.pdf"),
            inverse_image_pixel_size=0.15e-3,
            disk_pos_m=(-6.2e-3, 15e-3),
            disk_inner_radius_m=2.7e-3,
            disk_outer_radius_end_m=3.5e-3,
            inset_position="right",
            skip_goudarzi=skip_goudarzi,
            f_number=f_number_l115v,
            fig_and_axes=(fig, axes[1]),
            hold=True,
            titles=False,
            xlabels=False,
            row_name="",
        )
    except Exception as e:
        plot_error(run_name, e)

    label_x = axes[0, 0].get_position().x0 - 0.06
    label_y = (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2

    fig.text(
        label_x,
        label_y,
        "5 single elements\ntransmits",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=7,
        fontweight="bold",
    )

    try:
        run_name = "carotid_plane_wave"
        plot_header(run_name)
        run_directory = results_dir / run_name
        data_path = get_latest_opt_vars(run_directory)
        plot_gcnr(
            data_path=data_path,
            output_path=Path(output_dir, f"{run_name}.pdf"),
            inverse_image_pixel_size=0.15e-3,
            disk_pos_m=(-6.2e-3, 15e-3),
            disk_inner_radius_m=2.7e-3,
            disk_outer_radius_end_m=3.5e-3,
            inset_position="right",
            skip_goudarzi=skip_goudarzi,
            f_number=f_number_l115v,
            fig_and_axes=(fig, axes[2]),
            titles=False,
            xlabels=True,
            row_name="plane wave",
        )
    except Exception as e:
        plot_error(run_name, e)


run_name = "cardiac"
run_directory = results_dir / run_name
data_path = get_latest_opt_vars(run_directory)
if run_name in activated_runs:
    try:
        plot_header(run_name)
        plot_side_by_side(
            data_path=data_path,
            output_path=Path(output_dir, f"{run_name}.pdf"),
            db_min=-60,
            skip_goudarzi=skip_goudarzi,
            inverse_image_radius=0.5e-3,
            f_number=f_number_s51,
            type="cardiac",
            # fig_and_axes=(fig, axes[2]),
            # top_row=False,
            # show_plot=True,
        )
    except Exception as e:
        plot_error(run_name, e)
