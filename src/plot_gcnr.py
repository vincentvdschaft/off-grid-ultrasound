# region
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jaxus.utils.log as log
from jaxus import (
    gcnr_compute_disk,
    gcnr_plot_disk_annulus,
    iterate_axes,
    load_hdf5,
    plot_beamformed,
    plot_to_darkmode,
    use_dark_style,
    use_light_style,
)
from jaxus.beamforming import (
    CartesianPixelGrid,
    beamform_das,
    beamform_dmas,
    beamform_mv,
    find_t_peak,
    log_compress,
)
from jaxus.data import load_hdf5
from src.methods.goudarzi import admm_inverse_beamform_compounded
from src.methods.utils import get_grid
from src.utils import cache_outputs, get_kernel_image, waveform_samples_block

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# endregion
def plot_gcnr(
    data_path,
    output_path,
    f_number=3,
    plot_to_darkmode=False,
    inverse_image_pixel_size=0.08e-3,
    inverse_image_radius=0.18e-3,
    db_min=-60,
    beamforming_grid_scaling=1,
    show_plot=False,
    skip_goudarzi=False,
    disk_pos_m=(4.1e-3, 15.2e-3),
    disk_inner_radius_m=2.4e-3,
    disk_outer_radius_start_m=3.0e-3,
    disk_outer_radius_end_m=3.3e-3,
    inset_plot_margin_factor=1.2,
    inset_position="left",
    fig_and_axes=None,
    hold=False,
    titles=True,
    xlabels=True,
    row_name="",
):

    # ======================================================================================
    # Load solve
    # ======================================================================================
    # region
    try:
        data = np.load(data_path)
    except Exception:
        log.error(f"Failed to load data from {data_path}")
        return

    inverse_grid_xlim = data["grid_xlim"]
    inverse_grid_zlim = data["grid_zlim"]

    scat_x = data["scat_x"]
    scat_z = data["scat_z"]
    scat_amp = data["scat_amp"]
    SELECTED_TX = data["selected_tx"]

    n_ax = data["n_ax"]

    # Get the hdf5 path relative to verasonics data root
    hdf5_path = Path(str(data["path"]))

    # ======================================================================================
    # Enable caching
    # ======================================================================================
    beamform_das_cached = cache_outputs("temp/cache")(beamform_das)
    beamform_mv_cached = cache_outputs("temp/cache")(beamform_mv)
    beamform_dmas_cached = cache_outputs("temp/cache")(beamform_dmas)
    admm_inverse_beamform_compounded_cached = cache_outputs("temp/cache")(
        admm_inverse_beamform_compounded
    )
    get_kernel_image_cached = cache_outputs("temp/cache")(get_kernel_image)

    # ======================================================================================
    # Generate inverse image
    # ======================================================================================

    im_inverse = get_kernel_image_cached(
        xlim=inverse_grid_xlim,
        zlim=inverse_grid_zlim,
        scatterer_x=scat_x,
        scatterer_z=scat_z,
        scatterer_amplitudes=scat_amp,
        pixel_size=inverse_image_pixel_size,
        radius=inverse_image_radius,
        falloff_power=2,
    )
    im_inverse = log_compress(im_inverse, normalize=True)

    data_dict = load_hdf5(
        hdf5_path,
        frames=[
            0,
        ],
        transmits=SELECTED_TX,
        reduce_probe_to_2d=True,
    )

    raw_data = data_dict["raw_data"][:, :n_ax]

    n_tx = raw_data.shape[1]

    sound_speed = data_dict["sound_speed"]
    sampling_frequency = data_dict["sampling_frequency"]
    wavelength = sound_speed / data_dict["center_frequency"]
    # endregion

    # ======================================================================================
    # Beamform with different beamformers
    # ======================================================================================
    # region

    # ======================================================================================
    # Beamform with different beamformers
    # ======================================================================================
    width = inverse_grid_xlim[1] - inverse_grid_xlim[0]
    height = inverse_grid_zlim[1] - inverse_grid_zlim[0]

    pixel_grid = get_grid(
        width=width,
        height=height,
        sampling_frequency=sampling_frequency,
        sound_speed=sound_speed,
        wavelength=wavelength,
    )

    # --------------------------------------------------------------------------------------
    # Delay-and-Sum (DAS)
    # --------------------------------------------------------------------------------------
    log.info("Performing DAS beamforming...")
    # Beamform with DAS and remove frame dimension
    im_das = beamform_das_cached(
        rf_data=raw_data,
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=data_dict["sound_speed"],
        lens_thickness=1.5e-3,
        carrier_frequency=data_dict["center_frequency"],
        f_number=f_number,
        pixel_positions=pixel_grid.pixel_positions_flat,
        t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6)
        * np.ones(n_tx),
        initial_times=data_dict["initial_times"],
        tx_apodizations=data_dict["tx_apodizations"],
        rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
        iq_beamform=True,
        progress_bar=True,
    )[0]

    im_das = log_compress(im_das, normalize=True)
    im_das = np.reshape(im_das, (pixel_grid.n_rows, pixel_grid.n_cols))

    # --------------------------------------------------------------------------------------
    # Minimum Variance (MV)
    # --------------------------------------------------------------------------------------
    log.info("Performing MV beamforming...")
    # Beamform with MV and remove frame dimension
    im_mv = beamform_mv_cached(
        rf_data=raw_data,
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=data_dict["sound_speed"],
        lens_thickness=1.5e-3,
        carrier_frequency=data_dict["center_frequency"],
        f_number=f_number,
        pixel_positions=pixel_grid.pixel_positions_flat,
        t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6)
        * np.ones(n_tx),
        initial_times=data_dict["initial_times"],
        tx_apodizations=data_dict["tx_apodizations"],
        rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
        iq_beamform=True,
        subaperture_size=30,
        diagonal_loading=1e-3,
        pixel_chunk_size=4096 * 4,
        progress_bar=True,
    )[0]

    # Clip the top 10 values to the 10th largest value. The highest values are likely
    # due to ill-conditioned correlation matrices.
    im_mv = np.abs(im_mv)
    im_mv_sorted = np.sort(im_mv.flatten())
    max_val = im_mv_sorted[-10]
    im_mv = np.clip(im_mv, 0, max_val)

    im_mv = log_compress(im_mv, normalize=True)
    im_mv = np.reshape(im_mv, (pixel_grid.n_rows, pixel_grid.n_cols))

    # --------------------------------------------------------------------------------------
    # Delay Multiply and Sum (DMAS)
    # --------------------------------------------------------------------------------------
    log.info("Performing DMAS beamforming...")
    im_dmas = beamform_dmas_cached(
        rf_data=raw_data,
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=data_dict["sound_speed"],
        lens_thickness=1.5e-3,
        carrier_frequency=data_dict["center_frequency"],
        f_number=f_number,
        pixel_positions=pixel_grid.pixel_positions_flat,
        t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6)
        * np.ones(n_tx),
        initial_times=data_dict["initial_times"],
        tx_apodizations=data_dict["tx_apodizations"],
        rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
        # pixel_chunk_size=1024,
        progress_bar=True,
    )

    im_dmas = log_compress(im_dmas, normalize=True)
    im_dmas = np.reshape(im_dmas, (pixel_grid.n_rows, pixel_grid.n_cols))

    # --------------------------------------------------------------------------------------
    # Regularization by denoising (RED)
    # --------------------------------------------------------------------------------------
    waveform_samples = waveform_samples_block(
        data_dict["waveform_samples_two_way"], data_dict["tx_waveform_indices"]
    )
    if skip_goudarzi:
        im_red = np.ones_like(im_das) * -60
    else:
        im_red = admm_inverse_beamform_compounded_cached(
            raw_data=raw_data[0, :, :, :, 0],
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            tx_apodizations=data_dict["tx_apodizations"],
            initial_times=data_dict["initial_times"],
            t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6)
            * np.ones(n_tx),
            sound_speed=data_dict["sound_speed"],
            waveform_samples=waveform_samples,
            sampling_frequency=data_dict["sampling_frequency"],
            carrier_frequency=data_dict["center_frequency"],
            pixel_positions=pixel_grid.pixel_positions,
            nlm_h_parameter=0.8,
            mu=2000,
            method="RED",
            epsilon=8e-3,
            f_number=f_number,
            chunk_size=1024,
        )
        im_red = log_compress(im_red, normalize=True)

    # endregion
    # ======================================================================================
    # Compute gCNR for all results
    # ======================================================================================
    # region
    gcnr_value = {}
    for im, key, xlim, zlim in (
        (im_das, "DAS", pixel_grid.collim, pixel_grid.rowlim),
        (im_mv, "MV", pixel_grid.collim, pixel_grid.rowlim),
        (im_dmas, "DMAS", pixel_grid.collim, pixel_grid.rowlim),
        (im_red, "RED", pixel_grid.collim, pixel_grid.rowlim),
        (im_inverse, "Inverse", inverse_grid_xlim, inverse_grid_zlim),
    ):
        gcnr_value[key] = gcnr_compute_disk(
            im,
            xlims_m=xlim,
            zlims_m=zlim,
            disk_pos_m=disk_pos_m,
            inner_radius_m=disk_inner_radius_m,
            outer_radius_start_m=disk_outer_radius_start_m,
            outer_radius_end_m=disk_outer_radius_end_m,
            num_bins=256,
        )
    # endregion

    # ======================================================================================
    # Define the inset plot function
    # ======================================================================================
    # region
    def add_inset(ax, im, extent):
        xlim_parent = ax.get_xlim()
        ylim_parent = ax.get_ylim()
        xlim = (
            disk_pos_m[0] - disk_outer_radius_end_m * inset_plot_margin_factor,
            disk_pos_m[0] + disk_outer_radius_end_m * inset_plot_margin_factor,
        )
        ylim = (
            disk_pos_m[1] + disk_outer_radius_end_m * inset_plot_margin_factor,
            disk_pos_m[1] - disk_outer_radius_end_m * inset_plot_margin_factor,
        )

        ratio = np.abs(xlim[1] - xlim[0]) / np.abs(ylim[1] - ylim[0])
        width = 10e-3
        height = width / ratio

        if inset_position == "left":
            ax_inset = ax.inset_axes(
                bounds=[
                    xlim_parent[0] + 1e-3,  # xlim_parent[1] - width - 1e-3,
                    ylim_parent[0] - height - 1e-3,
                    width,
                    height,
                ],
                transform=ax.transData,
            )
        elif inset_position == "right":
            ax_inset = ax.inset_axes(
                bounds=[
                    xlim_parent[1] - width - 1e-3,
                    ylim_parent[0] - height - 1e-3,
                    width,
                    height,
                ],
                transform=ax.transData,
            )
        else:
            raise ValueError(
                f"Invalid inset position: {inset_position}. Must be 'left' or 'right'"
            )
        # # Flip the image vertically. This is needed because the inset axes plot does not
        # # work well with the [zmax, zmin] limits.
        # im = np.flipud(im)
        ax_inset.imshow(
            im,
            extent=extent,
            vmin=db_min,
            cmap="gray",
        )
        ax_inset.set_xlim(xlim)
        ax_inset.set_ylim(ylim)
        # Remove the ticks
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        return ax_inset

    # endregion

    # ======================================================================================
    # Plot the results
    # ======================================================================================
    # region
    if plot_to_darkmode:
        use_dark_style()
    else:
        use_light_style()

    if fig_and_axes is not None:
        fig, axes = fig_and_axes
    else:
        fig, axes = plt.subplots(1, 5, figsize=(7, 6))
    ax_das = axes[0]
    ax_mv = axes[1]
    ax_dmas = axes[2]
    ax_red = axes[3]
    ax_inverse = axes[4]

    extent_inverse = [
        *inverse_grid_xlim,
        inverse_grid_zlim[1],
        inverse_grid_zlim[0],
    ]
    name = f"INFER\n" if titles else ""
    plot_beamformed(
        ax_inverse,
        im_inverse,
        extent_m=extent_inverse,
        title=f"{name}",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )

    pg = data_dict["probe_geometry"]
    # xlim = np.array([pg[:, 0].min(), pg[:, 0].max()]) * 1.3
    # ax_inverse.set_xlim(xlim)

    ax_inverse_inset = add_inset(ax_inverse, im_inverse, extent_inverse)

    if not xlabels:
        ax_inverse.set_xlabel("")
        ax_inverse.set_xticklabels([])

    name = f"DAS\n" if titles else ""
    plot_beamformed(
        ax_das,
        im_das,
        pixel_grid.extent,
        title=f"{name}",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_das.set_xlim(ax_inverse.get_xlim())
    ax_das.set_ylim(ax_inverse.get_ylim())
    ax_das_inset = add_inset(ax_das, im_das, pixel_grid.extent)

    if not xlabels:
        ax_das.set_xlabel("")
        ax_das.set_xticklabels([])

    name = f"MV\n" if titles else ""
    plot_beamformed(
        ax_mv,
        im_mv,
        pixel_grid.extent,
        title=f"{name}",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_mv.set_xlim(ax_inverse.get_xlim())
    ax_mv.set_ylim(ax_inverse.get_ylim())
    ax_mv_inset = add_inset(ax_mv, im_mv, pixel_grid.extent)

    if not xlabels:
        ax_mv.set_xlabel("")
        ax_mv.set_xticklabels([])

    name = f"DMAS\n" if titles else ""
    plot_beamformed(
        ax_dmas,
        im_dmas,
        pixel_grid.extent,
        title=f"{name}",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_dmas.set_xlim(ax_inverse.get_xlim())
    ax_dmas.set_ylim(ax_inverse.get_ylim())
    ax_dmas_inset = add_inset(ax_dmas, im_dmas, pixel_grid.extent)

    if not xlabels:
        ax_dmas.set_xlabel("")
        ax_dmas.set_xticklabels([])

    name = f"RED\n" if titles else ""
    plot_beamformed(
        ax_red,
        im_red,
        pixel_grid.extent,
        title=f"{name}",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_red.set_xlim(ax_inverse.get_xlim())
    ax_red.set_ylim(ax_inverse.get_ylim())
    ax_red_inset = add_inset(ax_red, im_red, pixel_grid.extent)

    if not xlabels:
        ax_red.set_xlabel("")
        ax_red.set_xticklabels([])

    # Disable the y-axis for all main plots except the first one
    for ax in [ax_mv, ax_dmas, ax_red, ax_inverse]:
        ax.yaxis.set_visible(False)

    # --------------------------------------------------------------------------------------
    # Add gCNR circles
    # --------------------------------------------------------------------------------------

    all_inset_axes = np.array(
        [
            ax_das_inset,
            ax_mv_inset,
            ax_dmas_inset,
            ax_red_inset,
            ax_inverse_inset,
        ]
    )
    all_axes = np.array([ax_das, ax_mv, ax_dmas, ax_red, ax_inverse, *all_inset_axes])
    for ax in iterate_axes(all_inset_axes):
        gcnr_plot_disk_annulus(
            ax,
            pos_m=disk_pos_m,
            inner_radius_m=disk_inner_radius_m,
            outer_radius_start_m=disk_outer_radius_start_m,
            outer_radius_end_m=disk_outer_radius_end_m,
            opacity=1.0,
        )

    if not plot_to_darkmode:
        for ax in iterate_axes(all_axes):
            ax.set_facecolor("black")
        for ax in iterate_axes(all_inset_axes):
            spine_color = "#AAAAAA"
            # Make the spines white
            ax.spines["bottom"].set_color(spine_color)
            ax.spines["left"].set_color(spine_color)
            ax.spines["top"].set_color(spine_color)
            ax.spines["right"].set_color(spine_color)

    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    # plt.tight_layout(h_pad=0)

    # Get the location of the leftmost subplot
    label_x = ax_das.get_position().x0 - 0.06
    label_y = ax_das.get_position().y0 + ax_das.get_position().height / 2

    fig.text(
        label_x,
        label_y,
        row_name,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=7,
        fontweight="bold",
    )

    # Make all subplot titles bold
    for ax in iterate_axes(all_axes):
        ax.title.set_fontweight("bold")

    if hold:
        return

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    log.info(f"Saved figure to {log.yellow(output_path)}")

    # Also save the figure as a PDF
    output_path = output_path.with_suffix(".pdf")
    plt.savefig(output_path, bbox_inches="tight")
    log.info(f"Saved figure to {log.yellow(output_path)}")

    if show_plot:
        plt.show()
    else:
        plt.close()
    # endregion
