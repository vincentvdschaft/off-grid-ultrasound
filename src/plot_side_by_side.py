"""This script loads in the results of an inverse solver and compares the results to
all the baseline beamformers."""

# region
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x_last = 0
y_last = 0
import jaxus.utils.log as log
from jaxus import (
    iterate_axes,
    load_hdf5,
    plot_beamformed,
    use_light_style,
    use_dark_style,
    gcnr,
    gcnr_compute_disk,
    gcnr_plot_disk_annulus,
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
from src.utils import get_kernel_image, waveform_samples_block
from src.methods.goudarzi import admm_inverse_beamform_compounded
from src.utils.cache import cache_outputs
from src.methods.utils import get_grid, get_data_root

# endregion


def plot_side_by_side(
    data_path,
    output_path,
    f_number=1.5,
    plot_to_darkmode=False,
    inverse_image_pixel_size=0.08e-3,
    inverse_image_radius=0.25e-3,
    db_min=-60,
    beamforming_grid_scaling=1,
    show_plot=False,
    skip_goudarzi=False,
    fig_and_axes=None,
    type="cirs",
):

    # ==================================================================================
    # Load inverse solver data
    # ==================================================================================
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        log.error(f"Could not find file at {data_path}")
        sys.exit(1)

    # Get the limits of the initial grid
    inverse_grid_xlim = data["grid_xlim"]
    inverse_grid_zlim = data["grid_zlim"]

    # Load the scatterer positions and amplitudes
    scat_x = data["scat_x"]
    scat_z = data["scat_z"]
    scat_amp = data["scat_amp"]

    # Get the transmits that were used in the solver
    selected_tx = data["selected_tx"]

    n_tx = len(selected_tx)

    hdf5_path = Path(str(data["path"]))

    log.info("Generating inverse image...")
    get_kernel_image_cached = cache_outputs("temp/cache")(get_kernel_image)
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
        frames=[data["frame"]],
        transmits=selected_tx,
        reduce_probe_to_2d=True,
    )

    sound_speed = data_dict["sound_speed"]
    sampling_frequency = data_dict["sampling_frequency"]
    wavelength = sound_speed / data_dict["center_frequency"]

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
    # Delay-and-sum (DAS)
    # --------------------------------------------------------------------------------------
    log.info("Performing DAS beamforming...")
    # Beamform with DAS and remove frame dimension
    beamform_das_cached = cache_outputs("temp/cache")(beamform_das)

    im_das = beamform_das_cached(
        rf_data=data_dict["raw_data"],
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=1000,
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
    # Minimum variance (MV)
    # --------------------------------------------------------------------------------------
    log.info("Performing MV beamforming...")
    # Beamform with MV and remove frame dimension
    beamform_mv_cached = cache_outputs("temp/cache")(beamform_mv)
    im_mv = beamform_mv_cached(
        rf_data=data_dict["raw_data"],
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=1000,
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
        diagonal_loading=0.05,
        pixel_chunk_size=4096 * 16,
        progress_bar=True,
    )[0]

    im_mv = log_compress(im_mv, normalize=True)
    im_mv = np.reshape(im_mv, (pixel_grid.n_rows, pixel_grid.n_cols))

    # --------------------------------------------------------------------------------------
    # Delay-multiply-and-sum (DMAS)
    # --------------------------------------------------------------------------------------
    log.info("Performing DMAS beamforming...")
    beamform_dmas_cached = cache_outputs("temp/cache")(beamform_dmas)
    im_dmas = beamform_dmas_cached(
        rf_data=data_dict["raw_data"],
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=1000,
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
    admm_inverse_beamform_compounded_cached = cache_outputs("temp/cache")(
        admm_inverse_beamform_compounded
    )
    waveform_samples = waveform_samples_block(
        data_dict["waveform_samples_two_way"], data_dict["tx_waveform_indices"]
    )
    if skip_goudarzi:
        im_red = np.ones_like(im_das) * -60
    else:
        im_red = admm_inverse_beamform_compounded_cached(
            raw_data=data_dict["raw_data"][0, :, :, :, 0],
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

    # ==================================================================================
    # Compute gcnr values
    # ==================================================================================
    if type == "cardiac":
        points_region1 = (
            (-0.015101720815021022, 0.016799929630052446),
            (-0.010376440254086683, 0.01609113754591232),
            (-0.006123687749245793, 0.017036193658099164),
            (-0.0018709352444048893, 0.017744985742239316),
            (0.0023818172604360005, 0.018926305882472905),
            (0.006634569765276904, 0.02128894616294008),
            (0.00876094601769735, 0.0245966425555941),
            (0.010414794214024359, 0.028376867004341555),
            (0.01135985032621123, 0.03428346770550947),
            (0.0123049064383981, 0.03948127632253724),
            (0.0123049064383981, 0.04302523674323798),
            (0.010651058242071076, 0.04515161299565844),
            (0.007815889905510479, 0.04538787702370514),
            (0.004744457540903163, 0.04208018063105112),
            (0.0009642330921557085, 0.03948127632253724),
            (-0.003524783440731913, 0.03782742812621022),
            (-0.008250064001666238, 0.03640984395792993),
            (-0.01273908053455386, 0.03617357992988321),
            (-0.018409417207675055, 0.03333841159332261),
            (-0.020299529432048782, 0.026250490751921124),
            (-0.019354473319861912, 0.020580154078799928),
        )
        points_region2 = (
            (0.013249962550584957, 0.04208018063105112),
            (0.013013698522538253, 0.034755995761602904),
            (0.011832378382304665, 0.02790433894824812),
            (0.00970600212988422, 0.02247026630317364),
            (0.005453249625043316, 0.019162569910519622),
            (0.0009642330921557085, 0.01727245768614588),
            (-0.003997311496825348, 0.015854873517865603),
            (-0.009903912197993248, 0.01514608143372545),
            (-0.012266552478460424, 0.013492233237398427),
            (-0.011085232338226836, 0.011838385041071403),
            (-0.00801379997361952, 0.012547177125211556),
            (-0.003524783440731913, 0.013728497265445144),
            (0.0035631374006695887, 0.015854873517865603),
            (0.007815889905510479, 0.017744985742239316),
            (0.008524681989650632, 0.01845377782637947),
            (0.012777434494491535, 0.016799929630052446),
            (0.01655765894323899, 0.0175087217141926),
            (0.01750271505542586, 0.021052682134893363),
            (0.01844777116761273, 0.024832906583640818),
            (0.015612602831052133, 0.027668074920201402),
            (0.01655765894323899, 0.03073950728480873),
            (0.019392827279799588, 0.03522852381769634),
            (0.02270052367245362, 0.03617357992988321),
            (0.022464259644406903, 0.039953804378630664),
            (0.020574147420033176, 0.044442820911518285),
            (0.012777434494491535, 0.0432615007712847),
        )
        gcnr_vals = {}
        for im, name, extent in zip(
            [im_das, im_mv, im_dmas, im_red, im_inverse],
            ["DAS", "MV", "DMAS", "RED", "INFER"],
            [
                pixel_grid.extent,
                pixel_grid.extent,
                pixel_grid.extent,
                pixel_grid.extent,
                [
                    inverse_grid_xlim[0],
                    inverse_grid_xlim[1],
                    inverse_grid_zlim[1],
                    inverse_grid_zlim[0],
                ],
            ],
        ):

            region1 = extract_polygon_area(
                im, points_region1, (extent[0], extent[1]), (extent[2], extent[3])
            )
            region2 = extract_polygon_area(
                im, points_region2, (extent[0], extent[1]), (extent[2], extent[3])
            )
            gcnr_vals[name] = gcnr(region1, region2)

        for key, val in gcnr_vals.items():
            log.info(f"{key}: {val:.2f}")
    elif type == "cirs":
        disk_pos_m = (-3.8e-3, 31.5e-3)
        inner_radius_m = 3.0e-3
        outer_radius_start_m = 5e-3
        outer_radius_end_m = 7e-3

        disk_pos_m2 = (-32.7e-3, 72e-3)
        inner_radius_m2 = 2.0e-3
        outer_radius_start_m2 = 4e-3
        outer_radius_end_m2 = 6e-3

        gcnr_vals = {}
        gcnr_vals2 = {}
        for im, name, extent in zip(
            [im_das, im_mv, im_dmas, im_red, im_inverse],
            ["DAS", "MV", "DMAS", "RED", "INFER"],
            [
                pixel_grid.extent,
                pixel_grid.extent,
                pixel_grid.extent,
                pixel_grid.extent,
                [
                    inverse_grid_xlim[0],
                    inverse_grid_xlim[1],
                    inverse_grid_zlim[1],
                    inverse_grid_zlim[0],
                ],
                [
                    inverse_grid_xlim[0],
                    inverse_grid_xlim[1],
                    inverse_grid_zlim[1],
                    inverse_grid_zlim[0],
                ],
            ],
        ):
            gcnr_val = gcnr_compute_disk(
                im,
                xlims_m=(extent[0], extent[1]),
                zlims_m=(extent[3], extent[2]),
                disk_pos_m=disk_pos_m,
                inner_radius_m=inner_radius_m,
                outer_radius_start_m=outer_radius_start_m,
                outer_radius_end_m=outer_radius_end_m,
            )
            gcnr_vals[name] = gcnr_val

            gcnr_val2 = gcnr_compute_disk(
                im,
                xlims_m=(extent[0], extent[1]),
                zlims_m=(extent[3], extent[2]),
                disk_pos_m=disk_pos_m2,
                inner_radius_m=inner_radius_m2,
                outer_radius_start_m=outer_radius_start_m2,
                outer_radius_end_m=outer_radius_end_m2,
            )
            gcnr_vals2[name] = gcnr_val2

        for key, val in gcnr_vals.items():
            print(f"{key}: {val:.2f}")
        for key, val in gcnr_vals2.items():
            print(f"{key}: {gcnr_vals2[key]:.2f}")

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
        fig, axes = plt.subplots(1, 5, figsize=(3.5, 3))

    x = 0
    y = 0

    def on_click(event):
        global x_last, y_last
        # Check if the click was in the axes
        if event.inaxes is not None:
            ax = event.inaxes  # The axes instance
            x, y = event.xdata, event.ydata  # The coordinates in axes (data) units
            print(f"({x}, {y}),")
            if x_last != 0 and y_last != 0:
                ax.plot([x_last, x], [y_last, y], color="red", linewidth=0.5)
            # Update the plot
            fig.canvas.draw()
            x_last, y_last = x, y

    # Connect the click event to the handler
    fig.canvas.mpl_connect("button_press_event", on_click)
    ax_das = axes[0]
    ax_mv = axes[1]
    ax_dmas = axes[2]
    ax_red = axes[3]
    ax_inverse = axes[4]

    plot_beamformed(
        ax_inverse,
        im_inverse,
        extent_m=[
            *inverse_grid_xlim,
            inverse_grid_zlim[1],
            inverse_grid_zlim[0],
        ],
        title=f"INFER\n(full model)",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    xlim = ax_inverse.get_xlim()
    ylim = ax_inverse.get_ylim()

    xlim = list(xlim)
    xlim[1] = np.min([xlim[1], 40e-3])
    xlim[0] = -xlim[1]
    ax_inverse.set_xlim(xlim)

    plot_beamformed(
        ax_das,
        im_das,
        pixel_grid.extent,
        title=f"DAS",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_das.set_xlim(xlim)
    ax_das.set_ylim(ylim)

    plot_beamformed(
        ax_mv,
        im_mv,
        pixel_grid.extent,
        title=f"MV",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_mv.set_xlim(xlim)
    ax_mv.set_ylim(ylim)

    plot_beamformed(
        ax_dmas,
        im_dmas,
        pixel_grid.extent,
        title=f"DMAS",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_dmas.set_xlim(xlim)
    ax_dmas.set_ylim(ylim)

    plot_beamformed(
        ax_red,
        im_red,
        pixel_grid.extent,
        title=f"RED",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_red.set_xlim(xlim)
    ax_red.set_ylim(ylim)

    # ----------------------------------------------------------------------------------
    # Plot regions
    # ----------------------------------------------------------------------------------
    if type == "cardiac":
        # Get color cycle
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # Add the first point to the end to close the polygon
        points_region1 = list(points_region1) + [points_region1[0]]
        points_region2 = list(points_region2) + [points_region2[0]]
        ax_das.plot(
            *zip(*points_region1), color=color_cycle[0], linewidth=0.4, alpha=0.5
        )
        ax_das.plot(
            *zip(*points_region2), color=color_cycle[1], linewidth=0.4, alpha=0.5
        )

    # ----------------------------------------------------------------------------------
    # Plot gCNR disks
    # ----------------------------------------------------------------------------------
    else:
        for ax, im, title, extent in zip(
            [ax_das, ax_mv, ax_dmas, ax_red, ax_inverse],
            [im_das, im_mv, im_dmas, im_red, im_inverse],
            ["DAS", "MV", "DMAS", "RED", "INFER"],
            [
                pixel_grid.extent,
                pixel_grid.extent,
                pixel_grid.extent,
                pixel_grid.extent,
                [
                    inverse_grid_xlim[0],
                    inverse_grid_xlim[1],
                    inverse_grid_zlim[1],
                    inverse_grid_zlim[0],
                ],
                [
                    inverse_grid_xlim[0],
                    inverse_grid_xlim[1],
                    inverse_grid_zlim[1],
                    inverse_grid_zlim[0],
                ],
            ],
        ):
            gcnr_plot_disk_annulus(
                ax,
                pos_m=disk_pos_m,
                inner_radius_m=inner_radius_m,
                outer_radius_start_m=outer_radius_start_m,
                outer_radius_end_m=outer_radius_end_m,
                opacity=1.0,
            )

            gcnr_plot_disk_annulus(
                ax,
                pos_m=disk_pos_m2,
                inner_radius_m=inner_radius_m2,
                outer_radius_start_m=outer_radius_start_m2,
                outer_radius_end_m=outer_radius_end_m2,
                opacity=1.0,
            )
            break

    # --------------------------------------------------------------------------------------
    # Have the inverse image share the same axis limits as the other images
    # --------------------------------------------------------------------------------------
    ax_das_xlim = ax_das.get_xlim()
    ax_das_ylim = ax_das.get_ylim()
    ax_inverse.set_xlim(ax_das_xlim)
    ax_inverse.set_ylim(ax_das_ylim)

    # all_inset_axes = np.array([ax_das_inset, ax_mv_inset, ax_dmas_inset, ax_inverse_inset])
    all_axes = np.array([ax_das, ax_mv, ax_dmas, ax_red, ax_inverse])

    for ax in iterate_axes(all_axes):
        ax.set_facecolor("black")

    # Disable the y-axis for all main plots except the first one
    for ax in [ax_mv, ax_dmas, ax_red, ax_inverse]:
        ax.yaxis.set_visible(False)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.05)

    # Create the directory if it does not exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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


def plot_side_by_side2(
    data_path1,
    title1,
    data_path2,
    title2,
    output_path,
    f_number=1.5,
    plot_to_darkmode=False,
    inverse_image_pixel_size=0.08e-3,
    inverse_image_radius=0.25e-3,
    db_min=-60,
    beamforming_grid_scaling=1,
    show_plot=False,
    skip_goudarzi=False,
):

    # ==================================================================================
    # Load inverse solver data
    # ==================================================================================
    try:
        data1 = np.load(data_path1)
    except FileNotFoundError:
        log.error(f"Could not find file at {data_path1}")
        sys.exit(1)

    # Get the limits of the initial grid
    inverse_grid_xlim = data1["grid_xlim"]
    inverse_grid_zlim = data1["grid_zlim"]

    # Load the scatterer positions and amplitudes
    scat_x = data1["scat_x"]
    scat_z = data1["scat_z"]
    scat_amp = data1["scat_amp"]

    # Get the transmits that were used in the solver
    selected_tx = data1["selected_tx"]

    n_tx = len(selected_tx)

    hdf5_path = Path(str(data1["path"]))

    log.info("Generating inverse image...")
    get_kernel_image_cached = cache_outputs("temp/cache")(get_kernel_image)
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
    # normval = np.sort(np.abs(im_inverse.flatten()))[int(0.999 * im_inverse.size)]
    # im_inverse /= normval

    im_inverse = log_compress(im_inverse, normalize=True)

    data_dict = load_hdf5(
        hdf5_path,
        frames=[
            data1["frame"],
        ],
        transmits=selected_tx,
        reduce_probe_to_2d=True,
    )

    wavelength = data_dict["sound_speed"] / data_dict["center_frequency"]

    # ----------------------------------------------------------------------------------
    # Load the second dataset
    # ----------------------------------------------------------------------------------
    try:
        data2 = np.load(data_path2)
    except FileNotFoundError:
        log.error(f"Could not find file at {data_path2}")
        sys.exit(1)

    # Load the scatterer positions and amplitudes
    scat_x2 = data2["scat_x"]
    scat_z2 = data2["scat_z"]
    scat_amp2 = data2["scat_amp"]

    log.info("Generating second inverse image...")
    get_kernel_image_cached = cache_outputs("temp/cache")(get_kernel_image)
    im_inverse2 = get_kernel_image_cached(
        xlim=inverse_grid_xlim,
        zlim=inverse_grid_zlim,
        scatterer_x=scat_x2,
        scatterer_z=scat_z2,
        scatterer_amplitudes=scat_amp2,
        pixel_size=inverse_image_pixel_size,
        radius=inverse_image_radius,
        falloff_power=2,
    )

    # im_inverse2 /= normval

    im_inverse2 = log_compress(im_inverse2, normalize=True)

    sound_speed = data_dict["sound_speed"]
    sampling_frequency = data_dict["sampling_frequency"]
    wavelength = sound_speed / data_dict["center_frequency"]

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
    # Delay-and-sum (DAS)
    # --------------------------------------------------------------------------------------
    log.info("Performing DAS beamforming...")
    # Beamform with DAS and remove frame dimension
    beamform_das_cached = cache_outputs("temp/cache")(beamform_das)

    im_das = beamform_das_cached(
        rf_data=data_dict["raw_data"],
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=1000,
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
    # Minimum variance (MV)
    # --------------------------------------------------------------------------------------
    log.info("Performing MV beamforming...")
    # Beamform with MV and remove frame dimension
    beamform_mv_cached = cache_outputs("temp/cache")(beamform_mv)
    im_mv = beamform_mv_cached(
        rf_data=data_dict["raw_data"],
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=1000,
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
        diagonal_loading=0.0001,
        pixel_chunk_size=4096 * 16,
        progress_bar=True,
    )[0]

    im_mv = log_compress(im_mv, normalize=True)
    im_mv = np.reshape(im_mv, (pixel_grid.n_rows, pixel_grid.n_cols))

    # --------------------------------------------------------------------------------------
    # Delay-multiply-and-sum (DMAS)
    # --------------------------------------------------------------------------------------
    log.info("Performing DMAS beamforming...")
    beamform_dmas_cached = cache_outputs("temp/cache")(beamform_dmas)
    im_dmas = beamform_dmas_cached(
        rf_data=data_dict["raw_data"],
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=1000,
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
    admm_inverse_beamform_compounded_cached = cache_outputs("temp/cache")(
        admm_inverse_beamform_compounded
    )
    waveform_samples = waveform_samples_block(
        data_dict["waveform_samples_two_way"], data_dict["tx_waveform_indices"]
    )
    if skip_goudarzi:
        im_red = np.ones_like(im_das) * -60
    else:
        im_red = admm_inverse_beamform_compounded_cached(
            raw_data=data_dict["raw_data"][0, :, :, :, 0],
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

    disk_pos_m = (-3.8e-3, 31.5e-3)
    inner_radius_m = 3.0e-3
    outer_radius_start_m = 5e-3
    outer_radius_end_m = 7e-3

    disk_pos_m2 = (-32.7e-3, 72e-3)
    inner_radius_m2 = 2.0e-3
    outer_radius_start_m2 = 4e-3
    outer_radius_end_m2 = 6e-3
    # ==================================================================================
    # Compute gCNR values
    # ==================================================================================
    gcnr_vals = {}
    gcnr_vals2 = {}
    for im, name, extent in zip(
        [im_das, im_mv, im_dmas, im_red, im_inverse, im_inverse2],
        ["DAS", "MV", "DMAS", "RED", "INFER", "INFER2"],
        [
            pixel_grid.extent,
            pixel_grid.extent,
            pixel_grid.extent,
            pixel_grid.extent,
            [
                inverse_grid_xlim[0],
                inverse_grid_xlim[1],
                inverse_grid_zlim[1],
                inverse_grid_zlim[0],
            ],
            [
                inverse_grid_xlim[0],
                inverse_grid_xlim[1],
                inverse_grid_zlim[1],
                inverse_grid_zlim[0],
            ],
        ],
    ):
        gcnr_val = gcnr_compute_disk(
            im,
            xlims_m=(extent[0], extent[1]),
            zlims_m=(extent[3], extent[2]),
            disk_pos_m=disk_pos_m,
            inner_radius_m=inner_radius_m,
            outer_radius_start_m=outer_radius_start_m,
            outer_radius_end_m=outer_radius_end_m,
        )
        gcnr_vals[name] = gcnr_val

        gcnr_val2 = gcnr_compute_disk(
            im,
            xlims_m=(extent[0], extent[1]),
            zlims_m=(extent[3], extent[2]),
            disk_pos_m=disk_pos_m2,
            inner_radius_m=inner_radius_m2,
            outer_radius_start_m=outer_radius_start_m2,
            outer_radius_end_m=outer_radius_end_m2,
        )
        gcnr_vals2[name] = gcnr_val2

    for key, val in gcnr_vals.items():
        print(f"{key}: {val:.2f}")
    for key, val in gcnr_vals2.items():
        print(f"{key}: {gcnr_vals2[key]:.2f}")

    # ======================================================================================
    # Plot the results
    # ======================================================================================
    # region
    if plot_to_darkmode:
        use_dark_style()
    else:
        use_light_style()

    log.warning(f"Dynnamic range: {db_min}")

    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.8))
    ax_das = axes[0, 0]
    ax_dmas = axes[0, 1]
    ax_mv = axes[0, 2]
    ax_red = axes[1, 0]
    ax_inverse = axes[1, 1]
    ax_inverse2 = axes[1, 2]

    plot_beamformed(
        ax_inverse,
        im_inverse,
        extent_m=[
            *inverse_grid_xlim,
            inverse_grid_zlim[1],
            inverse_grid_zlim[0],
        ],
        title=f"{title1}",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    xlim = ax_inverse.get_xlim()
    ylim = ax_inverse.get_ylim()

    xlim = list(xlim)
    xlim[1] = np.min([xlim[1], 40e-3])
    xlim[0] = -xlim[1]
    ax_inverse.set_xlim(xlim)

    plot_beamformed(
        ax_inverse2,
        im_inverse2,
        extent_m=[
            *inverse_grid_xlim,
            inverse_grid_zlim[1],
            inverse_grid_zlim[0],
        ],
        title=f"{title2}",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )

    plot_beamformed(
        ax_das,
        im_das,
        pixel_grid.extent,
        title=f"DAS",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_das.set_xlim(xlim)
    ax_das.set_ylim(ylim)

    plot_beamformed(
        ax_mv,
        im_mv,
        pixel_grid.extent,
        title=f"MV",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_mv.set_xlim(xlim)
    ax_mv.set_ylim(ylim)

    plot_beamformed(
        ax_dmas,
        im_dmas,
        pixel_grid.extent,
        title=f"DMAS",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_dmas.set_xlim(xlim)
    ax_dmas.set_ylim(ylim)

    plot_beamformed(
        ax_red,
        im_red,
        pixel_grid.extent,
        title=f"RED",
        probe_geometry=data_dict["probe_geometry"],
        vmin=db_min,
    )
    ax_red.set_xlim(xlim)
    ax_red.set_ylim(ylim)

    # ----------------------------------------------------------------------------------
    # Plot gCNR disks
    # ----------------------------------------------------------------------------------
    for ax, im, title, extent in zip(
        [ax_das, ax_mv, ax_dmas, ax_red, ax_inverse, ax_inverse2],
        [im_das, im_mv, im_dmas, im_red, im_inverse, im_inverse2],
        ["DAS", "MV", "DMAS", "RED", "INFER", "INFER2"],
        [
            pixel_grid.extent,
            pixel_grid.extent,
            pixel_grid.extent,
            pixel_grid.extent,
            [
                inverse_grid_xlim[0],
                inverse_grid_xlim[1],
                inverse_grid_zlim[1],
                inverse_grid_zlim[0],
            ],
            [
                inverse_grid_xlim[0],
                inverse_grid_xlim[1],
                inverse_grid_zlim[1],
                inverse_grid_zlim[0],
            ],
        ],
    ):
        gcnr_plot_disk_annulus(
            ax,
            pos_m=disk_pos_m,
            inner_radius_m=inner_radius_m,
            outer_radius_start_m=outer_radius_start_m,
            outer_radius_end_m=outer_radius_end_m,
            opacity=1.0,
        )

        gcnr_plot_disk_annulus(
            ax,
            pos_m=disk_pos_m2,
            inner_radius_m=inner_radius_m2,
            outer_radius_start_m=outer_radius_start_m2,
            outer_radius_end_m=outer_radius_end_m2,
            opacity=1.0,
        )
        break

    # ----------------------------------------------------------------------------------
    # Plot arrow
    # ----------------------------------------------------------------------------------
    vector_tip = np.array([-26e-3, 66e-3])
    vector = np.array([-4e-3, 4e-3])

    for ax in [ax_das, ax_mv, ax_dmas, ax_red, ax_inverse, ax_inverse2]:
        ax.arrow(
            vector_tip[0] - vector[0],
            vector_tip[1] - vector[1],
            dx=vector[0],
            dy=vector[1],
            color="#CCCCCC",
            head_width=3.2e-3,
            linewidth=0.02e-3,
        )
    # --------------------------------------------------------------------------------------
    # Have the inverse image share the same axis limits as the other images
    # --------------------------------------------------------------------------------------
    ax_das_xlim = ax_das.get_xlim()
    ax_das_ylim = ax_das.get_ylim()
    ax_inverse.set_xlim(ax_das_xlim)
    ax_inverse.set_ylim(ax_das_ylim)

    # all_inset_axes = np.array([ax_das_inset, ax_mv_inset, ax_dmas_inset, ax_inverse_inset])
    all_axes = np.array([ax_das, ax_mv, ax_dmas, ax_red, ax_inverse, ax_inverse2])

    for ax in iterate_axes(all_axes):
        ax.set_facecolor("black")

    # Disable the y-axis for all but the leftmost plots
    for ax in [ax_dmas, ax_mv, ax_inverse, ax_inverse2]:
        ax.yaxis.set_visible(False)

    for ax in [ax_das, ax_dmas, ax_mv]:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(hspace=0.04)

    # Create the directory if it does not exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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


def extract_polygon_area(image, points, xlim, ylim):
    """
    Extracts the pixel values inside a polygon from an image.

    Parameters
    ----------
    image : numpy.ndarray
        The image as a 2D or 3D numpy array.
    points : list of tuples
        List of (x, y) tuples defining the polygon vertices.
    xlim : tuple
        Limits in the horizontal direction (xmin, xmax).
    ylim : tuple
        Limits in the vertical direction (ymin, ymax).

    Returns
    -------
    numpy.ndarray
        An array of pixel values inside the polygon.
    """
    image = np.array(image)

    n_rows, n_cols = image.shape

    # Create a polygon path object
    poly_path = matplotlib.path.Path(points)

    # Create a grid of coordinates covering the image
    x, y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], n_cols), np.linspace(ylim[0], ylim[1], n_rows)
    )

    # Flatten the meshgrid coordinates for easy processing
    coordinates = np.vstack((x.ravel(), y.ravel())).T

    # Use the path object to check which grid points are inside the polygon
    mask = poly_path.contains_points(coordinates)

    # Reshape the mask back to the shape of the x, y meshgrid
    mask = mask.reshape(x.shape)

    # plt.imshow(mask)
    # plt.show()

    # Apply the mask to the image to extract the pixels
    return image[mask]
