"""This script loads in the results of an inverse solver and compares the results to
all the baseline beamformers."""

# region
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jaxus.utils.log as log
from jaxus import (
    iterate_axes,
    load_hdf5,
    plot_beamformed,
    use_light_style,
    use_dark_style,
)
from jaxus import (
    CartesianPixelGrid,
    beamform_das,
    beamform_dmas,
    beamform_mv,
    find_t_peak,
    log_compress,
)
from jaxus import simulate_rf_data, plot_rf
from jaxus.data import load_hdf5
from src.methods.goudarzi import admm_inverse_beamform_compounded
from src.methods.utils import get_grid
from src.utils import cache_outputs, get_kernel_image, waveform_samples_block


# endregion


def plot_reconstruction(
    data_path: Path,
    output_path: Path,
    vmin: float = -60,
    beamforming_grid_scaling: int = 1,
    f_number: float = 0.75,
    horizontal: bool = True,
    plot_to_darkmode: bool = False,
    inverse_image_pixel_size: tuple = (0.08e-3,),
    inverse_image_radius: float = 0.5e-3,
    skip_goudarzi: bool = False,
    show_plot: bool = False,
):

    try:
        data = np.load(data_path)
    except FileNotFoundError:
        log.error(f"Could not find file {data_path}")
        return

    # Get the limits of the initial grid
    inverse_grid_xlim = data["grid_xlim"]
    inverse_grid_zlim = data["grid_zlim"]

    # Load the scatterer positions and amplitudes
    scat_x = data["scat_x"]
    scat_z = data["scat_z"]
    scat_amp = data["scat_amp"]

    rf_inverse = data["rf_data_hat"]

    # Add dimensions to conform to standard shape
    rf_inverse = rf_inverse[None, ..., None]

    try:
        ax_min = data["ax_min"]
    except KeyError:
        ax_min = 0

    # Get the transmits that were used in the solver
    selected_tx = data["selected_tx"]

    n_tx = len(selected_tx)

    hdf5_path = Path(str(data["path"]))
    log.info("Generating inverse image...")
    get_kernel_image_cached = cache_outputs("temp/cache")(get_kernel_image)
    im_inverse_prelogcompression = get_kernel_image_cached(
        xlim=inverse_grid_xlim,
        zlim=inverse_grid_zlim,
        scatterer_x=scat_x,
        scatterer_z=scat_z,
        scatterer_amplitudes=scat_amp,
        pixel_size=inverse_image_pixel_size,
        radius=inverse_image_radius,
        falloff_power=2,
    )
    im_inverse = log_compress(im_inverse_prelogcompression, normalize=True)

    data_dict = load_hdf5(
        hdf5_path,
        frames=[
            0,
        ],
        transmits=selected_tx,
        reduce_probe_to_2d=True,
    )

    wavelength = data_dict["sound_speed"] / data_dict["center_frequency"]
    n_el = data_dict["probe_geometry"].shape[0]
    element_width_wl = data_dict["element_width"] / wavelength
    n_ax = data["n_ax"]
    rf_data = data_dict["raw_data"][:, :, :n_ax]
    rf_normalization_factor = np.std(rf_data)
    rf_data = rf_data / rf_normalization_factor

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

    log.info(
        f"Creating pixel grid with {pixel_grid.n_rows} x {pixel_grid.n_cols} pixels"
    )
    # --------------------------------------------------------------------------------------
    # Delay-and-sum (DAS)
    # --------------------------------------------------------------------------------------
    log.info("Performing DAS beamforming...")
    # Beamform with DAS and remove frame dimension
    beamform_das_cached = cache_outputs("temp/cache")(beamform_das)
    im_das = beamform_das_cached(
        rf_data=rf_data,
        probe_geometry=data_dict["probe_geometry"],
        t0_delays=data_dict["t0_delays"],
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        sound_speed_lens=1000,
        lens_thickness=1.5e-3,
        carrier_frequency=data_dict["center_frequency"],
        tx_apodizations=data_dict["tx_apodizations"],
        f_number=f_number,
        pixel_positions=pixel_grid.pixel_positions_flat,
        t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6)
        * np.ones(n_tx),
        initial_times=data_dict["initial_times"],
        rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
        iq_beamform=True,
        progress_bar=True,
    )[0]

    # im_das = log_compress(im_das, normalize=True)
    # im_das = np.reshape(im_das, (pixel_grid.n_rows, pixel_grid.n_cols))

    im_das_log_compressed = log_compress(im_das, normalize=False)
    normalization_offset = np.max(im_das_log_compressed)
    im_das_log_compressed -= normalization_offset

    # ======================================================================================
    # Simulate RF based on DAS image
    # ======================================================================================

    log.info("Simulating RF data based on DAS image...")

    simulate_rf_data_cached = cache_outputs("temp/cache")(simulate_rf_data)
    rf_das = simulate_rf_data_cached(
        n_ax=n_ax,
        scatterer_positions=pixel_grid.pixel_positions_flat,
        scatterer_amplitudes=np.abs(im_das).flatten() / n_el**2,
        t0_delays=data_dict["t0_delays"],
        probe_geometry=data_dict["probe_geometry"],
        element_angles=np.zeros(data_dict["probe_geometry"].shape[0]),
        sampling_frequency=data_dict["sampling_frequency"],
        sound_speed=data_dict["sound_speed"],
        carrier_frequency=data_dict["center_frequency"],
        tx_apodizations=data_dict["tx_apodizations"],
        initial_times=data_dict["initial_times"],
        element_width_wl=element_width_wl,
        progress_bar=True,
    )

    admm_inverse_beamform_compounded_cached = cache_outputs("temp/cache")(
        admm_inverse_beamform_compounded
    )
    waveform_samples = waveform_samples_block(
        data_dict["waveform_samples_two_way"], data_dict["tx_waveform_indices"]
    )
    if not skip_goudarzi:
        im_admm, phis = admm_inverse_beamform_compounded_cached(
            raw_data=rf_data[0, ..., 0],
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
            mu=5,
            ax_min=ax_min,
            method="RED",
            epsilon=8e-3,
            f_number=f_number,
            return_phi=True,
        )

        phi = phis[0]

        # phi = csr_array((phi["data"], phi["indices"], phi["indptr"]), shape=phi["shape"])

        # phi = phi_matrices[0]
        n_el = data_dict["probe_geometry"].shape[0]

        rf_admm = phi @ im_admm.flatten()

        rf_admm = np.reshape(rf_admm, (1, 1, n_ax - ax_min, n_el, 1))
        # Pad ax_min zeros to the beginning of the RF data
        rf_admm = np.pad(rf_admm, ((0, 0), (0, 0), (ax_min, 0), (0, 0), (0, 0)))
    else:
        im_admm = im_das
        rf_admm = rf_das

    # ======================================================================================
    # Beamform the residuals
    # ======================================================================================
    im_beamformed_residuals = {}
    mse = {}

    rf_data = np.array(rf_data)
    rf_das = np.array(rf_das)
    rf_admm = np.array(rf_admm)
    rf_inverse = np.array(rf_inverse)
    rf_data[:, :, :ax_min] = 0
    rf_das[:, :, :ax_min] = 0
    rf_admm[:, :, :ax_min] = 0
    rf_inverse[:, :, :ax_min] = 0

    # Create a tuple of the residuals and the true RF data
    rf_residuals = (rf_data, rf_data - rf_das, rf_data - rf_admm, rf_data - rf_inverse)
    keys = ("true", "das", "admm", "inverse")

    # Beamform the residuals
    for rf, key in zip(rf_residuals, keys):
        # Beamform the residual
        new_im = beamform_das(
            rf_data=rf,
            probe_geometry=data_dict["probe_geometry"],
            t0_delays=data_dict["t0_delays"],
            sampling_frequency=data_dict["sampling_frequency"],
            sound_speed=data_dict["sound_speed"],
            sound_speed_lens=1000,
            lens_thickness=1.5e-3,
            carrier_frequency=data_dict["center_frequency"],
            tx_apodizations=data_dict["tx_apodizations"],
            f_number=f_number,
            pixel_positions=pixel_grid.pixel_positions_flat,
            t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6)
            * np.ones(n_tx),
            initial_times=data_dict["initial_times"],
            rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
            iq_beamform=True,
            progress_bar=True,
        )[0]

        # Log compress the image
        new_im = log_compress(new_im, normalize=False)

        new_im -= normalization_offset

        # Reshape the image
        new_im = np.reshape(new_im, (pixel_grid.n_rows, pixel_grid.n_cols))

        im_beamformed_residuals[key] = new_im
        mse[key] = np.mean(rf**2)

    # ==================================================================================
    # Define inset plot function
    # ==================================================================================
    scatterer_pos_m = [6.2e-3, 22.2e-3]
    inset_size_m = 3e-3
    inset_plot_margin_factor = 1.0
    inset_position = "right"
    xlim = (
        scatterer_pos_m[0] - inset_size_m * inset_plot_margin_factor,
        scatterer_pos_m[0] + inset_size_m * inset_plot_margin_factor,
    )
    ylim = (
        scatterer_pos_m[1] + inset_size_m * inset_plot_margin_factor,
        scatterer_pos_m[1] - inset_size_m * inset_plot_margin_factor,
    )

    def add_inset(ax, im, extent):
        xlim_parent = ax.get_xlim()
        ylim_parent = ax.get_ylim()

        ratio = np.abs(xlim[1] - xlim[0]) / np.abs(ylim[1] - ylim[0])
        width = 80
        height = width / ratio

        if inset_position == "left":
            ax_inset = ax.inset_axes(
                bounds=[
                    xlim_parent[0] + 5,  # xlim_parent[1] - width - 1e-3,
                    ylim_parent[0] - height - 5,
                    width,
                    height,
                ],
                transform=ax.transData,
            )
        elif inset_position == "right":
            ax_inset = ax.inset_axes(
                bounds=[
                    xlim_parent[1] - width + 18,
                    ylim_parent[0] - height - 7,
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
        vmax = np.std(im) * 10
        ax_inset.imshow(im, extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax_inset.set_xlim(xlim)
        ax_inset.set_ylim(ylim)
        # Remove the ticks
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        return ax_inset

    # ======================================================================================
    # Plot the results
    # ======================================================================================
    if plot_to_darkmode:
        use_dark_style()
    else:
        use_light_style()

    fig, axes = plt.subplots(2, 4, figsize=(3.5, 3))
    # Reduce the space between the plots
    plt.subplots_adjust(wspace=0.04, hspace=0.4)
    # Set the default font size
    plt.rcParams.update({"font.size": 5})
    # Set the titles to small
    plt.rcParams.update({"axes.titlesize": 5})

    col_das = 1
    col_admm = 2
    col_inverse = 3

    ax_rf_true = axes[0, 0]
    ax_rf_das = axes[0, col_das]
    ax_inverse = axes[0, col_inverse]
    ax_red = axes[0, col_admm]

    ax_das_true = axes[1, 0]
    ax_das_residual = axes[1, col_das]
    ax_inverse_residual = axes[1, col_inverse]
    ax_red_residual = axes[1, col_admm]

    lim_el = (0, -1)
    lim_ax = (256, 512)

    plot_rf(
        ax_rf_true,
        rf_data[0, 0, lim_ax[0] : lim_ax[1], lim_el[0] : lim_el[1], 0],
        title="True RF data",
        start_sample=lim_ax[0],
    )

    plot_rf(
        ax_rf_das,
        rf_das[0, 0, lim_ax[0] : lim_ax[1], lim_el[0] : lim_el[1], 0],
        title=f"DAS\n\nMSE: {mse['das']:.2e}",
        start_sample=lim_ax[0],
    )

    plot_rf(
        ax_inverse,
        rf_inverse[0, 0, lim_ax[0] : lim_ax[1], lim_el[0] : lim_el[1], 0],
        title=f"INFER\n\nMSE: {mse['inverse']:.2e}",
        start_sample=lim_ax[0],
    )

    plot_rf(
        ax_red,
        rf_admm[0, 0, lim_ax[0] : lim_ax[1], lim_el[0] : lim_el[1], 0],
        title=f"RED\n\nMSE: {mse['admm']:.2e}",
        start_sample=lim_ax[0],
    )

    plot_beamformed(
        ax_das_true,
        im_beamformed_residuals["true"],
        extent_m=pixel_grid.extent,
        title="DAS beamformed\ntrue RF data",
        vmin=vmin,
        vmax=0,
    )
    offset = np.max(im_beamformed_residuals["das"])
    # im_beamformed_residuals["das"] -= offset
    plot_beamformed(
        ax_das_residual,
        im_beamformed_residuals["das"],
        extent_m=pixel_grid.extent,
        title=f"beamformed\nresidual",
        vmin=vmin,
        vmax=offset,
    )

    plot_beamformed(
        ax_red_residual,
        im_beamformed_residuals["admm"],
        extent_m=pixel_grid.extent,
        title=f"beamformed\nresidual",
        vmin=vmin,
        vmax=0,
    )

    plot_beamformed(
        ax_inverse_residual,
        im_beamformed_residuals["inverse"],
        extent_m=pixel_grid.extent,
        title=f"beamformed\nresidual",
        vmin=vmin,
        vmax=0,
    )

    # Disable the y-axis for all but the leftmost plots
    for ax in [
        ax_das_residual,
        ax_inverse_residual,
        ax_red_residual,
        ax_rf_das,
        ax_inverse,
        ax_red,
    ]:
        ax.yaxis.set_visible(False)

    dz = 256
    dx = 80

    vector_tip = np.array([17, 285])
    vector = np.array([-6, -6 / dx * dz])

    for ax in [ax_rf_das, ax_inverse, ax_red, ax_rf_true]:
        ax.arrow(
            vector_tip[0] - vector[0],
            vector_tip[1] - vector[1],
            dx=vector[0],
            dy=vector[1],
            color="#CCCCCC",
            head_width=6,
            linewidth=0.6,
        )

        ax.set_xticks([0, 20, 40, 60])
    # plt.tight_layout()

    im_das_real = np.real(im_das).reshape(pixel_grid.n_rows, pixel_grid.n_cols)
    im_inverse_real = np.real(im_inverse_prelogcompression)
    im_admm_real = np.real(im_admm).reshape(pixel_grid.n_rows, pixel_grid.n_cols)
    add_inset(ax_rf_das, im_das_real, pixel_grid.extent)
    add_inset(ax_inverse, im_inverse_real, pixel_grid.extent)
    add_inset(ax_red, im_admm_real, pixel_grid.extent)

    # Get color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    window_width = xlim[1] - xlim[0]
    window_height = ylim[1] - ylim[0]

    # Add a box indicating the inset view in the residual plots
    for ax in [ax_das_residual, ax_inverse_residual, ax_red_residual]:

        ax.add_patch(
            plt.Rectangle(
                (
                    scatterer_pos_m[0] - window_width / 2,
                    scatterer_pos_m[1] - window_height / 2,
                ),
                window_width,
                window_height,
                edgecolor=color_cycle[0],
                facecolor="none",
                linewidth=0.1,
                linestyle="-",
            )
        )

    output_path = Path(output_path)

    # Create the output directory if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)

    # Save with pdf extension
    path = output_path.with_suffix(".pdf")
    plt.savefig(path, bbox_inches="tight")
    log.info(f"Saved figure to {log.yellow(path)}")

    if show_plot:
        plt.show()
    else:
        plt.close()
