from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from jaxus import plot_beamformed, use_light_style

from analysis.plot_utils import Label

from .evaluate_result import ImageEvaluation, load_evaluation


class RowLabel:
    def __init__(self, text, row_start, row_end=None, fontsize=7, x_margin=0.07):
        self.text = text
        self.fontsize = fontsize
        self.row_start = row_start
        self.row_end = row_end or row_start
        self.x_margin = x_margin


def plot_evaluation_rows(
    evaluations: List[ImageEvaluation],
    rowlabels,
    collabels: List[Label] = None,
    vmin=-60,
    figsize=(7, 5),
):

    use_light_style()

    assert isinstance(evaluations, list)
    assert all(isinstance(evaluation, ImageEvaluation) for evaluation in evaluations)

    n_rows = len(evaluations)
    n_cols = max([len(evaluations[i].methods) for i in range(len(evaluations))])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Set title font size
    plt.rcParams["axes.titlesize"] = 3

    axes = np.array(axes).reshape(n_rows, n_cols)

    # Get the first color of the colorcycle
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    for row, evaluation in enumerate(evaluations):
        for col, method in enumerate(evaluation.methods):
            ax = axes[row, col]
            plot_beamformed_old(
                ax,
                evaluation.images[method],
                extent_m=evaluation.target_region_m,
                vmin=vmin,
                probe_geometry=evaluation.probe_geometry_m,
            )
            from jaxus import gcnr_plot_disk_annulus

            for gcnr_obj in evaluation.gcnr_points:
                gcnr_plot_disk_annulus(
                    ax,
                    pos_m=gcnr_obj.position,
                    inner_radius_m=gcnr_obj.inner_radius,
                    outer_radius_start_m=gcnr_obj.outer_radius_start,
                    outer_radius_end_m=gcnr_obj.outer_radius_end,
                    linewidth=0.2,
                )
            if col == 0:
                for fwhm_id, fwhm_obj in enumerate(evaluation.fwhm_results[method]):
                    if fwhm_obj.plot_dot:
                        ax.plot(
                            fwhm_obj.x,
                            fwhm_obj.y,
                            "x",
                            markersize=0.2,
                            markerfacecolor="none",
                            color=color,
                        )
                    if fwhm_obj.plot_id:
                        ax.text(
                            fwhm_obj.x,
                            fwhm_obj.y,
                            f"{fwhm_id}",
                            fontsize=3,
                            color=color,
                            ha="left",
                            va="bottom",
                        )

            if row == 0:
                ax.set_title(
                    collabels[col].raw_label if collabels else method, fontsize=7
                )

            if row < n_rows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")

            if col > 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")

    # Set the space between the plots
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    for row_label in rowlabels:
        r0, r1 = row_label.row_start, row_label.row_end
        add_rowlabel(
            fig,
            axes[r0, 0],
            axes[r1, 0],
            row_label.text,
            fontsize=row_label.fontsize,
            x_margin=row_label.x_margin,
        )
    wrap_titles(axes)
    return fig, axes


def wrap_titles(axes, max_chars=20):
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes = axes.flatten()
    for ax in axes:
        title = ax.get_title()
        if len(title) > max_chars:
            title = title[:max_chars] + "\n" + title[max_chars:]
            ax.set_title(title)


def remove_extreme_ticks(axes):
    assert axes.ndim == 2

    n_rows, n_cols = axes.shape

    for col in range(n_cols):
        xticks = axes[-1, col].get_xticklabels()
        if col > 0:
            xticks[0].set_visible(False)
        if col < n_cols - 1:
            xticks[-1].set_visible(False)


def add_rowlabel(fig, ax_start, ax_end, text, fontsize=7, x_margin=0.07):
    label_x = ax_start.get_position().x0 - x_margin

    label_y = (ax_start.get_position().y0 + ax_end.get_position().y1) / 2

    if label_y == ax_start.get_position().y0:
        label_y = ax_start.get_position().y1

    fig.text(
        label_x,
        label_y,
        text,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
        fontweight="bold",
    )


def _add_inset(ax, image, extent, inset_center, inset_size):
    xlim_parent = ax.get_xlim()
    ylim_parent = ax.get_ylim()
    xlim = (
        inset_center[0] - inset_size[0] / 2,
        inset_center[0] + inset_size[0] / 2,
    )
    ylim = (
        inset_center[1] - inset_size[1] / 2,
        inset_center[1] + inset_size[1] / 2,
    )

    ratio = np.abs(xlim[1] - xlim[0]) / np.abs(ylim[1] - ylim[0])
    width = 10e-3
    height = width / ratio

    ax_inset = ax.inset_axes(
        bounds=[
            xlim_parent[0] + 1e-3,  # xlim_parent[1] - width - 1e-3,
            ylim_parent[0] - height - 1e-3,
            width,
            height,
        ],
        transform=ax.transData,
    )

    parent_vmin = ax.get_images()[0].get_clim()[0]
    ax_inset.imshow(
        image,
        extent=extent,
        vmin=parent_vmin,
        cmap="gray",
    )
    ax_inset.set_xlim(xlim)
    ax_inset.set_ylim(ylim)

    # Remove the ticks
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    return ax_inset
