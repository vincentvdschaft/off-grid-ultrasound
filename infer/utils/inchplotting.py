import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch, FancyArrow, Rectangle
from matplotlib.transforms import Bbox


class MPLFigure:
    """Figure class. Wraps a matplotlib figure object."""

    def __init__(self, figsize=None):
        self.fig = plt.figure(figsize=figsize)
        self.figsize = self.fig.get_size_inches()
        self.axes = []
        self.grids = []

    def add_ax(self, x0, y0, width, height, aspect=None):
        """Add an axis to the figure with the given bounding box in inches."""
        width, height = interpret_width_height_aspect(
            width=width, height=height, aspect=aspect
        )
        bbox_inches = Bbox.from_bounds(x0, y0, width, height)
        return self._add_ax(bbox_inches)

    def _add_ax(self, bbox_inches):
        """Add an axis to the figure with the given bounding box in inches."""
        bbox = self.bbox_norm(bbox_inches)
        ax = self.fig.add_axes(bbox)
        self.axes.append(ax)
        return ax

    def bbox_norm(self, bbox_inches):
        """Normalizes a bounding box in inches to a bounding box in normalized figure coordinates."""
        x0 = bbox_inches.x0 / self.figsize[0]
        y0 = 1.0 - bbox_inches.y1 / self.figsize[1]
        x1 = bbox_inches.x1 / self.figsize[0]
        y1 = 1.0 - bbox_inches.y0 / self.figsize[1]

        return Bbox(((x0, y0), (x1, y1)))

    def bbox_norm_inv(self, bbox_norm):
        """Inverse of bbox_norm."""
        x0 = bbox_norm.x0 * self.figsize[0]
        y0 = (1.0 - bbox_norm.y1) * self.figsize[1]
        x1 = bbox_norm.x1 * self.figsize[0]
        y1 = (1.0 - bbox_norm.y0) * self.figsize[1]
        return Bbox(((x0, y0), (x1, y1)))

    def coord_norm(self, x, y):
        """Converts a coordinate in inches to a coordinate in normalized figure coordinates."""
        x_norm = x / self.figsize[0]
        y_norm = 1.0 - y / self.figsize[1]
        return x_norm, y_norm

    def coord_norm_inv(self, x_norm, y_norm):
        """Inverse of coord_norm."""
        x = x_norm * self.figsize[0]
        y = (1.0 - y_norm) * self.figsize[1]
        return x, y

    def width_norm(self, width):
        """Converts a width in inches to a width in normalized figure coordinates."""
        return width / self.figsize[0]

    def height_norm(self, height):
        """Converts a height in inches to a height in normalized figure coordinates."""
        return height / self.figsize[1]

    def get_ax_bbox(self, ax):
        bbox = ax.get_position()
        return self.bbox_norm_inv(bbox)

    def add_axes_grid(
        self,
        n_rows,
        n_cols,
        x0,
        y0,
        spacing,
        im_width=None,
        im_height=None,
        im_aspect=None,
    ):
        """Add a grid of axes to the figure. The bottom left corner of the grid is at (x0, y0) in inches."""

        im_width, im_height = interpret_width_height_aspect(
            width=im_width, height=im_height, aspect=im_aspect
        )

        axes_array = np.empty((n_rows, n_cols), dtype=object)
        for row in range(n_rows):
            for col in range(n_cols):
                bbox_inches = Bbox.from_bounds(
                    x0 + (im_width + spacing) * col,
                    y0 + (im_height + spacing) * row,
                    im_width,
                    im_height,
                )
                ax = self._add_ax(bbox_inches)
                axes_array[row, col] = ax
        self.grids.append(axes_array)

        return axes_array

    def get_total_bbox(self, margin=0.2):
        """Get the total bounding box of the figure in inches."""
        x0 = y0 = x1 = y1 = None

        target_axes = set(self.axes)

        def nonemin(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        def nonemax(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return max(a, b)

        for ax in target_axes:
            bbox = ax.get_position()
            x0 = nonemin(x0, bbox.x0)
            y0 = nonemin(y0, bbox.y0)
            x1 = nonemax(x1, bbox.x1)
            y1 = nonemax(y1, bbox.y1)

        # ----------------------------------------------------------------------
        # Add the margin
        # ----------------------------------------------------------------------
        bbox = Bbox(((x0, y0), (x1, y1)))
        print(f"bbox before: {bbox}")
        bbox = add_margin_to_bbox(bbox, margin)
        print(f"bbox after: {bbox}")

        bbox = Bbox(((bbox.x0, 1.0 - bbox.y1), (bbox.x1, 1.0 - bbox.y0)))
        return self.bbox_norm_inv(bbox)

    def add_text(self, x, y, text, **kwargs):
        """Add text to the figure."""
        x_norm = x / self.figsize[0]
        y_norm = 1.0 - y / self.figsize[1]
        return self.fig.text(x_norm, y_norm, text, **kwargs)

    def savefig(self, *args, margin=0.3, **kwargs):
        """Save the figure."""
        if not "bbox_inches" in kwargs:
            kwargs["bbox_inches"] = self.get_total_bbox(margin=margin)
        self.fig.savefig(*args, **kwargs)

    def get_ax_width(self, ax):
        """Returns the width of the Axes in inches.

        Returns
        -------
        width : float
            The width of the figure in inches.
        """
        bbox = ax.get_position()
        width = bbox.x1 - bbox.x0
        width *= self.figsize[0]
        return width

    def get_ax_height(self, ax):
        """Returns the height of the Axes in inches.

        Returns
        -------
        height : float
            The height of the figure in inches.
        """
        bbox = ax.get_position()
        height = bbox.y1 - bbox.y0
        height *= self.figsize[1]
        return height

    def get_ax_position(self, ax):
        """Returns the position of the Axes in figure coordinates.

        Returns
        -------
        x0, y0 : float, float
            The position of the Axes in figure coordinates.
        """
        bbox = ax.get_position()
        bbox = self.bbox_norm_inv(bbox)
        return bbox.x0, bbox.y0

    def data_to_figure_coords(self, ax, x, y):
        """"""
        return data_to_figure_coords(fig=self.fig, ax=ax, x=x, y=y)

    def add_inset_plot(
        self,
        ax,
        width=None,
        height=None,
        aspect=None,
        position="top left",
        margin=0.2,
    ):
        """"""
        parent_x0, parent_y0 = self.get_ax_position(ax)
        parent_width = self.get_ax_width(ax)
        parent_height = self.get_ax_height(ax)
        width, height = interpret_width_height_aspect(
            width=width, height=height, aspect=aspect
        )
        if position == "top left":
            x0 = parent_x0 + margin
            y0 = parent_y0 + margin
        elif position == "top right":
            x0 = parent_x0 + parent_width - width - margin
            y0 = parent_y0 + margin
        elif position == "bottom left":
            x0 = parent_x0 + margin
            y0 = parent_y0 + parent_height - height - margin
        elif position == "bottom right":
            x0 = parent_x0 + parent_width - width - margin
            y0 = parent_y0 + parent_height - height - margin
        else:
            raise ValueError(f"Invalid position: {position}")

        ax = self.add_ax(x0=x0, y0=y0, width=width, height=height)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def add_arrow(self, ax, data_x, data_y, angle_deg, length_inches=1, **kwargs):
        """Add an arrow to the figure."""
        fig_x, fig_y = self.data_to_figure_coords(ax, data_x, data_y)
        dx = -length_inches * np.cos(angle_deg * np.pi / 180)
        dy = -length_inches * np.sin(angle_deg * np.pi / 180)

        dx = self.width_norm(dx)
        dy = self.height_norm(dy)
        return self.fig.add_artist(
            FancyArrow(
                fig_x - dx, fig_y - dy, dx, dy, length_includes_head=True, **kwargs
            )
        )

    def add_rectangle_ax(
        self, ax, data_x, data_y, data_width, data_height, *args, **kwargs
    ):
        """Add a rectangle to the figure in the data coordinates of the Axes."""
        fig_x0, fig_y0 = self.data_to_figure_coords(ax, data_x, data_y)
        fig_x1, fig_y1 = self.data_to_figure_coords(
            ax, data_x + data_width, data_y + data_height
        )
        fig_width = fig_x1 - fig_x0
        fig_height = fig_y1 - fig_y0

        return self.fig.add_artist(
            Rectangle(
                (fig_x0, fig_y0),
                fig_width,
                fig_height,
                *args,
                **kwargs,
            )
        )

    def add_colorbar(
        self,
        x0,
        y0,
        width,
        height,
        cmap,
        vmin,
        vmax,
        ticks,
        orientation="vertical",
        **kwargs,
    ):
        """Adds a colorbar to the figure."""
        ax_cbar = self.add_ax(
            x0=x0,
            y0=y0,
            width=width,
            height=height,
        )

        colorbar = matplotlib.colorbar.ColorbarBase(
            ax_cbar,
            cmap=plt.get_cmap(cmap),
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
            orientation=orientation,
            ticks=ticks,
            **kwargs,
        )
        self.cbar_axes.append(ax_cbar)
        self.colorbars.append(colorbar)
        return colorbar

    def draw_bbox_connection(self, bbox0, bbox1, *args, **kwargs):
        from .boxconnection import get_bbox_connections

        bbox0 = self.bbox_norm(bbox0)
        bbox1 = self.bbox_norm(bbox1)
        lines, hull = get_bbox_connections(bbox0, bbox1)

        # Draw the lines
        for line in lines:
            line = list(line)
            self.fig.add_artist(
                ConnectionPatch(line[0], line[1], "figure fraction", *args, **kwargs)
            )


def interpret_width_height_aspect(width, height, aspect):
    """Interprets the width, height, and aspect parameters to form just a width and height. If aspect is provided, either as a float or and extent, one of the other two parameters can be inferred.

    Parameters
    ----------
    width : float
        The width of the ax. May be ommitted if height and aspect are provided.
    height : float
        The height of the ax. May be ommitted if width and aspect are provided.
    aspect : float or size 4 tuple/list/array
        The aspect ratio (delta y)/(delta x) or extent from which the aspect ratio can
        be computed.

    Returns
    -------
    width, height : float, float
        The width and height of the ax.
    """
    if width is not None and height is not None:
        return width, height

    aspect = extent_to_ratio(aspect)

    if width is None:
        assert height is not None, "Either width or height should be specified."
        width = height / aspect
    else:
        height = width * aspect

    return float(width), float(height)


def remove_internal_ticks(grid):
    """Remove internal ticks from a grid of axes."""
    n_rows, n_cols = grid.shape
    for row in range(n_rows):
        for col in range(n_cols):
            ax = grid[row, col]
            if row != n_rows - 1:
                ax.set_xticks([])
            if col != 0:
                ax.set_yticks([])


def remove_internal_labels(grid):
    """Remove internal labels from a grid of axes."""
    n_rows, n_cols = grid.shape
    for row in range(n_rows):
        for col in range(n_cols):
            ax = grid[row, col]
            if row != n_rows - 1:
                ax.set_xlabel("")
            if col != 0:
                ax.set_ylabel("")


def extent_to_ratio(extent):
    """Converts an extent [x0, x1, y0, y1] to a ratio dy/dx. If the input is already a single number it is just returned."""
    if isinstance(extent, (float, int)):
        return extent
    assert isinstance(extent, (tuple, list, np.ndarray))
    extent = list(extent)
    extent = _sort_extent(extent)
    width = extent[1] - extent[0]
    height = extent[3] - extent[2]
    try:
        ratio = height / width
    except ZeroDivisionError as exc:
        raise ZeroDivisionError("Width of extent cannot be 0!") from exc
    return ratio


def _sort_extent(extent):
    x0 = min(extent[0], extent[1])
    x1 = max(extent[0], extent[1])
    y1 = max(extent[2], extent[3])
    y0 = min(extent[2], extent[3])
    return np.array([x0, x1, y0, y1])


def data_to_figure_coords(fig, ax, x, y):
    coords = x, y
    # Transform (x, y) from data coordinates to display coordinates
    # coords = ax.transAxes.transform(coords)
    coords = ax.transData.transform(coords)

    # Transform display coordinates to figure coordinates
    fig_x, fig_y = fig.transFigure.inverted().transform(coords)

    return fig_x, fig_y


def add_margin_to_bbox(bbox, margin):
    """Adds a margin to a bounding box.

    Parameters
    ----------
    bbox : Bbox
        The bounding box.
    margin : float or Bbox
        The margin to add to the bounding box. If a float, the same margin is added to all sides. If a Bbox, the margin is added to each side separately.

    Returns
    -------
    bbox : Bbox
        The bounding box with the margin added.
    """
    if isinstance(margin, (float, int)):
        margin = Bbox([[margin, margin], [margin, margin]])
    elif isinstance(margin, (list, tuple, np.ndarray)):
        margin = Bbox(margin)
    x0 = bbox.x0 - margin.x0
    y0 = bbox.y0 - margin.y0
    x1 = bbox.x1 + margin.x1
    y1 = bbox.y1 + margin.y1
    return Bbox([[x0, y0], [x1, y1]])
