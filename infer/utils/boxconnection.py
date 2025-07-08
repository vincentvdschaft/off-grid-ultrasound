import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle


def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Parameters
    ----------
    points : list of tuples
        A list of (x, y) tuples representing the points.

    Returns
    -------
    hull : list of tuples
        A list of (x, y) tuples representing the points on the convex hull.
    """
    # Sort the points lexicographically (tuples are compared element-wise)
    points = sorted(points)

    # Function to determine the orientation of three points
    def orientation(p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    # Construct the lower hull
    lower_hull = []
    for point in points:
        while (
            len(lower_hull) >= 2
            and orientation(lower_hull[-2], lower_hull[-1], point) <= 0
        ):
            lower_hull.pop()
        lower_hull.append(point)

    # Construct the upper hull
    upper_hull = []
    for point in reversed(points):
        while (
            len(upper_hull) >= 2
            and orientation(upper_hull[-2], upper_hull[-1], point) <= 0
        ):
            upper_hull.pop()
        upper_hull.append(point)

    # Concatenate the lower and upper hull, removing the last point of each half because it is repeated
    return lower_hull[:-1] + upper_hull[:-1]


def create_random_bbox():
    x = np.random.randint(0, 100, 2)
    width = np.random.randint(1, 20)
    height = np.random.randint(1, 20)
    return Bbox([x, x + [width, height]])


def bbox_get_corners(bbox):
    """Get the corners of a matplotlib Bbox object."""
    return (
        (bbox.x0, bbox.y0),
        (bbox.x1, bbox.y0),
        (bbox.x1, bbox.y1),
        (bbox.x0, bbox.y1),
    )


def points_to_line_segments(points):
    """Converts a list of ordered points to a set of line segments."""
    return set(
        [(points[i], points[i + 1]) for i in range(len(points) - 1)]
        + [(points[-1], points[0])]
    )


def get_bbox_connections(bbox0, bbox1):
    """Finds the line segments that connect two bounding boxes."""
    corners0 = bbox_get_corners(bbox0)
    corners1 = bbox_get_corners(bbox1)

    # Combine all corners
    corners = corners0 + corners1

    # Get the convex hull
    hull = convex_hull(corners)

    hull_lines = points_to_line_segments(hull)
    bbox1_edges = points_to_line_segments(corners1)
    bbox0_edges = points_to_line_segments(corners0)

    connecting_lines = hull_lines - (bbox0_edges | bbox1_edges)

    return connecting_lines, hull
