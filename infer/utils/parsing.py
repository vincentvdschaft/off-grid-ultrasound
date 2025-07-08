import numpy as np


def parse_range(input_str: str, max_index: int = None) -> np.array:
    """Parses a range string into a list of integers. The string is any number of
    indices (3), ranges (1-3), or a single 'all'. 'all' can only be used if `max_range`
    is provided. These are combined into a single numpy array of indices in increasing
    order.

    Parameters
    ----------
    input_str : str
        The string to parse.
    max_index : int, default=None
        The maximum index.

    Returns
    -------
    indices : np.array
        The indices as a numpy array.
    """
    indices = set()

    input_str = str(input_str)

    # Remove all commas.
    input_str = input_str.replace(",", "")
    input_str = input_str.replace("[", "")
    input_str = input_str.replace("]", "")

    for part in input_str.split():
        if part == "all":
            if max_index is None:
                raise ValueError("Cannot use 'all' without specifying max_index.")
            indices.update(range(max_index))
        elif "-" in part:
            start, end = part.split("-")
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))

    # Ensure that the indices are within the range.
    if max_index is not None:
        indices = indices.intersection(range(max_index))

    return np.array(sorted(indices))
