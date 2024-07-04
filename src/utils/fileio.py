"""This module contains some convenience functions for file input/output."""

from pathlib import Path
from datetime import datetime


def create_unique_dir(parent_directory, name, prepend_date=False):
    """
    Creates a new directory with a unique id-number in the name.

    Parameters
    ----------
    parent_directory : str or Path
        The directory in which the file should be created
    name : str
        The desired directory name

    Returns
    -------
    Path
        The path of the newly created file
    """
    # Get the date string
    date_str = ""
    if prepend_date:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S_")

    # Create any parent directories if necessary
    Path(parent_directory).mkdir(parents=True, exist_ok=True)
    # Find the new filename
    file_path = get_unique_filename(parent_directory, date_str + name)
    # Create the directory
    file_path.mkdir(parents=True)

    return file_path


def get_unique_filename(parent_directory, name, extension=""):
    """Finds a unique file with a unique id-number in the name. If no files
    are present the id number will be 0. If there are files present the id
    number will be one larger than the largest one present. The name of the file will be
    turned to lowercase because the windows filesystem is case-insensitive.
    ```
    filename = get_unique_filename("data", "file", ".txt")
    ```

    Parameters
    ----------
    parent_directory : str or Path
        The directory in which the file should be created name
    filename : str or Path
        The desired filename including the file extension.
    name : str or Path
        The desired filename excluding the file extension.
    extension : str, default=""
        The file extension possibly including the dot.

    Returns
    -------
    Path
        The path of the newly created file
    """
    stem = Path(name).stem

    # Turn to lowercase
    stem = stem.lower()

    if len(extension) != 0 and extension[0] != ".":
        extension = "." + extension

    glob_pattern = name + "[0-9]" * 6 + extension
    # Find existing files matching name
    existing = list(Path(parent_directory).glob(glob_pattern))

    if len(existing) == 0:
        id = 0
    else:
        highest_id = 0
        for file in existing:
            name = str(file.stem)
            name = name[len(stem) : len(stem) + 6]
            highest_id = max(highest_id, int(name))
        id = highest_id + 1

    return Path(parent_directory, stem + str(id).zfill(6) + extension)


def get_latest_opt_vars(path):
    """Returns the path of the last file from the opt_vars folder, sorted
    alphabetically."""

    # Make the path absolute
    path = Path(path).resolve()

    opt_vars_folder = path / "opt_vars"

    if not opt_vars_folder.exists():
        raise FileNotFoundError(f"Folder {opt_vars_folder} does not exist.")

    files = sorted(opt_vars_folder.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No files found in {opt_vars_folder}.")

    return files[-1]
