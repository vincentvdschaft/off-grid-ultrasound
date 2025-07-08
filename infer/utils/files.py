import os
import shutil
import zipfile
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List, Tuple, Union
from uuid import uuid4

import yaml


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
    if prepend_date:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S_")
    else:
        date_str = ""
    path = Path(parent_directory, date_str + name)

    # Create any parent directories if necessary
    Path(parent_directory).mkdir(parents=True, exist_ok=True)
    # Find the new filename
    file_path = add_number_if_exists(path)
    # Create the directory
    file_path.mkdir(parents=True)

    return file_path


def add_number_if_exists(path):
    """If a file with the same name exists, add a number to the filename.
    If this new name also exists, increment the number until a unique name is found.

    Parameters
    ----------
    path : str or Path
        The path to the file to check for existence.

    Returns
    -------
    file_path : Path
        The new path that is guaranteed to be unique.
    """
    path = Path(path)

    if not path.exists():
        return path

    stem = path.stem
    extension = path.suffix
    parent = path.parent

    n = 0
    zfill_len = 6
    current_path = parent / (stem + f"{str(n).zfill(zfill_len)}" + extension)
    while current_path.exists():
        current_path = parent / (stem + f"{str(n).zfill(zfill_len)}" + extension)
        # Increment the number
        n += 1
        # Increase the zfill length if necessary
        if n > 10**zfill_len - 1:
            zfill_len += 1

    return current_path


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
    file_path : Path
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
        id_nr = 0
    else:
        highest_id = 0
        for file in existing:
            name = str(file.stem)
            name = name[len(stem) : len(stem) + 6]
            highest_id = max(highest_id, int(name))
        id_nr = highest_id + 1

    return Path(parent_directory, stem + str(id_nr).zfill(6) + extension)


def save_dict_as_yaml(dict, path):
    with open(path, "w") as file:
        yaml.dump(dict, file, default_flow_style=False)


def create_week_folder(parent_directory: Path) -> Path:
    """Creates a new directory with the current week number in the name."""
    week_nr = datetime.now().isocalendar()[1]

    # Create the new directory
    new_dir = Path(parent_directory, f"week-{week_nr}")
    new_dir.mkdir(parents=True, exist_ok=True)

    return new_dir


def create_date_folder(parent_directory: Path) -> Path:
    """Creates a new directory with the current date in the name."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Create the new directory
    new_dir = Path(parent_directory, date_str)
    new_dir.mkdir(parents=True, exist_ok=True)

    return new_dir


def copy_repo(
    source_dir: list[Path],
    source_dirs_recursive: list[Path],
    destination_dir: Path,
    extensions: list[str] = None,
):
    """Copies all files with the given extensions from the source directory to
    a new `source_code` folder in the destination directory.
    The directory structure is preserved.

    Parameters
    ----------
    source_dir : Path
        The root directory to copy files from.
    source_dirs_recursive : List[Path]
        The directories to copy files from recursively.
    destination_dir : Path
        The directory to copy files to. A file in the root of the source directory will
        be copied to `destination_dir/source_code`.
    extensions : list[str]
        The file extensions to copy. If None, the default extensions are
        [".py", ".yaml", ".yml"].
    """

    # Default extensions
    if extensions is None:
        extensions = [".py", ".yaml", ".yml"]

    # Create the destination directory
    source_code_dir = destination_dir / "source_code"
    source_code_dir.mkdir(parents=True, exist_ok=True)

    globs_recursive = [p.glob("**/*") for p in source_dirs_recursive]

    # Copy the files
    for file in chain(source_dir.glob("*"), *globs_recursive):
        if file.is_file() and file.suffix in extensions:
            # Create the new directory
            new_dir = source_code_dir / file.relative_to(source_dir).parent
            new_dir.mkdir(parents=True, exist_ok=True)
            # Copy the file
            new_file = new_dir / file.name
            shutil.copy2(file, new_file)


def get_source_dir():
    """Returns the path to the source directory by moving up until a folder with a
    `conftest.py` file is found."""
    path = Path(__file__).resolve()

    # Move up until a folder with `conftest.py` is found
    while not (path / "conftest.py").exists():
        if path == path.parent:
            raise FileNotFoundError("Could not find the source directory.")

        path = path.parent

    return path


def make_read_only(path: Union[Path, str], recursive: bool = False):
    """Makes a file read-only."""
    path = Path(path)

    if path.is_file():
        path.chmod(0o444)
    elif path.is_dir():
        path.chmod(0o555)
        if recursive:
            for child in path.iterdir():
                make_read_only(child, recursive=True)


def resolve_data_path(path):
    """Interprets a data path string and returns a Path object to an existing file or
    raises an error.

    Notes
    -----
    The function goes through the following steps:
    1. Check if the path is absolute. If it is, return the path.
    2. Check if the path is relative to the current working directory. If it is, return
       the path.
    3. Check if the path is relative to the DATA_ROOT environment variable.
         If it is, return the path.
    4. Raise a FileNotFoundError.
    """

    assert isinstance(path, (str, Path)), "The path must be a string or a Path object."

    # Check if the path is absolute
    if Path(path).is_absolute():
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"The file {data_path} does not exist.")
        return data_path

    # Check if the path is relative to the current working directory
    data_path = Path.cwd() / path

    if data_path.exists():
        return data_path

    # Check if the path is relative to the DATA_ROOT
    verasonics_root = os.environ.get("DATA_ROOT", "")

    if verasonics_root == "":
        raise FileNotFoundError(
            "The file does not exist relative to the working dir and the "
            "DATA_ROOT environment variable is not set."
        )

    data_path = Path(verasonics_root) / path
    data_path = data_path.resolve()

    if data_path.exists():
        return data_path

    data_path = Path(verasonics_root) / "usbmd" / path
    data_path = data_path.resolve()

    if data_path.exists():
        return data_path

    data_path = Path(verasonics_root) / "verasonics" / path
    data_path = data_path.resolve()

    if data_path.exists():
        return data_path

    raise FileNotFoundError(
        f"The file {path} does not exist relative to the working dir or relative to the DATA_ROOT."
    )


def find_result(
    directory_name: Union[str, Path], search_directory: Union[str, Path, list[Path]]
):
    """Finds the result directory with the given name in the search directory or directories.

    Parameters
    ----------
    directory_name : str or Path
        The name of the directory to find (e.g. `20240913_161317_infer_run`).
    search_directory : str or Path or list[Path]
        The directory or directories to search in.

    Returns
    -------
    Path
        The path to the result directory.
    """
    if isinstance(search_directory, (str, Path)):
        search_directory = [search_directory]

    for directory in search_directory:
        for result in Path(directory).glob(f"**/{directory_name}"):
            return result

    raise FileNotFoundError(f"Could not find the directory {directory_name}.")


def zip_directory(directory_path, zip_file_path):
    """
    Zips the contents of a directory.

    Parameters:
    - directory_path (str): The path to the directory to zip.
    - zip_file_path (str): The path to the output zip file.
    """
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the directory tree
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Create the complete file path
                file_path = os.path.join(root, file)
                # Write the file to the zip archive
                zip_file.write(file_path, os.path.relpath(file_path, directory_path))
