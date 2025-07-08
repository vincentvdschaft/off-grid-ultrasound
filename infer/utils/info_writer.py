from pathlib import Path
import yaml

info_dict = {}
info_path = None


def set_path(path):
    """Set the path to write the info_dict to."""
    global info_path
    
    if path is None:
        info_path = None
        return
    
    info_path = Path(path)
    assert info_path.suffix == ".yaml", "info_path should be a .yaml file"
    info_path.parent.mkdir(parents=True, exist_ok=True)


def write_info(key, value):
    """Add a key-value pair to the info_dict and write it to the info_path if it is defined."""
    info_dict[key] = value

    _dump_info()


def _dump_info():
    """Write the info_dict to the info_path if it is defined."""

    if info_path is None:
        return

    with open(info_path, "w", encoding="utf-8") as f:
        yaml.dump(info_dict, f)
