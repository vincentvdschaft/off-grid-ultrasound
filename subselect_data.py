import h5py
from pathlib import Path

path = Path("data/S5-1_cardiac.hdf5")

with h5py.File(path, "r+") as f:
    image = f["data"]["image"][121]
    image = image[None]

    print(image.shape)

    # Remove the image dataset from the file
    del f["data"]["image"]

    # Create a new dataset with the same name
    f["data"]["image"] = image
