import numpy as np
from src.methods.inverse_beamform import inverse_beamform

# The shape of the initial grid of scatterers (nz, nx)
grid_shape = (384, 256)

# The number of axial samples in the simulation
n_ax = 1024 + 256 + 64
n_steps = 2000
grid_dx_wl = 0.6
grid_dz_wl = 0.4

working_dir = inverse_beamform(
    hdf5_path=r"data/S5-1_phantom.hdf5",
    frame=0,
    selected_tx=np.array(
        [
            90,
        ]
    ),
    n_ax=n_ax,
    grid_shape=grid_shape,
    run_name="quick_example",
    n_steps=n_steps,
    wavefront_only=True,
    batch_size=128,
    gradient_accumulation=4,
    grid_dx_wl=grid_dx_wl,
    grid_dz_wl=grid_dz_wl,
)
