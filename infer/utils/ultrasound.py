import jax.numpy as jnp
from jax import vmap

@vmap
def get_vfocus(focus_distance_m, polar_angle_rad, azimuthal_angle_rad=None):
    """Returns the position of the virtual focus in meters.
    
    Parameters
    ----------
    focus_distance_m : jnp.array
        The distances to the focus in meters.
    polar_angle_rad : jnp.array
        The polar angles in radians.
    azimuthal_angle_rad : jnp.array, optional
        The azimuthal angle in radians. If None, the function runs in 2D mode.
    
    Returns
    -------
    vfocus : jnp.array
        The position of the virtual focus in meters.
    """
    if azimuthal_angle_rad is None:
        pos_vfocus_m = jnp.array(
            [
                focus_distance_m * jnp.sin(polar_angle_rad),
                focus_distance_m * jnp.cos(polar_angle_rad),
            ]
        )
        return pos_vfocus_m
    
    pos_vfocus_m = jnp.array(
        [
            focus_distance_m * jnp.sin(polar_angle_rad) * jnp.cos(azimuthal_angle_rad),
            focus_distance_m * jnp.sin(polar_angle_rad) * jnp.sin(azimuthal_angle_rad),
            focus_distance_m * jnp.cos(polar_angle_rad),
        ]
    )
    return pos_vfocus_m