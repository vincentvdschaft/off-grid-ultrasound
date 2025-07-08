from dataclasses import asdict, dataclass, fields
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jaxus import compute_lensed_travel_time_2d
from tqdm import tqdm

from infer.waveforms import waveform_sampled


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "probe_geometry_m",
        "initial_times_s",
        "tgc_gain_curve",
        "sampling_frequency_hz",
        "center_frequency_hz",
        "tw_indices",
        "sound_speed_mps",
        "t0_delays_tx_s",
        "active_element_idx",
        "tx_apodizations_tx",
        "element_width_m",
        "pos_vfocus_m",
        "tx_travel_texture",
        "rx_travel_texture",
        "target_region_m",
    ],
    meta_fields=[
        "waveform_samples",
        "forward_model_type",
        "slope",
        "intercept",
        "n_grad_scat",
        "scat_amp_reparameterization",
        "apply_lens_correction",
        "lens_thickness",
        "lens_sound_speed_mps",
        "optimize_scatterer_positions",
        "enable_wavelength_scaling",
        "hash",
        "symlog_epsilon",
    ],
)
@dataclass
class ForwardSettings:
    """Container class to pass tensors to the forward model. This class is registered
    with JAX to be considered an internal node in the pytree structure. This allows
    it to work with `jit` and `vmap`.

    This class is introduced to avoid passing a large number of arguments to the
    forward model functions."""

    probe_geometry_m: jnp.array
    initial_times_s: jnp.array
    tgc_gain_curve: jnp.array
    sampling_frequency_hz: jnp.array
    center_frequency_hz: jnp.array
    tw_indices: jnp.array
    sound_speed_mps: jnp.array
    t0_delays_tx_s: jnp.array
    active_element_idx: jnp.array
    tx_apodizations_tx: jnp.array
    element_width_m: float
    pos_vfocus_m: jnp.array
    # Auxiliary variables
    waveform_samples: Callable
    forward_model_type: str
    slope: jnp.array
    n_grad_scat: int
    intercept: jnp.array
    scat_amp_reparameterization: bool
    apply_lens_correction: bool
    lens_thickness: float
    lens_sound_speed_mps: float
    tx_travel_texture: jnp.array
    rx_travel_texture: jnp.array
    target_region_m: jnp.array
    hash: int = 0
    optimize_scatterer_positions: bool = True
    symlog_epsilon: float = 0.01
    enable_wavelength_scaling: bool = True

    def __dict__(self):
        self_dict = asdict(self)
        # Remove the waveform function
        # self_dict.pop("waveform_fn")

        return self_dict

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash

    def add_hash(self):
        """Computes a hash from all the data fields."""
        hash_str = ""
        arrays = []
        self_dict = asdict(self)
        # Remove hash item
        self_dict.pop("hash")
        for key, value in self_dict.items():
            if key in ("rx_travel_texture", "tx_travel_texture"):
                continue
            if isinstance(value, jnp.ndarray):
                arrays.append(value.reshape(-1))
            else:
                hash_str += str(value)

        # Concatenate all arrays
        arrays = jnp.concatenate(arrays)
        hash_str += "".join([str(n) for n in arrays])
        self.hash = hash(hash_str)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "sound_speed_offset_mps",
        "gain",
        "angle_scaling",
        "initial_times_shift_s",
    ],
    meta_fields=[],
)
@dataclass
class OptVars:
    """Container class to hold the optimization variables."""

    sound_speed_offset_mps: jnp.array
    gain: jnp.array
    angle_scaling: jnp.array
    initial_times_shift_s: jnp.array

    def __dict__(self):
        return asdict(self)

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return [f.name for f in fields(self)]


def reparameterize_opt_vars(opt_vars_opt: OptVars):
    """Reparameterizes the optimization variables to ensure they are positive."""

    return OptVars(
        sound_speed_offset_mps=opt_vars_opt.sound_speed_offset_mps * 300,
        gain=opt_vars_opt.gain,
        angle_scaling=opt_vars_opt.angle_scaling,
        initial_times_shift_s=opt_vars_opt.initial_times_shift_s * 1e-6,
    )


def inverse_reparameterize_opt_vars(opt_vars: OptVars):
    """Inverse reparameterizes the optimization variables to ensure they are positive."""

    return OptVars(
        sound_speed_offset_mps=opt_vars.sound_speed_offset_mps / 300,
        gain=opt_vars.gain,
        angle_scaling=opt_vars.angle_scaling,
        initial_times_shift_s=opt_vars.initial_times_shift_s / 1e-6,
    )


def symlog(x, epsilon=0.01):
    """Symmetric logarithmic scale function."""
    x = x / epsilon
    return jnp.where(
        jnp.abs(x) < epsilon,
        x,
        jnp.sign(x) * (jnp.log(jnp.abs(x) + 1)),
    )


def symexp(y, epsilon=0.01):
    """Inverse of the symmetric logarithmic scale function."""
    return jnp.sign(y) * (jnp.exp(jnp.abs(y)) - 1) * epsilon


@partial(jit, static_argnums=(1,))
def reparameterize_scat_amp(scat_amp_opt, repar_type="positive", symlog_epsilon=0.01):
    """Reparameterizes the scatterer amplitudes.

    Parameters
    ----------
    scat_amp_opt : jnp.array
        The scatterer amplitudes.
    type : str, optional
        The type of reparameterization. Can be "positive", "symlog", or "none", by default "positive".
    """
    if repar_type == "positive":
        return jnp.exp(scat_amp_opt)
    elif repar_type == "symlog":
        return symexp(scat_amp_opt, symlog_epsilon)

    return scat_amp_opt


def inverse_reparameterize_scat_amp(
    scat_amp, repar_type="positive", symlog_epsilon=0.01
):
    """Inverse eparameterizes the scatterer amplitudes.

    Parameters
    ----------
    scat_amp_opt : jnp.array
        The scatterer amplitudes.
    type : str, optional
        The type of reparameterization. Can be "positive", "symlog", or "none", by default "positive".
    """
    if repar_type == "positive":
        return jnp.log(scat_amp)
    elif repar_type == "symlog":
        return symlog(scat_amp, symlog_epsilon)

    return scat_amp


def reparameterize_scat_pos(scat_pos_opt, forward_settings, opt_vars_opt):
    """Reparameterizes the scatterer position"""
    opt_vars = reparameterize_opt_vars(opt_vars_opt)
    if forward_settings.enable_wavelength_scaling:
        sound_speed = forward_settings.sound_speed_mps - opt_vars.sound_speed_offset_mps
        wavelength = sound_speed / forward_settings.center_frequency_hz
        scaling = wavelength
    else:
        scaling = 1e-3

    return scat_pos_opt * scaling


def inverse_reparameterize_scat_pos(scat_pos_m, forward_settings, opt_vars):
    """Inverse reparameterizes the scatterer position"""
    if forward_settings.enable_wavelength_scaling:
        sound_speed = forward_settings.sound_speed_mps - opt_vars.sound_speed_offset_mps
        wavelength = sound_speed / forward_settings.center_frequency_hz
        scaling = 1 / wavelength
    else:
        scaling = 1e3
    return scat_pos_m * scaling


@jit
def directivity(theta, element_width_wl=0.5):
    """Computes the directivity of a single element at a given angle. This is a scaling
    factor that is 1 at angle 0.

    Parameters
    ----------
    theta : jnp.array
        The angle of the direction of interest in radians.
    element_width_wl : float, optional
        The width of the element in wavelengths, by default 0.5.

    Returns
    -------
    directivity : jnp.array
        The directivity of the element at the given angle.
    """
    return (jnp.sinc(element_width_wl * jnp.sin(theta))) * jnp.cos(theta)


def directivity_2d(element_pos_m, scat_pos_m, element_width_wl=0.5, angle_scaling=1.0):
    """Computes the directivity of a single element at a given angle. This is a scaling
    factor that is 1 at angle 0.

    Parameters
    ----------
    element_pos_m : jnp.array
        The position of the element in meters.
    scat_pos_m : jnp.array
        The position of the scatterer in meters.
    element_width_wl : float, optional
        The width of the element in wavelengths, by default 0.5.

    Returns
    -------
    directivity : jnp.array
        The directivity of the element at the given angle.
    """
    theta = jnp.arctan2(
        scat_pos_m[0] - element_pos_m[0], scat_pos_m[-1] - element_pos_m[-1]
    )

    return directivity(theta, element_width_wl)  # * angle_scaling)


directivity_2d_array = vmap(directivity_2d, in_axes=(0, None, None, None))


def directivity_3d(element_pos_m, scat_pos_m, element_width_wl=0.5):
    """Computes the directivity of a single element at a given angle. This is a scaling
    factor that is 1 at angle 0.

    Parameters
    ----------
    element_pos_m : jnp.array
        The position of the element in meters.
    scat_pos_m : jnp.array
        The position of the scatterer in meters.
    element_width_wl : float, optional
        The width of the element in wavelengths, by default 0.5.

    Returns
    -------
    directivity : jnp.array
        The directivity of the element at the given angle.
    """
    theta = jnp.arctan2(
        scat_pos_m[0] - element_pos_m[0], scat_pos_m[-1] - element_pos_m[-1]
    )
    phi = jnp.arctan2(
        scat_pos_m[1] - element_pos_m[1], scat_pos_m[-1] - element_pos_m[-1]
    )

    return directivity(theta, element_width_wl) * directivity(phi, element_width_wl)


directivity_3d_array = vmap(directivity_3d, in_axes=(0, None, None))


def forward_model(
    tx: int,
    ax: int,
    ch: int,
    scat_pos_opt,
    scat_amp_opt,
    forward_settings: ForwardSettings,
    opt_vars_opt: OptVars,
):
    """Simulates a single rf sample."""
    probe_geometry_m = forward_settings.probe_geometry_m
    t0_delays_tx_s = forward_settings.t0_delays_tx_s
    active_element_idx = forward_settings.active_element_idx
    initial_times_s = forward_settings.initial_times_s
    tgc_gain_curve = forward_settings.tgc_gain_curve
    sampling_frequency_hz = forward_settings.sampling_frequency_hz
    tw_indices = forward_settings.tw_indices
    # waveform_fn = forward_settings.waveform_fn
    waveform_samples = forward_settings.waveform_samples
    sound_speed_mps = forward_settings.sound_speed_mps
    tx_apodizations_tx = forward_settings.tx_apodizations_tx
    pos_vfocus_m = forward_settings.pos_vfocus_m
    element_width_m = forward_settings.element_width_m
    wavelength = sound_speed_mps / (0.25 * sampling_frequency_hz)
    element_width_wl = element_width_m / wavelength
    apply_lens_correction = forward_settings.apply_lens_correction
    lens_thickness = forward_settings.lens_thickness
    lens_sound_speed_mps = forward_settings.lens_sound_speed_mps

    # Apply reparameterization to the optimization variables
    opt_vars = reparameterize_opt_vars(opt_vars_opt)

    sound_speed_mps = forward_settings.sound_speed_mps - opt_vars.sound_speed_offset_mps

    scat_amp = reparameterize_scat_amp(
        scat_amp_opt,
        forward_settings.scat_amp_reparameterization,
        forward_settings.symlog_epsilon,
    )

    scat_pos_m = reparameterize_scat_pos(scat_pos_opt, forward_settings, opt_vars_opt)

    if probe_geometry_m.shape[1] == 2:
        in_3d = False
    else:
        in_3d = True

    rcv_element_position = probe_geometry_m[ch]

    # Compute the time the sample was recorded
    t_sample = (
        ax / sampling_frequency_hz
        + initial_times_s[tx]
        # + opt_vars.initial_times_shift_s
    )

    directivity_tx = 1.0

    if forward_settings.forward_model_type == "wavefront_only_general_interpolated":
        tau_rx = rx_travel_time(
            scat_pos_m,
            rcv_element_position,
            forward_settings.rx_travel_texture,
            forward_settings.target_region_m,
        )
        tau_tx = tx_travel_time(
            scat_pos_m,
            forward_settings.tx_travel_texture[tx],
            extent_m=forward_settings.target_region_m,
        )
        tau = tau_tx + tau_rx
        dist_rx = tau_rx * sound_speed_mps
        dist_tx = tau_tx * sound_speed_mps

        directivity_tx = 1.0
    elif forward_settings.forward_model_type == "fully_general_interpolated":
        tau_rx = rx_travel_time(
            scat_pos_m,
            rcv_element_position,
            forward_settings.rx_travel_texture,
            forward_settings.target_region_m,
        )

        tau_tx = (
            vmap(rx_travel_time, in_axes=(None, 0, None, None))(
                scat_pos_m,
                active_element_idx[tx],
                forward_settings.rx_travel_texture,
                forward_settings.target_region_m,
            )
            + t0_delays_tx_s[tx]
        )
        tau = tau_tx + tau_rx
        dist_rx = tau_rx * sound_speed_mps
        dist_tx = tau_tx * sound_speed_mps

        if in_3d:
            directivity_tx = directivity_3d_array(
                active_element_idx[tx], scat_pos_m, element_width_wl
            )
        else:
            directivity_tx = directivity_2d_array(
                active_element_idx[tx],
                scat_pos_m,
                element_width_wl,
                opt_vars.angle_scaling,
            )

    else:
        # Compute the travel distances to the scatterer and receiving element
        if forward_settings.forward_model_type == "virtual_source":
            dist_tx = jnp.linalg.norm(pos_vfocus_m[tx] - scat_pos_m)
            tau_tx = dist_tx / sound_speed_mps

        else:
            if apply_lens_correction:
                dist_tx = (
                    vmap(
                        compute_lensed_travel_time,
                        in_axes=(0, None, None, None, None, None),
                    )(
                        active_element_idx[tx],
                        scat_pos_m,
                        lens_thickness,
                        lens_sound_speed_mps,
                        sound_speed_mps,
                        1,
                    )
                    * sound_speed_mps
                )
            else:
                dist_tx = jnp.linalg.norm(
                    active_element_idx[tx] - scat_pos_m[None], axis=1
                )

            tau_tx = dist_tx / sound_speed_mps + t0_delays_tx_s[tx]
        if apply_lens_correction:
            dist_rx = (
                compute_lensed_travel_time(
                    rcv_element_position,
                    scat_pos_m,
                    lens_thickness,
                    lens_sound_speed_mps,
                    sound_speed_mps,
                    1,
                )
                * sound_speed_mps
            )
        else:
            dist_rx = jnp.linalg.norm(rcv_element_position - scat_pos_m)

        # Compute the travel times
        tau_rx = dist_rx / sound_speed_mps

        # If wavefront_only is True, we only consider the first arrival
        if forward_settings.forward_model_type == "wavefront_only_general":
            tau_tx = jnp.min(tau_tx)
            dist_tx = jnp.min(dist_tx)
            directivity_tx = 1.0
        elif forward_settings.forward_model_type == "virtual_source":
            directivity_tx = 1.0
        elif forward_settings.forward_model_type == "fully_general":
            if in_3d:
                directivity_tx = directivity_3d_array(
                    active_element_idx[tx], scat_pos_m, element_width_wl
                )
            else:
                directivity_tx = directivity_2d_array(
                    active_element_idx[tx],
                    scat_pos_m,
                    element_width_wl,
                    opt_vars.angle_scaling,
                )

        # Compute the total travel time
        tau = tau_tx + tau_rx

    # Compute the sample by indexing the waveform function
    sample = waveform_sampled(waveform_samples, tw_indices[tx], t_sample - tau)

    # Apply attenuation due to spread
    sample *= jnp.clip(1e-3 / dist_tx, None, 1)

    # Apply the gain
    sample *= opt_vars.gain[ch]

    # Scale with the directivity of the element
    if in_3d:
        directivity_rx = directivity_3d(
            rcv_element_position, scat_pos_m, element_width_wl
        )
    else:
        directivity_rx = directivity_2d(
            rcv_element_position, scat_pos_m, element_width_wl, opt_vars.angle_scaling
        )

    sample *= directivity_tx * directivity_rx

    # Apply the transmit apodization
    if forward_settings.forward_model_type == "fully_general":
        sample *= tx_apodizations_tx[tx]

    # Sum over all transmit elements
    sample = jnp.sum(sample)

    # Scale with the scatterer amplitude
    sample *= scat_amp

    # Apply attenuation due to spread
    sample *= jnp.clip(1e-3 / dist_rx, None, 1)

    # Apply the TGC gain curve
    sample *= tgc_gain_curve[ax]

    return sample


def forward_model_all_scat(
    tx, ax, ch, scat_pos_opt, scat_amp_opt, forward_settings, opt_vars_opt
):
    """Simulates a single rf sample for all scatterers."""

    # Define the axes that are batched over
    axis_tx = None
    axis_ax = None
    axis_ch = None
    axis_scat_pos_m = 0
    axis_scat_amp = 0
    axis_forward_settings = None
    axis_opt_vars = None

    # Vectorize the forward model function
    vmapped_fn = vmap(
        forward_model,
        in_axes=(
            axis_tx,
            axis_ax,
            axis_ch,
            axis_scat_pos_m,
            axis_scat_amp,
            axis_forward_settings,
            axis_opt_vars,
        ),
    )

    n_grad_scat = forward_settings.n_grad_scat
    if n_grad_scat > 0:
        # Compute the depth of the sample with respect to the origin
        depth = (
            ax
            / forward_settings.sampling_frequency_hz
            * forward_settings.sound_speed_mps
            / 2
        )
        # Compute the index
        index = (depth - forward_settings.intercept) / forward_settings.slope
        index = index.astype(int)

        # index = jnp.clip(index, 0, scat_amp_opt.shape[0] - 1)

        index0 = jnp.clip(index - int(n_grad_scat * 0.8), 0, scat_amp_opt.shape[0] - 1)
        index1 = jnp.clip(index0 + n_grad_scat, 0, scat_amp_opt.shape[0] - 1)
        index0 = jnp.clip(index1 - n_grad_scat, 0, scat_amp_opt.shape[0] - 1)

        # scat_pos = jnp.zeros((n_grad_scat, scat_pos_opt.shape[1]))
        # scat_amp = jnp.zeros(n_grad_scat)

        scat_pos_opt = jax.lax.dynamic_slice_in_dim(
            scat_pos_opt, index0, n_grad_scat, axis=0
        )
        scat_amp_opt = jax.lax.dynamic_slice_in_dim(
            scat_amp_opt, index0, n_grad_scat, axis=0
        )

    response = vmapped_fn(
        tx, ax, ch, scat_pos_opt, scat_amp_opt, forward_settings, opt_vars_opt
    )

    # Call the vectorized function and sum over all scatterers
    sum_over_all_scatterers = jnp.sum(
        response,
        axis=0,
    )

    return sum_over_all_scatterers


# @partial(jit, static_argnames="forward_settings")
def forward_model_multi_sample(
    tx, ax, ch, scat_pos_opt, scat_amp_opt, forward_settings, opt_vars_opt
):
    """Simulates a batch of rf samples for all scatterers."""

    # axis_tx = 0
    # axis_ax = 0
    # axis_ch = 0
    # axis_scat_pos_m = None
    # axis_scat_amp = None
    # axis_forward_settings = None
    # axis_opt_vars = None

    # # Vectorize the forward model function
    # vmapped_fn = vmap(
    #     forward_model_all_scat,
    #     in_axes=(
    #         axis_tx,
    #         axis_ax,
    #         axis_ch,
    #         axis_scat_pos_m,
    #         axis_scat_amp,
    #         axis_forward_settings,
    #         axis_opt_vars,
    #     ),
    # )

    # depths = jnp.linalg.norm(scat_pos_opt, axis=1)
    # ax_depths =

    # Call the vectorized function and sum over all scatterers
    return batched_forward_model(
        tx, ax, ch, scat_pos_opt, scat_amp_opt, forward_settings, opt_vars_opt
    )


def forward_model_batched(
    tx,
    ax,
    ch,
    scat_pos_opt,
    scat_amp_opt,
    forward_settings,
    opt_vars_opt,
    batch_size=32,
):
    """Simulates a batch of rf samples for all scatterers."""
    print("WARNING: This function is deprecated. Use execute_batched instead.")

    n_samples = tx.shape[0]

    # Split the samples into batches
    batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        batches += 1

    # Allocate the output
    output = []

    # Iterate over the batches
    for i in range(batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)

        # Compute the output for the batch
        new_output = forward_model_multi_sample(
            tx[start:end],
            ax[start:end],
            ch[start:end],
            scat_pos_opt,
            scat_amp_opt,
            forward_settings,
            opt_vars_opt,
        )

        # Append the output to the list
        output.append(new_output)
    output = jnp.concatenate(output, axis=0)

    return output


def execute_batched(fn, batch_size, kwarg_dict, batched_kwarg_dict, progress_bar=True):
    """Executes a function in batches over the batched arguments."""

    n_samples = batched_kwarg_dict[list(batched_kwarg_dict.keys())[0]].shape[0]

    # Split the samples into batches
    batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        batches += 1

    # Allocate the output
    output = []

    # Iterate over the batches
    for i in tqdm(range(batches), desc=fn.__name__, disable=not progress_bar):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)

        batch_kwargs = {}
        for key, value in batched_kwarg_dict.items():
            batch_kwargs[key] = value[start:end]

        # Compute the output for the batch
        new_output = fn(**batch_kwargs, **kwarg_dict)

        # Append the output to the list
        output.append(new_output)

    # Stack or concatenate the output
    if output[0].ndim == 0:
        output = jnp.stack(output)
    else:
        output = jnp.concatenate(output, axis=0)

    return output


def compute_rx_travel_texture(
    extent_m, shape, sound_speed_mps, lens_sound_speed_mps, lens_thickness_m
):
    x_vals = jnp.linspace(extent_m[0], extent_m[1], shape[0])
    z_vals = jnp.linspace(extent_m[2], extent_m[3], shape[1])

    x, z = jnp.meshgrid(x_vals, z_vals, indexing="ij")

    x, z = x.flatten(), z.flatten()

    n_iter = 5

    travel_times = vmap(
        compute_lensed_travel_time, in_axes=(None, 0, None, None, None, None)
    )(
        jnp.array([0.0, 0.0]),
        jnp.stack([x, z], axis=-1),
        lens_thickness_m,
        lens_sound_speed_mps,
        sound_speed_mps,
        n_iter,
    )

    return travel_times.reshape(shape)


@partial(jit, static_argnums=(0, 1, 2, 3, 4))
def compute_tx_travel_texture(
    extent_m,
    shape,
    sound_speed_mps,
    lens_sound_speed_mps,
    lens_thickness_m,
    t0_delays_tx_s,
    active_element_idx,
):
    x_vals = jnp.linspace(extent_m[0], extent_m[1], shape[0])
    z_vals = jnp.linspace(extent_m[2], extent_m[3], shape[1])

    x, z = jnp.meshgrid(x_vals, z_vals, indexing="ij")

    x, z = x.flatten(), z.flatten()

    n_iter = 5

    travel_times = (
        vmap(
            vmap(compute_lensed_travel_time, in_axes=(None, 0, None, None, None, None)),
            in_axes=(0, None, None, None, None, None),
        )(
            active_element_idx,
            jnp.stack([x, z], axis=-1),
            lens_thickness_m,
            lens_sound_speed_mps,
            sound_speed_mps,
            n_iter,
        )
        + t0_delays_tx_s[:, None]
    )

    travel_times = jnp.min(travel_times, axis=0)

    return travel_times.reshape(shape)


@jit
def tx_travel_time(scat_pos_m, tx_travel_texture, extent_m):
    texture_start = jnp.array([extent_m[0], extent_m[2]])
    texture_size = jnp.array(
        [
            extent_m[1] - extent_m[0],
            extent_m[3] - extent_m[2],
        ]
    )
    normalized_scat_pos = (
        (scat_pos_m - texture_start) / texture_size * tx_travel_texture.shape[0]
    )
    tau_tx = jax.scipy.ndimage.map_coordinates(
        tx_travel_texture,
        normalized_scat_pos,
        order=1,
    )
    return tau_tx


@jit
def rx_travel_time(scat_pos_m, rcv_element_pos_m, rx_travel_texture, extent_m):
    texture_start = jnp.array([extent_m[0], extent_m[2]])
    texture_size = jnp.array(
        [
            extent_m[1] - extent_m[0],
            extent_m[3] - extent_m[2],
        ]
    )

    normalized_scat_pos = (
        (scat_pos_m - rcv_element_pos_m - texture_start)
        / texture_size
        * rx_travel_texture.shape[0]
    )
    tau_rx = jax.scipy.ndimage.map_coordinates(
        rx_travel_texture, normalized_scat_pos, order=1
    )
    return tau_rx


from jax.scipy.ndimage import map_coordinates


# @partial(jit, static_argnums=(5,))
def batched_forward_model(
    tx: jnp.array,
    ax: jnp.array,
    ch: jnp.array,
    scat_pos_opt,
    scat_amp_opt,
    forward_settings: ForwardSettings,
    opt_vars_opt: OptVars,
):
    """Simulates a batch of rf samples for all scatterers."""

    opt_vars = reparameterize_opt_vars(opt_vars_opt)
    scat_pos = reparameterize_scat_pos(scat_pos_opt, forward_settings, opt_vars_opt)
    scat_amp = reparameterize_scat_amp(
        scat_amp_opt,
        forward_settings.scat_amp_reparameterization,
        forward_settings.symlog_epsilon,
    )
    sound_speed = forward_settings.sound_speed_mps - opt_vars.sound_speed_offset_mps
    wavelength_lens = (
        forward_settings.lens_sound_speed_mps / forward_settings.center_frequency_hz
    )

    if forward_settings.apply_lens_correction:
        dist = (
            vmap(
                vmap(
                    compute_lensed_travel_time_2d,
                    in_axes=(0, None, None, None, None, None),
                ),
                in_axes=(None, 0, None, None, None, None),
            )(
                forward_settings.probe_geometry_m,
                scat_pos,
                forward_settings.lens_thickness,
                forward_settings.lens_sound_speed_mps,
                sound_speed,
                0,
            )
            * sound_speed
        )
    else:
        dist = jnp.linalg.norm(
            forward_settings.probe_geometry_m[None] - scat_pos[:, None], axis=-1
        )

    dist_tx = dist.at[:, forward_settings.active_element_idx[tx]].get()
    tau_tx = dist_tx / sound_speed + forward_settings.t0_delays_tx_s[tx]

    dist_rx = dist[:, ch]
    tau_rx = dist_rx[..., None] / sound_speed

    if forward_settings.forward_model_type == "wavefront_only_general":
        tau_tx = jnp.min(tau_tx, axis=2)[..., None]
    # Attenuation due to spread
    att_rx = jnp.clip(1e-3 / dist_rx, None, 1)
    att_tx = jnp.clip(1e-3 / dist_tx, None, 1)


    # [scat, samp, txel]
    tau = tau_tx + tau_rx

    t = ax / forward_settings.sampling_frequency_hz

    sample_times = t[None, :, None] - tau

    waveform_idx = forward_settings.tw_indices[tx]
    tx_points = (jnp.ones_like(sample_times) * waveform_idx[None, :, None]).ravel()

    sample_points = jnp.stack([tx_points, sample_times.ravel() * 250e6], axis=0)

    samples = map_coordinates(
        forward_settings.waveform_samples, sample_points, 1
    ).reshape(tau.shape)

    theta = jnp.arctan2(
        scat_pos[:, None, 0] - forward_settings.probe_geometry_m[None, :, 0],
        scat_pos[:, None, 1] - forward_settings.probe_geometry_m[None, :, 1],
    )

    directivity_all = directivity(
        theta, forward_settings.element_width_m / wavelength_lens
    )
    if forward_settings.forward_model_type == "wavefront_only_general":
        directivity_tx = directivity_all[
            :, forward_settings.probe_geometry_m.shape[0] // 2
        ][:, None, None]
    else:
        directivity_tx = directivity_all[:, forward_settings.active_element_idx[tx]]
    directivity_rx = directivity_all[:, ch]

    samples = (
        jnp.sum(samples * directivity_tx * att_tx, axis=-1)
        * directivity_rx
        * att_rx
        * scat_amp[:, None]
        * forward_settings.tgc_gain_curve[ax][None]
    )

    return jnp.sum(samples, axis=0)

    # theta_tx = jnp.arctan2(
    #     scat_pos[:, None, 0]
