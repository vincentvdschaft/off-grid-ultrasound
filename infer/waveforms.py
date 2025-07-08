import jax.numpy as jnp
from jax import jit
from jax.scipy.ndimage import map_coordinates
import numpy as np
from itertools import product


def get_sampled(
    waveform_samples: jnp.array,
    sampling_frequency: float,
):
    """Returns a function that behaves like the interpolated waveform.
    The function takes a single time value and returns the interpolated
    value of the waveform at that time. To use the function for multiple time values,
    use `vmap`.

    Parameters
    ----------
    waveform_samples : np.array
        The samples of shape `(n_samp)`.

    Returns
    -------
    func_sampled : function
        The function that takes a single time value and returns the interpolated value
        of the waveform at that time.

    Example
    -------
    >>> sampling_frequency = 60
    >>> t = np.arange(200)/sampling_frequency
    >>> waveform_samples = np.sin(2*np.pi*3*t)
    >>> func_sampled = get_sampled(waveform_samples, sampling_frequency)
    >>> t = np.arange(200)/(2*sampling_frequency)
    >>> values = vmap(func_sampled)(t)
    """

    assert isinstance(waveform_samples, (np.ndarray, jnp.ndarray))
    assert waveform_samples.ndim == 1
    assert waveform_samples.dtype in (np.float32, np.float64)

    waveform_samples = jnp.array(waveform_samples)
    sampling_frequency = float(sampling_frequency)

    # Pad the waveform samples with a single zero at the beginning and end
    waveform_samples = jnp.concatenate(
        [jnp.array([0.0]), waveform_samples, jnp.array([0.0])]
    )

    n_samples = waveform_samples.shape[0]

    def func_sampled(t: float):

        # Convert time to sample index
        float_sample = t * sampling_frequency

        # Add one sample to compensate for the padding
        float_sample += 1

        # Clip the samples to the range of the signal
        float_sample = jnp.clip(float_sample, 0, n_samples - 1)

        # Convert to integer samples around the float sample
        n_min = jnp.floor(float_sample).astype(jnp.int32)
        n_max = jnp.ceil(float_sample).astype(jnp.int32)

        # Get the values of the signal at the integer samples
        sample_min_a = waveform_samples[n_min]
        sample_max_a = waveform_samples[n_max]

        # Find the start of the interpolation
        sample_start = sample_min_a
        sample_diff = sample_max_a - sample_min_a

        # Interpolate between the two samples
        interpolated = sample_start + sample_diff * (float_sample - n_min)

        return interpolated

    func_sampled = jit(func_sampled)

    return func_sampled


def get_sampled_multi(
    waveform_samples: jnp.array,
    sampling_frequency: float,
):
    """Returns a function that behaves like a collection of interpolated sampled
    waveforms. The function takes a single time value and returns the interpolated
    value of the waveform at that time. To use the function for multiple time values,
    use `vmap`.

    Parameters
    ----------
    waveform_samples : np.array
        The samples of shape `(n_tw, n_samp)`.

    Returns
    -------
    func_sampled : function
        The function that takes a single time value and returns the interpolated value
        of the waveform at that time.

    Example
    -------
    >>> sampling_frequency = 60
    >>> t = np.arange(200)/sampling_frequency
    >>> waveform_samples = np.sin(2*np.pi*3*t)
    >>> func_sampled = get_sampled(waveform_samples, sampling_frequency)
    >>> t = np.arange(200)/(2*sampling_frequency)
    >>> values = vmap(func_sampled)(t)
    """

    assert isinstance(waveform_samples, (np.ndarray, jnp.ndarray))
    assert waveform_samples.ndim == 2
    assert waveform_samples.dtype in (np.float32, np.float64)

    waveform_samples = jnp.array(waveform_samples)
    sampling_frequency = float(sampling_frequency)

    # Pad the waveform samples with a single zero at the beginning and end
    n_tw = waveform_samples.shape[0]
    waveform_samples = jnp.concatenate(
        [jnp.zeros((n_tw, 1)), waveform_samples, jnp.zeros((n_tw, 1))], axis=1
    )

    n_samples = waveform_samples.shape[1]

    def func_sampled(tw: int, t: float):

        # Convert time to sample index
        float_sample = t * sampling_frequency

        # Add one sample to compensate for the padding
        float_sample += 1

        # Clip the samples to the range of the signal
        float_sample = jnp.clip(float_sample, 0, n_samples - 1)

        # Convert to integer samples around the float sample
        n_min = jnp.floor(float_sample).astype(jnp.int32)
        n_max = n_min + 1

        # Get the values of the signal at the integer samples
        sample_min_a = waveform_samples[tw, n_min]
        sample_max_a = waveform_samples[tw, n_max]

        # Find the
        sample_start = sample_min_a
        sample_diff = sample_max_a - sample_min_a

        # Interpolate between the two samples
        interpolated = sample_start + sample_diff * (float_sample - n_min)

        return interpolated

    func_sampled = jit(func_sampled)

    return func_sampled

def waveform_sampled(waveform_samples, tw, t):
    
    n_tw, n_samp = waveform_samples.shape
    t_point_samples = t * 250e6
    
    point = jnp.stack([t_point_samples], axis=0)
    sample = map_coordinates(waveform_samples[tw], point, order=1)
    return sample
    


def combine_waveform_samples(waveform_samples_list):
    """Combines a list of waveform samples into a single numpy array. The size will be
    the maximum size of the waveform samples in the list.

    Parameters
    ----------
    waveform_samples_list : list
        A list of waveform samples of shape `(n_samp)`.

    Returns
    -------
    waveform_samples : np.array
        The waveform samples of shape `(n_tw, n_samp)`.
    """

    n_tw = len(waveform_samples_list)

    # Determine the maximum number of samples among the waveforms in the list
    n_samp = max(
        waveform_samples.shape[0] for waveform_samples in waveform_samples_list
    )

    # Create an array to store the waveform samples
    waveform_samples = np.zeros((n_tw, n_samp))

    # Copy the waveform samples into the array
    for i, current_waveform in enumerate(waveform_samples_list):
        current_size = current_waveform.shape[0]
        waveform_samples[i, :current_size] = current_waveform

    return waveform_samples


from scipy.signal import butter, filtfilt


def get_waveform_directivity_cube(
    waveform_samples: jnp.array,
    sampling_frequency: float,
    sound_speed_mps: float,
    element_width: float,
    n_angles: int,
):
    """_summary_

    Args:
        waveform_samples : jnp.array
            The matrix of waveform samples as returned by `combine_waveform_samples`.
        sampling_frequency : float
            The sampling frequency of the waveform samples.
        sound_speed_mps : float
            The speed of sound in meters per second.
        element_width : float
            The width of the transducer element in meters.
        n_angles : int
            The number of angles to sample the directivity function at.
            The angles will sampled as `np.asin(np.linspace(0, 1, n_angles))`.

    Returns:
        The waveform samples of shape `(n_tw, n_angles, n_samp)`. The first dimension
        is the transmit waveform index, which is used in cases where there are multiple
        different transmit waveforms. The second is the angle index, which always
        starts at 0 degrees and goes up to 90 degrees. The third dimension is the
        sample index.
    """
    assert isinstance(waveform_samples, (np.ndarray, jnp.ndarray))
    assert waveform_samples.ndim == 2
    assert n_angles > 1

    n_tw, n_samp = waveform_samples.shape

    waveform_samples_cube = np.zeros((n_tw, n_angles, n_angles, n_samp))

    # sin_theta_vals = np.sin(np.linspace(0, np.pi / 2, n_angles))
    sin_theta_vals = np.linspace(0, 1, n_angles)

    for tw in range(n_tw):
        for (index_angle_tx, sin_theta_tx), (index_angle_rx, sin_theta_rx) in product(
            enumerate(sin_theta_vals[:-1]), enumerate(sin_theta_vals[:-1])
        ):
            if index_angle_tx == index_angle_rx == 0:
                waveform_samples_cube[tw, 0, 0] = waveform_samples[tw]
                continue

            filtered = filter_directivity(
                waveform_samples[tw],
                sampling_frequency,
                sound_speed_mps,
                element_width,
                sin_theta_tx,
            )
            filtered = filter_directivity(
                filtered,
                sampling_frequency,
                sound_speed_mps,
                element_width,
                sin_theta_rx,
            )
            waveform_samples_cube[tw, index_angle_tx, index_angle_rx] = (
                filtered
                * jnp.cos(jnp.asin(sin_theta_tx))
                * jnp.cos(jnp.asin(sin_theta_rx))
            )

    return waveform_samples_cube


def filter_directivity(
    waveform_samples, sampling_frequency, sound_speed_mps, element_width, sin_theta
):
    n_samp = waveform_samples.shape[0]
    n_limit = np.clip(
        int(0.5 * sampling_frequency * element_width * sin_theta / sound_speed_mps * 2),
        1,
        None,
    )
    filter_taps = np.zeros(n_samp)
    filter_taps[:n_limit] = 1
    filter_taps = filter_taps / np.sum(filter_taps)
    filtered = np.convolve(waveform_samples, filter_taps)
    filtered = filtered[:n_samp]
    return filtered


# def waveform_sampled(waveform_directivity_cube, tw, t, sin_angle_rad_tx, sin_angle_rad_rx):
#     """Returns an interpolated sample of the waveform at a given time and angle.

#     Parameters
#     ----------
#     waveform_directivity_cube : jnp.array
#         The waveform samples of shape `(n_tw, n_angles, n_samp)`. The first dimension is the transmit waveform index, which is used in cases where there are multiple different transmit waveforms. The second is the angle index, which always starts at 0 degrees and goes up to 90 degrees. The third dimension is the sample index.
#     tw : int
#         The transmit waveform index.
#     t : float
#         The time at which to sample the waveform.
#     angle_rad : float
#         The angle at which to sample the waveform.
#     """

#     waveform_samples = waveform_directivity_cube[tw]
#     sampling_frequency = 250e6

#     n_angles = waveform_samples.shape[0]
#     t_point_samples = t * sampling_frequency
#     sin_angle_point_tx_steps = sin_angle_rad_tx * n_angles
#     sin_angle_point_rx_steps = sin_angle_rad_rx * n_angles
    

#     point = jnp.stack([sin_angle_point_tx_steps, sin_angle_point_rx_steps, t_point_samples], axis=0)

#     sample = map_coordinates(waveform_samples, point, order=1)

#     return sample