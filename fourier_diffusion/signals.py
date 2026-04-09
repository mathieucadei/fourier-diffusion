import numpy as np


def generate_signal(
    x_array: np.ndarray, 
    x_start: float=0, 
    x_end: float=0.5, 
    amplitude_min: float=-1, 
    amplitude_max: float=1
) -> np.ndarray:
    """Generate a piecewise constant signal defined on the interval [0, 1)."""

    signal_values = np.zeros_like(x_array)

    signal_values[(x_array < x_start) | (x_array > x_end)] = amplitude_min
    signal_values[(x_array >= x_start) & (x_array <= x_end)] = amplitude_max

    return signal_values