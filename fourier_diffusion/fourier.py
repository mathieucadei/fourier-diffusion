import numpy as np


def generate_mode_indices(num_modes: int, basis: str="periodic") -> np.ndarray:
    """Generate mode indices for the selected basis."""

    if basis == "periodic":

        mode_indices = np.empty(2 * num_modes + 1, dtype=int)
        mode_indices[0] = 0
        
        positive_indices = np.arange(1, num_modes + 1)

        mode_indices[1::2] = positive_indices
        mode_indices[2::2] = -positive_indices
    
        return mode_indices

    elif basis == "cosine":
        return np.arange(0, num_modes + 1, dtype=int)
    
    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")


def compute_coefficients(
    signal_values: np.ndarray, 
    x_array: np.ndarray, 
    mode_indices: np.ndarray, 
    basis: str="periodic"
) -> np.ndarray:
    """Compute Fourier coefficients from a sampled signal."""


    dx = x_array[1] - x_array[0]

    x_row = x_array[None, :]
    mode_column = mode_indices[:, None]

    if basis == "periodic":
        basis_matrix = np.exp(-2j * np.pi * mode_column * x_row)
        return (signal_values * basis_matrix).sum(axis=1) * dx
     
    elif basis == "cosine":
        basis_matrix = np.cos(np.pi * mode_column * x_row)
        coefficients = (signal_values * basis_matrix).sum(axis=1) * dx
        coefficients[1:] = 2 * coefficients[1:]
        return coefficients

    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")


def compute_series_terms(
    mode_indices: np.ndarray, 
    coefficients: np.ndarray, 
    x_array: np.ndarray, 
    basis: str="periodic"
) -> np.ndarray:
    """Compute the contribution of each mode at each x value."""

    x_column = x_array[:, None]

    if basis == "periodic":
        return coefficients * np.exp(mode_indices * 2j * np.pi * x_column)
    
    elif basis == "cosine":
        return coefficients * np.cos(np.pi * mode_indices * x_column)

    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")


def compute_series(series_terms: np.ndarray) -> np.ndarray:
    """Return cumulative partial sums of the Fourier series."""

    return np.cumsum(series_terms, axis=1).T