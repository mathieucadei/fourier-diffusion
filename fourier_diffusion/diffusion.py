import numpy as np


def compute_heat_equation_solution(
    series_terms: np.ndarray, 
    mode_indices: np.ndarray, 
    time_array: np.ndarray, 
    nu: float, 
    basis: str="periodic"
) -> np.ndarray:
    """Compute the analytical heat equation solution from Fourier modes."""
    
    t = time_array[:, None, None]
    n = mode_indices[None, None, :]

    if basis == "periodic":
        decay = np.exp(-nu * ((2 * np.pi * n)**2 * t))
    
    elif basis == "cosine":
        decay = np.exp(-nu * ((np.pi * n)**2 * t))
    
    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")

    return np.sum(series_terms[None, :, :] * decay, axis=2).real
