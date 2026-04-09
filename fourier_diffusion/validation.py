import numpy as np


def compute_error(
    reference_solution: np.ndarray,
    test_solution: np.ndarray,
) -> np.ndarray:
    """Return pointwise error = test - reference."""

    return test_solution - reference_solution


def compute_error_metrics(
    error: np.ndarray,
) -> tuple[float, float]:
    """Return max absolute error and RMS error."""

    max_abs_error = np.max(np.abs(error))
    rms_error = np.sqrt(np.mean(error**2))

    return max_abs_error, rms_error