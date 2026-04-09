from .signals import generate_signal
from .fourier import (
    generate_mode_indices,
    compute_coefficients,
    compute_series_terms,
    compute_series,
)
from .diffusion import compute_heat_equation_solution
from .numerical import solve_heat_equation_explicit
from .validation import compute_error, compute_error_metrics
from .plotting import (
    plot_signals,
    plot_epicycles,
    plot_heat_equation_solution,
)