from .signals import generate_signal
from .fourier import (
    generate_mode_indices,
    compute_coefficients,
    compute_series_terms,
    compute_series,
)
from .diffusion import compute_heat_equation_solution
from .plotting import (
    plot_signals,
    plot_epicycles,
    plot_heat_equation_solution,
)