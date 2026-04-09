import numpy as np

from fourier_diffusion import (
    generate_signal,
    generate_mode_indices,
    compute_coefficients,
    compute_series_terms,
    compute_series,
    compute_heat_equation_solution,
    plot_signals,
    plot_epicycles,
    plot_heat_equation_solution,
)


x_min = 0.0
x_max = 1.0
num_x = 1001

t_min = 0.0
t_max = 10.0
num_t = 1001

x_start = 0.0
x_end = 0.5
amplitude_min = -1.0
amplitude_max = 1.0

num_modes = 500
nu = 0.03
basis = "periodic"

sample_index = 35
partial_sums_index = -1

save_fig = False

x_array = np.linspace(x_min, x_max, num_x, endpoint=False)
time_array = np.linspace(t_min, t_max, num_t, endpoint=False)

signal_values = generate_signal(
    x_array, 
    x_start=x_start, 
    x_end=x_end, 
    amplitude_min=amplitude_min, 
    amplitude_max=amplitude_max
)

mode_indices = generate_mode_indices(num_modes, basis=basis)

mode_coefficients = compute_coefficients(
    signal_values, 
    x_array, 
    mode_indices, 
    basis=basis
)

series_terms = compute_series_terms(
    mode_indices, 
    mode_coefficients, 
    x_array, 
   
    basis=basis
)

partial_sums = compute_series(series_terms)

heat_equation_solution = compute_heat_equation_solution(
    series_terms, 
    mode_indices, 
    time_array, 
    nu, 
    basis=basis
)

plot_signals(x_array, signal_values, basis=basis, save_fig=save_fig)
plot_epicycles(series_terms, sample_index=sample_index, save_fig=save_fig)
plot_epicycles(series_terms, animate=True, save_fig=save_fig)
plot_signals(x_array, signal_values, partial_sums, basis=basis, save_fig=save_fig)
plot_signals(
    x_array, 
    signal_values, 
    partial_sums[partial_sums_index], 
    basis=basis, 
    save_fig=save_fig
)

plot_heat_equation_solution(
    x_array, time_array, 
    heat_equation_solution, 
    basis=basis, 
    animate=False, 
    save_fig=save_fig
)

plot_heat_equation_solution(
    x_array, 
    time_array, 
    heat_equation_solution, 
    basis=basis, 
    animate=True, 
    save_fig=save_fig)