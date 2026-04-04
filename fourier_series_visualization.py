import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
import os
from pathlib import Path
import sys

def generate_step_signal(x_array, domain_length=1):

    signal_values = np.zeros_like(x_array)

    signal_values[x_array < domain_length/2] = 1
    signal_values[x_array > domain_length/2] = -1

    return signal_values


def generate_frequency_indices(num_frequencies):

    frequency_indices = np.empty(2 * num_frequencies + 1)   # make an empty array of the right length

    frequency_indices[0] = 0                              # first value is 0
    positive_indices = np.arange(1, num_frequencies + 1)   # [1, 2, 3, ..., num_frequencies]

    frequency_indices[1::2] = positive_indices                           # put positives at positions 1,3,5,...
    frequency_indices[2::2] = -positive_indices                          # put negatives at positions 2,4,6,...
    
    return frequency_indices


def compute_fourier_coefficients(signal_values, x_array, frequency_indices):

    dx = x_array[1] - x_array[0]

    x_row = x_array[None, :]
    frequency_column = frequency_indices[:, None]

    complex_exponential = np.exp(
        -2 * np.pi * 1j * frequency_column * x_row
    )

    return (signal_values * complex_exponential).sum(axis=1) * dx


def compute_fourier_series_terms(frequency_indices, fourier_coefficients_array, x_array):

    x_column = x_array[:, None]

    return fourier_coefficients_array*np.exp(frequency_indices*2*np.pi*1j*x_column)


def compute_fourier_series(fourier_series_terms):
    return np.cumsum(fourier_series_terms, axis=1).T


def plot_signals(x_array, original_signal, fourier_signals=None, title="Signal vs x", x_label="x", y_label="Amplitude"):
    """
    x_array: 1D array of x values.
    signals: list of 1D arrays, each same length as x_array.
    labels: list of strings, one per signal.
    title: plot title string.
    x_label: x axis label string.
    y_label: y axis label string.
    """
    plt.figure(figsize=(20, 12))

    plt.plot(x_array, original_signal, label="Original Signal", color='black')

    if fourier_signals is not None:
        if np.ndim(fourier_signals) == 1:
            fourier_signals = [fourier_signals]

        if len(fourier_signals) > 0:
            n = len(fourier_signals)
            cmap = cm.plasma

            if n == 1:
                plt.plot(x_array, fourier_signals[0], label="Fourier Signal")

            else:
                norm = colors.Normalize(vmin=0, vmax=n-1)  

                for i, y in enumerate(fourier_signals):
                    plt.plot(x_array, y, color=cmap(norm(i)))
                
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                ax = plt.gca()  # Get current axes
                plt.colorbar(sm, ax=ax, label="Frequency")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.show()


def plot_fourier_series_terms(fourier_series_terms, dx=25, animate=False):

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Fourier series vector sum (epicycles)")

    all_cumulative = np.cumsum(fourier_series_terms, axis=1)
    limit = 1.2 * np.max(np.abs(all_cumulative))

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_aspect("equal")

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: f"{val:g}i")
    )

    ax.grid(True)

    if animate is False:

        U = fourier_series_terms[dx].real
        V = fourier_series_terms[dx].imag

        cumulative = np.cumsum(fourier_series_terms[dx])

        x = np.concatenate(([0], cumulative.real[:-1]))
        y = np.concatenate(([0], cumulative.imag[:-1]))

        radius = np.abs(fourier_series_terms[dx])

        ax.quiver(
            x,
            y,
            U,
            V,
            angles='xy',
            scale_units='xy',
            scale=1
        )

        for xi, yi, ri in zip(x, y, radius):
            circle = Circle(
                (xi, yi),
                ri,
                color='r',
                fill=False
            )
            ax.add_patch(circle)

        plt.show()
        
    else:

        U = fourier_series_terms[0].real
        V = fourier_series_terms[0].imag

        cumulative = np.cumsum(fourier_series_terms[0])

        x = np.concatenate(([0], cumulative.real[:-1]))
        y = np.concatenate(([0], cumulative.imag[:-1]))

        centers = list(zip(x, y))

        radius = np.abs(fourier_series_terms[0])

        circles = []

        qvr = ax.quiver(
            x,
            y,
            U,
            V,
            angles='xy',
            scale_units='xy',
            scale=1
        )

        for ci, ri in zip(centers, radius):
            circle = Circle(
                ci,
                ri,
                color='r',
                fill=False
            )

            ax.add_patch(circle)

            circles.append(circle)

        def update(frame):

            nonlocal qvr

            qvr.remove()

            U = fourier_series_terms[frame].real
            V = fourier_series_terms[frame].imag

            cumulative = np.cumsum(fourier_series_terms[frame])

            x = np.concatenate(([0], cumulative.real[:-1]))
            y = np.concatenate(([0], cumulative.imag[:-1]))

            centers = list(zip(x, y))

            qvr = ax.quiver(
                x,
                y,
                U,
                V,
                angles='xy',
                scale_units='xy',
                scale=1
            )

            for ci, circle in zip(centers, circles):
                circle.center = (ci)

            return (qvr, *circles)
        
        ani = FuncAnimation(fig, update, frames=len(fourier_series_terms), interval=100, blit=False)

        os.makedirs("animations", exist_ok=True)

        variable_name = "fourier_series_terms"
        script_name = Path(__file__).stem

        ani.save(f"animations/{variable_name}.mp4", writer="ffmpeg")

        plt.show()

if __name__ == "__main__":

    x_array = np.linspace(0, 1, 1001, endpoint=False)
    signal_values = generate_step_signal(x_array)

    plot_signals(x_array, signal_values)

    num_frequencies = 500
    frequency_indices = generate_frequency_indices(num_frequencies)

    fourier_coefficients = compute_fourier_coefficients(signal_values, x_array, frequency_indices)

    fourier_series_terms = compute_fourier_series_terms(frequency_indices, fourier_coefficients, x_array)

    plot_fourier_series_terms(fourier_series_terms, dx=35)

    plot_fourier_series_terms(fourier_series_terms, animate=True)

    fourier_series_array = compute_fourier_series(fourier_series_terms)

    plot_signals(x_array, signal_values, fourier_series_array)

    plot_signals(x_array, signal_values, fourier_series_array[-1])

    time_array = np.linspace(0, 1, 1001, endpoint=False)

    nu = 0.05

    # heat_equation_solution = np.exp(-nu * 4 * np.pi**2 * frequency_indices[:, None]**2 * time_array[None, :]) * fourier_series_array

    k = frequency_indices[None, None, :]          # shape (1, 1, n_k)
    x = x_array[None, :, None]                    # shape (1, n_x, 1)
    t = time_array[:, None, None]                 # shape (n_t, 1, 1)
    c_k = fourier_coefficients[None, None, :]     # shape (1, 1, n_k)

    heat_equation_solution = np.sum(
        c_k
        * np.exp(2j * np.pi * k * x)
        * np.exp(-nu * (2 * np.pi * k)**2 * t),
        axis=2
    ).real



    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d') 
    x = x_array
    y = time_array
    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(
    X,
    Y,
    heat_equation_solution,
    cmap=cm.viridis
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("z")

    plt.show()