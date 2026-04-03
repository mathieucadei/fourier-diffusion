import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation

def generate_step_signal(time_array, time_span=1):

    signal_values = np.zeros_like(time_array)

    signal_values[time_array < time_span/2] = 1
    signal_values[time_array > time_span/2] = -1

    return signal_values


def generate_frequency_indices(num_frequencies):

    frequency_indices = np.empty(2 * num_frequencies + 1)   # make an empty array of the right length

    frequency_indices[0] = 0                              # first value is 0
    positive_indices = np.arange(1, num_frequencies + 1)   # [1, 2, 3, ..., num_frequencies]

    frequency_indices[1::2] = positive_indices                           # put positives at positions 1,3,5,...
    frequency_indices[2::2] = -positive_indices                          # put negatives at positions 2,4,6,...
    
    return frequency_indices


def compute_fourier_coefficients(signal_values, time_array, frequency_indices):

    time_step = time_array[1] - time_array[0]

    time_row = time_array[None, :]
    frequency_column = frequency_indices[:, None]

    complex_exponential = np.exp(
        -2 * np.pi * 1j * frequency_column * time_row
    )

    return (signal_values * complex_exponential).sum(axis=1) * time_step


def compute_fourier_series_terms(frequency_indices, fourier_coefficients_array, time_array):

    time_column = time_array[:, None]

    return fourier_coefficients_array*np.exp(frequency_indices*2*np.pi*1j*time_column)


def compute_fourier_series(fourier_series_terms):
    return np.sum(fourier_series_terms, axis=1)


def plot_signals(time_array, original_signal, fourier_signals=None, title="Signal vs time", x_label="Time", y_label="Amplitude"):
    """
    time_array: 1D array of x values.
    signals: list of 1D arrays, each same length as time_array.
    labels: list of strings, one per signal.
    title: plot title string.
    x_label: x axis label string.
    y_label: y axis label string.
    """
    plt.figure(figsize=(20, 12))

    plt.plot(time_array, original_signal, label="Original Signal", color='black')

    if fourier_signals is not None:
        if np.ndim(fourier_signals) == 1:
            fourier_signals = [fourier_signals]

        if len(fourier_signals) > 0:
            n = len(fourier_signals)
            cmap = cm.plasma

            if n == 1:
                plt.plot(time_array, fourier_signals[0], label="Fourier Signal")

            else:
                norm = colors.Normalize(vmin=0, vmax=n-1)  

                for i, y in enumerate(fourier_signals):
                    plt.plot(time_array, y, color=cmap(norm(i)))
                
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                plt.colorbar(sm, label="Frequency")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.show()


def plot_fourier_series_terms(fourier_series_terms, time_step=25, animate=False):

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

        U = fourier_series_terms[time_step].real
        V = fourier_series_terms[time_step].imag

        cumulative = np.cumsum(fourier_series_terms[time_step])

        x = np.concatenate(([0], cumulative.real[:-1]))
        y = np.concatenate(([0], cumulative.imag[:-1]))

        radius = np.abs(fourier_series_terms[time_step])

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

        plt.show()

if __name__ == "__main__":

    time_array = np.linspace(0, 1, 101)
    signal_values = generate_step_signal(time_array)
    
    plot_signals(time_array, signal_values)

    num_frequencies = 50
    frequency_indices = generate_frequency_indices(num_frequencies)

    fourier_coefficients = compute_fourier_coefficients(signal_values, time_array, frequency_indices)

    fourier_series_terms = compute_fourier_series_terms(frequency_indices, fourier_coefficients, time_array)

    plot_fourier_series_terms(fourier_series_terms, time_step=35)

    plot_fourier_series_terms(fourier_series_terms, animate=True)

    fourier_series_array = compute_fourier_series(fourier_series_terms)

    print(fourier_series_array)

    plot_signals(time_array, signal_values, fourier_series_array)

    # fourier_series_array_list = []
    # labels_list = []    

    # for i in range(0,51,1):

    #     num_frequencies = i
    #     frequency_indices = generate_frequency_indices(num_frequencies)
    #     fourier_coefficients = compute_fourier_coefficients(signal_values, time_array, frequency_indices)

    #     time_value_array = np.linspace(0, 1, 101)

    #     fourier_series_array = []

    #     labels_list.append(f"n = {i}")

    #     for j in time_value_array:
    #         fourier_series_term_array = compute_fourier_series_terms(frequency_indices, fourier_coefficients, time_value=j)
    #         fourier_series_array.append(compute_fourier_series(fourier_series_term_array))
        
    #     fourier_series_array_list.append(fourier_series_array)

    # fourier_series_array_list[-1], labels_list

    # plot_signals(time_array, signal_values, fourier_series_array_list)