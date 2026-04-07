import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
import os

def generate_step_signal(x_array, domain_length=1):

    signal_values = np.zeros_like(x_array)

    signal_values[x_array < domain_length/2] = 1
    signal_values[x_array > domain_length/2] = -1

    return signal_values


def generate_mode_indices(num_modes, basis="periodic"):

    if basis == "periodic":

        mode_indices = np.empty(2 * num_modes + 1, dtype=int)   # make an empty array of the right length
        mode_indices[0] = 0                                     # first value is 0
        
        positive_indices = np.arange(1, num_modes + 1)          # [1, 2, 3, ..., num_modes]

        mode_indices[1::2] = positive_indices                   # put positives at positions 1,3,5,...
        mode_indices[2::2] = -positive_indices                  # put negatives at positions 2,4,6,...
    
        return mode_indices

    elif basis == "cosine":
        return np.arange(0, num_modes + 1, dtype=int)
    
    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")


def compute_coefficients(signal_values, x_array, mode_indices, basis="periodic"):

    dx = x_array[1] - x_array[0]

    x_row = x_array[None, :]
    mode_column = mode_indices[:, None]

    if basis == "periodic":
        basis_matrix  = np.exp(
            -2j * np.pi * mode_column * x_row
        )
    
    elif basis == "cosine":
        basis_matrix = np.cos(np.pi * mode_column * x_row)

    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")
    
    return (signal_values * basis_matrix).sum(axis=1) * dx


def compute_series_terms(mode_indices, coefficients_array, x_array, basis="periodic"):

    x_column = x_array[:, None]

    if basis == "periodic":
        return coefficients_array*np.exp(mode_indices * 2j * np.pi * x_column)
    
    elif basis == "cosine":
        return coefficients_array*np.cos(np.pi * mode_indices * x_column)

    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")


def compute_series(series_terms):
    return np.cumsum(series_terms, axis=1).T


def compute_heat_equation_solution(series_terms, mode_indices, time_array, nu, basis="periodic"):
    
    t = time_array[:, None, None]
    n = mode_indices[None, None, :]

    if basis == "periodic":
        decay = np.exp(-nu * ((2 * np.pi * n)**2 * t))
    
    elif basis == "cosine":
        decay = np.exp(-nu * ((np.pi * n)**2 * t))
    
    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")

    return np.sum(series_terms[None, :, :] * decay, axis=2).real


def plot_signals(x_array, original_signal, series_signals=None, title="Signal vs x", x_label="x", y_label="Amplitude"):

    plt.figure(figsize=(20, 12))

    plt.plot(x_array, original_signal, label="Original Signal", color='black')

    if series_signals is not None:
        if np.ndim(series_signals) == 1:
            series_signals = [series_signals]

        if len(series_signals) > 0:
            n = len(series_signals)
            cmap = cm.plasma

            if n == 1:
                plt.plot(x_array, series_signals[0], label="Fourier Signal")

            else:
                norm = colors.Normalize(vmin=0, vmax=n-1)  

                for i, y in enumerate(series_signals):
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


def plot_epicycles(series_terms, dx=25, animate=False):

    if not np.iscomplexobj(series_terms):
        raise ValueError("plot_epicycles requires complex series_terms (periodic basis)")
    
    fig, ax = plt.subplots(figsize=(10, 10))

    all_cumulative = np.cumsum(series_terms, axis=1)
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

        U = series_terms[dx].real
        V = series_terms[dx].imag

        cumulative = np.cumsum(series_terms[dx])

        x = np.concatenate(([0], cumulative.real[:-1]))
        y = np.concatenate(([0], cumulative.imag[:-1]))

        radius = np.abs(series_terms[dx])

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

        fig.suptitle(f"Epicycles @Time Step={dx}")

        plt.show()
        
    else:

        U = series_terms[0].real
        V = series_terms[0].imag

        cumulative = np.cumsum(series_terms[0])

        x = np.concatenate(([0], cumulative.real[:-1]))
        y = np.concatenate(([0], cumulative.imag[:-1]))

        centers = list(zip(x, y))

        radius = np.abs(series_terms[0])

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

            U = series_terms[frame].real
            V = series_terms[frame].imag

            cumulative = np.cumsum(series_terms[frame])

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
            
            ax.set_title(f"Epicycles @Time Step={frame}")

            return (qvr, *circles)
        
        ani = FuncAnimation(fig, update, frames=len(series_terms), interval=100, blit=False)

        os.makedirs("animations", exist_ok=True)

        variable_name = "epycicles"

        ani.save(f"animations/{variable_name}.mp4", writer="ffmpeg")

        plt.show()


def plot_heat_equation_solution(x_array, time_array, heat_equation_solution, basis="periodic", animate=False):

    if animate is False:

        fig = plt.figure(figsize=(11, 7), dpi=100)
        ax = fig.add_subplot(projection='3d') 
        x = x_array
        y = time_array
        X, Y = np.meshgrid(x, y)

        ax.plot_surface(
        X,
        Y,
        heat_equation_solution,
        cmap=cm.viridis
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("z")

        plt.show()
    
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(x_array, heat_equation_solution[0], lw=2)

        ax.set_xlabel("x")
        ax.set_ylabel("u", rotation=0)
        ax.set_title(f"Heat Equation Solution {basis}")

        def update(frame):

            line.set_ydata(heat_equation_solution[frame])

            ax.set_title(f"Time: {time_array[frame]:.2f}")

            return line,

        ani = FuncAnimation(fig, update, frames=len(time_array), interval=100, blit=False)

        os.makedirs('animations', exist_ok=True)
        variable_name = f"heat_equation_solution_{basis}"
        ani.save(f'animations/{variable_name}.mp4', writer="ffmpeg")

        plt.show()



if __name__ == "__main__":

    x_array = np.linspace(0, 1, 1001, endpoint=False)
    time_array = np.linspace(0, 1, 1001, endpoint=False)

    num_frequencies = 500
    nu = 0.05
    basis = "periodic"

    signal_values = generate_step_signal(x_array)
    mode_indices = generate_mode_indices(num_frequencies, basis=basis)
    mode_coefficients = compute_coefficients(signal_values, x_array, mode_indices, basis=basis)
    series_terms = compute_series_terms(mode_indices, mode_coefficients, x_array, basis=basis)
    series_array = compute_series(series_terms)
    heat_equation_solution = compute_heat_equation_solution(series_terms, mode_indices, time_array, nu, basis=basis)

    plot_signals(x_array, signal_values)
    plot_epicycles(series_terms, dx=35)
    plot_epicycles(series_terms, animate=True)
    plot_signals(x_array, signal_values, series_array)
    plot_signals(x_array, signal_values, series_array[-1])
    plot_heat_equation_solution(x_array, time_array, heat_equation_solution, basis=basis, animate=False)
    plot_heat_equation_solution(x_array, time_array, heat_equation_solution, basis=basis, animate=True)