import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import os


def generate_signal(x_array, x_start=0, x_end=0.5, amplitude_min=-1, amplitude_max=1):

    signal_values = np.zeros_like(x_array)

    signal_values[(x_array < x_start) | (x_array > x_end)] = amplitude_min
    signal_values[(x_array >= x_start) & (x_array <= x_end)] = amplitude_max

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
        basis_matrix  = np.exp(-2j * np.pi * mode_column * x_row)
        return (signal_values * basis_matrix).sum(axis=1) * dx
     
    elif basis == "cosine":
        basis_matrix = np.cos(np.pi * mode_column * x_row)
        coefficients = (signal_values * basis_matrix).sum(axis=1) * dx
        coefficients[1:] = 2 * coefficients[1:]
        return coefficients

    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")


def compute_series_terms(mode_indices, coefficients, x_array, basis="periodic"):

    x_column = x_array[:, None]

    if basis == "periodic":
        return coefficients*np.exp(mode_indices * 2j * np.pi * x_column)
    
    elif basis == "cosine":
        return coefficients*np.cos(np.pi * mode_indices * x_column)

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


def plot_signals(x_array, original_signal, series_signals=None, title="Signal vs Position", x_label="Position", y_label="Amplitude", save_fig=False):

    plt.figure(figsize=(20, 12))

    plt.plot(x_array, original_signal, label="Original Signal", color='black')

    if save_fig == True:

        os.makedirs('figures', exist_ok=True)
        variable_name = f"original_signals"
        plt.savefig(f'figures/{variable_name}.png')

    if series_signals is not None:

        if np.ndim(series_signals) == 1:
            series_signals = [series_signals]

        if len(series_signals) > 0:
            n = len(series_signals)
            cmap = cm.plasma

            if n == 1:
                plt.plot(x_array, series_signals[0], label="Series Signal")

                if save_fig == True:
                    os.makedirs('figures', exist_ok=True)
                    variable_name = f"series_signal"
                    plt.savefig(f'figures/{variable_name}.png')

            else:
                norm = colors.Normalize(vmin=0, vmax=n-1)  

                for i, y in enumerate(series_signals):
                    plt.plot(x_array, y, color=cmap(norm(i)))
                
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                ax = plt.gca()  # Get current axes
                plt.colorbar(sm, ax=ax, label="Frequency")

                if save_fig == True:
                    os.makedirs('figures', exist_ok=True)
                    variable_name = f"multi_modes_series_signals"
                    plt.savefig(f'figures/{variable_name}.png')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.show()


def plot_epicycles(series_terms, dx=25, animate=False, save_fig=False):

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

        fig.suptitle(f"Epicycles @Time={dx}")

        if save_fig == True:
            os.makedirs('figures', exist_ok=True)
            variable_name = f"epicycles"
            fig.savefig(f'figures/{variable_name}.png')

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
            
            ax.set_title(f"Epicycles @Time={frame}")

            return (qvr, *circles)
        
        ani = FuncAnimation(fig, update, frames=len(series_terms), interval=100, blit=False)

        if save_fig == True:
            os.makedirs("animations", exist_ok=True)
            variable_name = "epicycles"
            ani.save(f"animations/{variable_name}.mp4", writer="ffmpeg")

        plt.show()


def plot_heat_equation_solution(x_array, time_array, heat_equation_solution, basis="periodic", animate=False, save_fig=False):

    if animate is False:

        fig = plt.figure(figsize=(11, 7), dpi=100)
        ax = fig.add_subplot(projection='3d') 
        x = x_array
        y = time_array
        X, Y = np.meshgrid(x, y)

        surf = ax.plot_surface(
        X,
        Y,
        heat_equation_solution,
        cmap=cm.coolwarm
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Time")
        ax.set_zlabel("Temperature")

        ax.set_title(f"Temperature profile ({basis} initial condition)")

        fig.colorbar(surf, ax=ax)

        if save_fig == True:
            os.makedirs('figures', exist_ok=True)
            variable_name = f"temperature_profile_{basis}"
            fig.savefig(f'figures/{variable_name}.png')

        plt.show()
    
    else:

        fig, ax = plt.subplots(figsize=(10, 6))

        y0 = heat_equation_solution[0]

        points = np.array([x_array, y0]).T
        segments = np.stack([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='coolwarm')
        lc.set_array(y0)
        lc.set_linewidth(2)

        vmax = np.max(np.abs(heat_equation_solution))
        lc.set_clim(-vmax, vmax)

        line = ax.add_collection(lc)

        fig.colorbar(lc, ax=ax)

        ax.set_xlim(x_array.min(), x_array.max())
        ax.set_ylim(
            np.min(heat_equation_solution),
            np.max(heat_equation_solution)
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Temperature")

        def update(frame):

            y = heat_equation_solution[frame]

            points = np.array([x_array, y]).T
            segments = np.stack([points[:-1], points[1:]], axis=1)

            line.set_segments(segments)
            line.set_array(y)

            ax.set_title(f"Temperature profile ({basis} initial condition, t = {time_array[frame]:.2f})")

            return line,

        ani = FuncAnimation(fig, update, frames=len(time_array), interval=100, blit=False)

        if save_fig == True:
            os.makedirs('animations', exist_ok=True)
            variable_name = f"temperature_profile_{basis}"
            ani.save(f'animations/{variable_name}.mp4', writer="ffmpeg")

        plt.show()



if __name__ == "__main__":

    x_array = np.linspace(0, 1, 1001, endpoint=False)
    time_array = np.linspace(0, 10, 1001, endpoint=False)

    num_frequencies = 500
    nu = 0.03
    basis = "periodic"  # "periodic" or "cosine"

    signal_values = generate_signal(x_array, x_start=0, x_end=0.5, amplitude_min=-1, amplitude_max=1)
    mode_indices = generate_mode_indices(num_frequencies, basis=basis)
    mode_coefficients = compute_coefficients(signal_values, x_array, mode_indices, basis=basis)
    series_terms = compute_series_terms(mode_indices, mode_coefficients, x_array, basis=basis)
    series_array = compute_series(series_terms)
    heat_equation_solution = compute_heat_equation_solution(series_terms, mode_indices, time_array, nu, basis=basis)

    plot_signals(x_array, signal_values, save_fig=False)
    plot_epicycles(series_terms, dx=35, save_fig=False)
    plot_epicycles(series_terms, animate=True, save_fig=False)
    plot_signals(x_array, signal_values, series_array, save_fig=False)
    plot_signals(x_array, signal_values, series_array[-1], save_fig=False)
    plot_heat_equation_solution(x_array, time_array, heat_equation_solution, basis=basis, animate=False, save_fig=False)
    plot_heat_equation_solution(x_array, time_array, heat_equation_solution, basis=basis, animate=True, save_fig=False)
