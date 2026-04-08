import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import os


def generate_signal(
    x_array: np.ndarray, 
    x_start: float=0, 
    x_end: float=0.5, 
    amplitude_min: float=-1, 
    amplitude_max: float=1
) -> np.ndarray:
    
    """Generate a piecewise constant signal defined on the interval [0, 1).
    The signal takes the value `amplitude_max` on the interval [x_start, x_end] and `amplitude_min` elsewhere.
    Parameters:
    - x_array: 1D numpy array of x values where the signal is evaluated.
    - x_start: Start of the interval where the signal takes the maximum value.
    - x_end: End of the interval where the signal takes the maximum value.
    - amplitude_min: Minimum amplitude of the signal.
    - amplitude_max: Maximum amplitude of the signal.
    Returns:
    - A 1D numpy array of signal values corresponding to the x_array.
    """

    signal_values = np.zeros_like(x_array)

    signal_values[(x_array < x_start) | (x_array > x_end)] = amplitude_min
    signal_values[(x_array >= x_start) & (x_array <= x_end)] = amplitude_max

    return signal_values


def generate_mode_indices(num_modes: int, basis: str="periodic") -> np.ndarray:

    """Generate mode indices for Fourier series expansion.
    For the periodic basis, the mode indices are arranged as [0, 1, -1, 2, -2, ..., num_modes, -num_modes].
    For the cosine basis, the mode indices are simply [0, 1, 2, ..., num_modes].
    Parameters:
    - num_modes: Number of modes to generate.
    - basis: Basis for the Fourier series expansion ('periodic' or 'cosine').
    Returns:
    - A 1D numpy array of mode indices corresponding to the specified basis.
    """

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

    """Compute Fourier coefficients for the given signal and mode indices.
    Parameters:     
    - signal_values: 1D numpy array of signal values evaluated at x_array.
    - x_array: 1D numpy array of x values where the signal is evaluated.
    - mode_indices: 1D numpy array of mode indices.
    - basis: Basis for the Fourier series expansion ('periodic' or 'cosine').
    Returns:    
    - A 1D numpy array of Fourier coefficients corresponding to the mode indices.
    """


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

    """Compute the terms of the Fourier series for each mode and x value.
    Parameters:    
    - mode_indices: 1D numpy array of mode indices.  
    - coefficients: 1D numpy array of Fourier coefficients corresponding to the mode indices.  
    - x_array: 1D numpy array of x values where the series is evaluated.  
    - basis: Basis for the Fourier series expansion ('periodic' or 'cosine').  
    Returns:    
    - A 2D numpy array of shape (num_x, num_modes) containing the Fourier series terms for each mode and x value.
    Each element (i, j) of the output array corresponds to the contribution of the j-th mode to the Fourier series at the i-th x value.
    """

    x_column = x_array[:, None]

    if basis == "periodic":
        return coefficients * np.exp(mode_indices * 2j * np.pi * x_column)
    
    elif basis == "cosine":
        return coefficients * np.cos(np.pi * mode_indices * x_column)

    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")


def compute_series(series_terms: np.ndarray) -> np.ndarray:
    
    """Compute the partial sums of the Fourier series by cumulatively summing the series terms along the mode axis.
    Parameters:
    - series_terms: 2D numpy array of shape (num_x, num_modes) containing the Fourier series terms for each mode and x value.
    Returns:
    - A 2D numpy array of shape (num_modes, num_x) containing the cumulative partial sums of the Fourier series.
    """

    return np.cumsum(series_terms, axis=1).T


def compute_heat_equation_solution(
    series_terms: np.ndarray, 
    mode_indices: np.ndarray, 
    time_array: np.ndarray, 
    nu: float, 
    basis: str="periodic"
) -> np.ndarray:

    """Compute the solution of the heat equation at different time steps using the Fourier series representation of the initial condition.
    Parameters:
    - series_terms: 2D numpy array of shape (num_x, num_modes) containing the Fourier series terms for each mode and x value.
    - mode_indices: 1D numpy array of mode indices corresponding to the Fourier series expansion.
    - time_array: 1D numpy array of time values where the solution is evaluated.
    - nu: Diffusion coefficient in the heat equation.
    - basis: Basis for the Fourier series expansion ('periodic' or 'cosine').
    Returns:
    - A 2D numpy array of shape (num_time, num_x) containing the solution of the heat equation at each time step and x value.
    """
    
    t = time_array[:, None, None]
    n = mode_indices[None, None, :]

    if basis == "periodic":
        decay = np.exp(-nu * ((2 * np.pi * n)**2 * t))
    
    elif basis == "cosine":
        decay = np.exp(-nu * ((np.pi * n)**2 * t))
    
    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")

    return np.sum(series_terms[None, :, :] * decay, axis=2).real


def plot_signals(
    x_array: np.ndarray, 
    original_signal: np.ndarray, 
    series_signals: np.ndarray=None, 
    title: str="Signal vs Position", 
    x_label: str="Position", 
    y_label: str="Amplitude", 
    basis: str="periodic", 
    save_fig: bool=False
) -> None:

    """Plot the original signal and the Fourier series approximations.
    Parameters:
    - x_array: 1D numpy array of x values.
    - original_signal: 1D numpy array of the original signal values.
    - series_signals: 1D or 2D numpy array of the Fourier series approximations.
    - title: Title for the plot.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - basis: Basis for the Fourier series expansion ('periodic' or 'cosine').
    - save_fig: Whether to save the figure.
     Returns:
    - None (the function displays the plot and optionally saves it to a file).
    """

    plt.figure(figsize=(20, 12))

    plt.plot(x_array, original_signal, label="Original Signal", color='black')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title} ({basis})")
    plt.legend()
    plt.tight_layout()

    if save_fig:

        os.makedirs('figures', exist_ok=True)
        variable_name = f"original_signal_{basis}"
        plt.savefig(f'figures/{variable_name}.png')

    if series_signals is not None:

        if np.ndim(series_signals) == 1:
            series_signals = [series_signals]

        if len(series_signals) > 0:
            n = len(series_signals)
            cmap = cm.plasma

            if n == 1:
                plt.plot(x_array, series_signals[0], label="Series Signal")

                if save_fig:
                    os.makedirs('figures', exist_ok=True)
                    variable_name = f"series_signals_{basis}"
                    plt.savefig(f'figures/{variable_name}.png')

            else:
                norm = colors.Normalize(vmin=0, vmax=n-1)  

                for i, y in enumerate(series_signals):
                    plt.plot(x_array, y, color=cmap(norm(i)))
                
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                ax = plt.gca()  # Get current axes
                plt.colorbar(sm, ax=ax, label="Frequency")

                if save_fig:
                    os.makedirs('figures', exist_ok=True)
                    variable_name = f"multi_modes_series_signals_{basis}"
                    plt.savefig(f'figures/{variable_name}.png')

    plt.show()


def plot_epicycles(
    series_terms: np.ndarray, 
    sample_index: int=25, 
    animate: bool=False, 
    save_fig: bool=False
) -> None:

    """Visualize the Fourier series as epicycles.
    Parameters:
    - series_terms: 2D numpy array of complex Fourier coefficients.
    - sample_index: Index of the time step to visualize.
    - animate: Whether to create an animated visualization.
    - save_fig: Whether to save the figure.
    Returns:
    - None (the function displays the plot and optionally saves it to a file).
    """

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

        U = series_terms[sample_index].real
        V = series_terms[sample_index].imag

        cumulative = np.cumsum(series_terms[sample_index])

        x = np.concatenate(([0], cumulative.real[:-1]))
        y = np.concatenate(([0], cumulative.imag[:-1]))

        radius = np.abs(series_terms[sample_index])

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

        fig.suptitle(f"Epicycles @Time={sample_index}")

        if save_fig:
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

        if save_fig:
            os.makedirs("animations", exist_ok=True)
            variable_name = "epicycles"
            ani.save(f"animations/{variable_name}.mp4", writer="ffmpeg")

        plt.show()


def plot_heat_equation_solution(
    x_array: np.ndarray, 
    time_array: np.ndarray, 
    heat_equation_solution: np.ndarray, 
    basis: str="periodic", 
    animate: bool=False, 
    save_fig: bool=False
) -> None:

    """Plot the solution of the heat equation as a function of position and time.
    Parameters:
    - x_array: 1D numpy array of x values.
    - time_array: 1D numpy array of time values.
    - heat_equation_solution: 2D numpy array of the heat equation solution.
    - basis: Basis for the Fourier series expansion ('periodic' or 'cosine').
    - animate: Whether to create an animated visualization.
    - save_fig: Whether to save the figure.
    Returns:
    - None (the function displays the plot and optionally saves it to a file).
    """

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

        if save_fig:
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

        if save_fig:
            os.makedirs('animations', exist_ok=True)
            variable_name = f"temperature_profile_{basis}"
            ani.save(f'animations/{variable_name}.mp4', writer="ffmpeg")

        plt.show()



if __name__ == "__main__":

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

    signal_values = generate_signal(x_array, x_start=x_start, x_end=x_end, amplitude_min=amplitude_min, amplitude_max=amplitude_max)
    mode_indices = generate_mode_indices(num_modes, basis=basis)
    mode_coefficients = compute_coefficients(signal_values, x_array, mode_indices, basis=basis)
    series_terms = compute_series_terms(mode_indices, mode_coefficients, x_array, basis=basis)
    partial_sums = compute_series(series_terms)
    heat_equation_solution = compute_heat_equation_solution(series_terms, mode_indices, time_array, nu, basis=basis)

    plot_signals(x_array, signal_values, basis=basis, save_fig=save_fig)
    plot_epicycles(series_terms, sample_index=sample_index, save_fig=save_fig)
    plot_epicycles(series_terms, animate=True, save_fig=save_fig)
    plot_signals(x_array, signal_values, partial_sums, basis=basis, save_fig=save_fig)
    plot_signals(x_array, signal_values, partial_sums[partial_sums_index], basis=basis, save_fig=save_fig)
    plot_heat_equation_solution(x_array, time_array, heat_equation_solution, basis=basis, animate=False, save_fig=save_fig)
    plot_heat_equation_solution(x_array, time_array, heat_equation_solution, basis=basis, animate=True, save_fig=save_fig)
