import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
import os


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
    """Plot the original signal and optional Fourier reconstructions."""

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
    """Visualize the periodic Fourier series as epicycles."""

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
    """Plot the analytical heat equation solution."""

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
