import numpy as np


def solve_heat_equation_explicit(
    initial_condition: np.ndarray,
    dx: float,
    dt: float,
    num_time_steps: int,
    nu: float,
) -> np.ndarray:
    """Solve 1D heat equation using explicit finite differences.

    initial_condition: shape (num_x,)
    dx: spatial step
    dt: time step
    num_time_steps: number of time steps
    nu: diffusion coefficient
    """

    num_x = len(initial_condition)

    solution = np.zeros((num_time_steps, num_x))
    solution[0] = initial_condition.copy()

    alpha = nu * dt / dx**2

    if alpha > 0.5:
        raise ValueError(
            f"Explicit scheme unstable: alpha = {alpha:.6f} > 0.5. "
            "Decrease dt or increase dx."
        )

    for t in range(1, num_time_steps):
        u = solution[t - 1]

        u_new = u.copy()

        # internal points
        u_new[1:-1] = (
            u[1:-1]
            + alpha * (u[2:] - 2 * u[1:-1] + u[:-2])
        )

        # periodic boundary (to match your Fourier periodic case)
        u_new[0] = u[0] + alpha * (u[1] - 2 * u[0] + u[-1])
        u_new[-1] = u[-1] + alpha * (u[0] - 2 * u[-1] + u[-2])

        solution[t] = u_new

    return solution