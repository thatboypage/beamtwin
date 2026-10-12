"""Static solution of a cantilever beam with a tip load (Step 1c)."""

import numpy as np

from beamtwin.fem.assembly import assemble_stiffness


def solve_cantilever(E: float, I: float, length: float, n_elements: int, tip_load: float) -> np.ndarray:
    """Solve a cantilever beam with a point load at its free end.

    The left end (node 0) is clamped: it cannot move and cannot tilt.
    The load pushes on the right end. Positive means upward.

    Returns u, the movement of every node: [w0, theta0, w1, theta1, ...].
    Deflections are u[0::2] and rotations are u[1::2].
    """
    K = assemble_stiffness(E, I, length, n_elements)
    n_dof = K.shape[0]

    # The load vector F: one entry per movement, all zero except the tip.
    F = np.zeros(n_dof)
    F[-2] = tip_load                # deflection of the last node

    # The clamp: movements 0 and 1 (node 0's deflection and rotation) are zero.
    fixed = [0, 1]
    free = [d for d in range(n_dof) if d not in fixed]

    # Keep only the rows and columns of the movements that are free to happen.
    K_free = K[np.ix_(free, free)]
    F_free = F[free]

    # Solve K u = F for the free movements.
    u = np.zeros(n_dof)
    u[free] = np.linalg.solve(K_free, F_free)
    return u
