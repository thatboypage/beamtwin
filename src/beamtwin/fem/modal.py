"""Natural frequencies and mode shapes of a cantilever beam (Step 2)."""

import numpy as np
from scipy.linalg import eigh

from beamtwin.fem.assembly import assemble_stiffness


def element_mass(rho: float, A: float, L: float) -> np.ndarray:
    """Return the 4x4 consistent mass matrix of one beam element.

    rho : density of the material (kg/m^3)
    A   : cross-section area (m^2)
    L   : length of the element (m)

    Movements are ordered [w1, theta1, w2, theta2], as in element_stiffness.
    "Consistent" means the mass is spread using the same cubic shapes as the
    stiffness matrix, instead of being lumped at the two nodes.
    See docs/theory.md for where the 156, 22L, 54 and 13L terms come from.
    """
    return (rho * A * L / 420) * np.array([
        [ 156,     22 * L,    54,    -13 * L   ],
        [ 22 * L,  4 * L**2,  13 * L, -3 * L**2],
        [ 54,      13 * L,    156,   -22 * L   ],
        [-13 * L, -3 * L**2, -22 * L,  4 * L**2],
    ])


def assemble_mass(rho: float, A: float, length: float, n_elements: int) -> np.ndarray:
    """Return the global mass matrix M, assembled exactly like K."""
    n_dof = 2 * (n_elements + 1)
    L_e = length / n_elements
    M = np.zeros((n_dof, n_dof))
    m_e = element_mass(rho, A, L_e)

    for e in range(n_elements):
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        M[np.ix_(dofs, dofs)] += m_e
    return M


def cantilever_modes(E: float, I: float, rho: float, A: float, length: float,
                     n_elements: int, n_modes: int = 4,
                     stiffness_factors: np.ndarray | None = None):
    """Solve K phi = omega^2 M phi for a cantilever clamped at node 0.

    stiffness_factors : optional per-element stiffness multipliers (1 = healthy),
        passed straight to assemble_stiffness. Used to model damage.

    Returns
    -------
    freqs_hz : natural frequencies in Hz, lowest first (length n_modes)
    shapes   : mode shapes, one column per mode, for ALL degrees of freedom
               (the clamped ones are zero). Each shape is scaled so its tip
               deflection is +1.
    """
    K = assemble_stiffness(E, I, length, n_elements, stiffness_factors)
    M = assemble_mass(rho, A, length, n_elements)
    n_dof = K.shape[0]

    # Clamp the left end, exactly as in the static solver.
    free = list(range(2, n_dof))
    K_f = K[np.ix_(free, free)]
    M_f = M[np.ix_(free, free)]

    # Generalised eigenvalue problem: eigenvalues are omega^2 (rad/s squared).
    omega_sq, phi = eigh(K_f, M_f, subset_by_index=[0, n_modes - 1])
    freqs_hz = np.sqrt(omega_sq) / (2 * np.pi)

    shapes = np.zeros((n_dof, n_modes))
    shapes[free, :] = phi
    shapes /= shapes[-2, :]          # tip deflection = +1 for every mode
    return freqs_hz, shapes


# beta_n * L for a cantilever: roots of 1 + cos(x) cosh(x) = 0
# (given to full precision, so the convergence study is not limited by rounding)
CANTILEVER_BETA_L = np.array([
    1.8751040687119611,
    4.6940911329741745,
    7.8547574382376126,
    10.995540734875467,
])


def cantilever_exact_frequencies(E: float, I: float, rho: float, A: float, length: float) -> np.ndarray:
    """Textbook natural frequencies (Hz) of a uniform cantilever, first four modes."""
    omega = CANTILEVER_BETA_L**2 * np.sqrt(E * I / (rho * A * length**4))
    return omega / (2 * np.pi)
