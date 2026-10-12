"""Stiffness matrix of a single Euler-Bernoulli beam element (Step 1a)."""

import numpy as np


def element_stiffness(E: float, I: float, L: float) -> np.ndarray:
    """Return the 4x4 stiffness matrix of one beam element.

    The element has two nodes. Each node can do two things:
    move up/down (deflection w) and tilt (rotation theta).
    So the element's movements are ordered [w1, theta1, w2, theta2].

    Each column is one "experiment": move one thing by 1 unit, hold the
    other three still, and record the four forces/moments needed.
    See docs/theory.md for the derivation of the 12, 6L, 4L^2 and 2L^2 terms.

    Parameters
    ----------
    E : Young's modulus of the material (Pa)
    I : second moment of area of the cross-section (m^4)
    L : length of the element (m)
    """
    return (E * I / L**3) * np.array([
        [ 12,     6 * L,   -12,     6 * L   ],
        [ 6 * L,  4 * L**2, -6 * L, 2 * L**2],
        [-12,    -6 * L,    12,    -6 * L   ],
        [ 6 * L,  2 * L**2, -6 * L, 4 * L**2],
    ])
