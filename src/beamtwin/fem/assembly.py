"""Join many beam elements into one global stiffness matrix (Step 1b)."""

import numpy as np

from beamtwin.fem.element import element_stiffness


def assemble_stiffness(E: float, I: float, length: float, n_elements: int,
                       stiffness_factors: np.ndarray | None = None) -> np.ndarray:
    """Return the global stiffness matrix K of a beam cut into equal elements.

    The beam has n_elements pieces, so it has n_elements + 1 nodes.
    Each node has 2 movements (deflection, rotation), so K is
    2 * (n_elements + 1) rows by the same number of columns.

    Movements are numbered along the beam:
        node 0 -> 0 (deflection), 1 (rotation)
        node 1 -> 2, 3
        node 2 -> 4, 5   ... and so on.

    Where two elements share a node, their stiffnesses add up.

    stiffness_factors : optional, one number per element. Element e gets
        bending stiffness EI * stiffness_factors[e]. A value of 1 is healthy;
        0.7 means that element has lost 30% of its stiffness (e.g. a crack).
        Leave as None for a uniform, healthy beam.
    """
    n_nodes = n_elements + 1
    n_dof = 2 * n_nodes                     # dof = "degree of freedom" = one movement
    L_e = length / n_elements               # length of each piece

    if stiffness_factors is None:
        stiffness_factors = np.ones(n_elements)
    stiffness_factors = np.asarray(stiffness_factors, dtype=float)
    if stiffness_factors.shape != (n_elements,):
        raise ValueError(f"stiffness_factors needs {n_elements} values, got {stiffness_factors.shape}")

    K = np.zeros((n_dof, n_dof))
    k_e = element_stiffness(E, I, L_e)      # stiffness of one healthy piece

    for e in range(n_elements):
        # Element e joins node e to node e + 1.
        # Its 4 movements sit at these positions in the big matrix:
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]

        # Add this piece's table (scaled by its health) into the big table.
        K[np.ix_(dofs, dofs)] += stiffness_factors[e] * k_e

    return K
