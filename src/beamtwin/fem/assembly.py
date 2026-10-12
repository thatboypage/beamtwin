"""Join many beam elements into one global stiffness matrix (Step 1b)."""

import numpy as np

from beamtwin.fem.element import element_stiffness


def assemble_stiffness(E: float, I: float, length: float, n_elements: int) -> np.ndarray:
    """Return the global stiffness matrix K of a beam cut into equal elements.

    The beam has n_elements pieces, so it has n_elements + 1 nodes.
    Each node has 2 movements (deflection, rotation), so K is
    2 * (n_elements + 1) rows by the same number of columns.

    Movements are numbered along the beam:
        node 0 -> 0 (deflection), 1 (rotation)
        node 1 -> 2, 3
        node 2 -> 4, 5   ... and so on.

    Where two elements share a node, their stiffnesses add up.
    """
    n_nodes = n_elements + 1
    n_dof = 2 * n_nodes                     # dof = "degree of freedom" = one movement
    L_e = length / n_elements               # length of each piece

    K = np.zeros((n_dof, n_dof))
    k_e = element_stiffness(E, I, L_e)      # every piece is identical here

    for e in range(n_elements):
        # Element e joins node e to node e + 1.
        # Its 4 movements sit at these positions in the big matrix:
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]

        # Add the small 4x4 table into the right place in the big table.
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k_e[a, b]

    return K
