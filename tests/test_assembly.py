"""Checks on the assembled global stiffness matrix (Step 1b)."""

import numpy as np

from beamtwin.fem.assembly import assemble_stiffness
from beamtwin.fem.element import element_stiffness


def test_size(steel_strip):
    K = assemble_stiffness(**steel_strip, n_elements=4)
    assert K.shape == (10, 10)


def test_global_is_symmetric(steel_strip):
    K = assemble_stiffness(**steel_strip, n_elements=4)
    assert np.allclose(K, K.T)


def test_shared_node_adds_two_elements(steel_strip):
    n = 4
    K = assemble_stiffness(**steel_strip, n_elements=n)
    k_e = element_stiffness(steel_strip["E"], steel_strip["I"], steel_strip["length"] / n)
    assert np.isclose(K[2, 2], k_e[2, 2] + k_e[0, 0])


def test_rigid_body_motions_need_no_force(steel_strip):
    n = 4
    K = assemble_stiffness(**steel_strip, n_elements=n)
    x = np.linspace(0, steel_strip["length"], n + 1)
    lift = np.tile([1.0, 0.0], n + 1)
    tilt = np.column_stack([x, np.ones_like(x)]).ravel()
    tol = 1e-9 * np.abs(K).max()
    assert np.allclose(K @ lift, 0.0, atol=tol)
    assert np.allclose(K @ tilt, 0.0, atol=tol)
