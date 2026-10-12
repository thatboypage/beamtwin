"""Checks on a single beam element (Step 1a)."""

import numpy as np

from beamtwin.fem.element import element_stiffness

E, I, L = 200e9, 0.05 * 0.01**3 / 12, 0.1


def test_element_is_symmetric():
    k = element_stiffness(E, I, L)
    assert np.allclose(k, k.T)


def test_pure_lift_needs_no_force():
    k = element_stiffness(E, I, L)
    lift = np.array([1.0, 0.0, 1.0, 0.0])
    assert np.allclose(k @ lift, 0.0, atol=1e-9 * np.abs(k).max())


def test_pure_tilt_needs_no_force():
    k = element_stiffness(E, I, L)
    tilt = np.array([0.0, 1.0, L, 1.0])
    assert np.allclose(k @ tilt, 0.0, atol=1e-9 * np.abs(k).max())


def test_doubling_length_makes_it_eight_times_floppier():
    k1 = element_stiffness(E, I, L)
    k2 = element_stiffness(E, I, 2 * L)
    assert np.isclose(k1[0, 0] / k2[0, 0], 8.0)
