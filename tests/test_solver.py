"""Checks on the cantilever solution against textbook formulas (Step 1c)."""

import numpy as np
import pytest

from beamtwin.fem.solver import solve_cantilever

P = -10.0  # 10 N downwards


@pytest.mark.parametrize("n_elements", [1, 2, 4, 8, 16])
def test_tip_deflection_matches_textbook(steel_strip, n_elements):
    E, I, L = steel_strip["E"], steel_strip["I"], steel_strip["length"]
    u = solve_cantilever(E, I, L, n_elements, P)
    assert np.isclose(u[-2], P * L**3 / (3 * E * I), rtol=1e-10)


def test_tip_rotation_matches_textbook(steel_strip):
    E, I, L = steel_strip["E"], steel_strip["I"], steel_strip["length"]
    u = solve_cantilever(E, I, L, 4, P)
    assert np.isclose(u[-1], P * L**2 / (2 * E * I), rtol=1e-10)


def test_deflection_along_beam_matches_textbook(steel_strip):
    """Exact curve for a tip load: w(x) = P x^2 (3L - x) / (6EI)."""
    E, I, L = steel_strip["E"], steel_strip["I"], steel_strip["length"]
    n = 8
    u = solve_cantilever(E, I, L, n, P)
    x = np.linspace(0, L, n + 1)
    exact = P * x**2 * (3 * L - x) / (6 * E * I)
    assert np.allclose(u[0::2], exact, rtol=1e-10, atol=1e-15)


def test_clamped_end_does_not_move(steel_strip):
    u = solve_cantilever(**steel_strip, n_elements=4, tip_load=P)
    assert u[0] == 0.0 and u[1] == 0.0


def test_response_is_linear(steel_strip):
    u1 = solve_cantilever(**steel_strip, n_elements=4, tip_load=P)
    u2 = solve_cantilever(**steel_strip, n_elements=4, tip_load=2 * P)
    assert np.allclose(u2, 2 * u1)
