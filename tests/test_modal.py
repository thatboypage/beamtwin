"""Checks on the mass matrix and natural frequencies (Step 2)."""

import numpy as np
import pytest

from beamtwin.fem.modal import (
    assemble_mass,
    cantilever_exact_frequencies,
    cantilever_modes,
    element_mass,
)

RHO = 7850.0                 # steel, kg/m^3
B, H = 0.05, 0.01
AREA = B * H


@pytest.fixture
def beam(steel_strip):
    return {**steel_strip, "rho": RHO, "A": AREA}


def test_element_mass_is_symmetric():
    m = element_mass(RHO, AREA, 0.1)
    assert np.allclose(m, m.T)


def test_total_mass_is_rho_A_L(beam):
    """Lifting the whole beam by 1 must 'see' exactly its real mass."""
    n = 4
    M = assemble_mass(beam["rho"], beam["A"], beam["length"], n)
    lift = np.tile([1.0, 0.0], n + 1)
    assert np.isclose(lift @ M @ lift, beam["rho"] * beam["A"] * beam["length"])


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_frequencies_match_textbook(beam, mode):
    exact = cantilever_exact_frequencies(**beam)
    freqs, _ = cantilever_modes(**beam, n_elements=32, n_modes=4)
    assert np.isclose(freqs[mode], exact[mode], rtol=1e-4)


def test_frequencies_converge_from_above(beam):
    """With a consistent mass matrix, FEM frequencies approach the exact ones from above."""
    exact = cantilever_exact_frequencies(**beam)
    errors = []
    for n in [4, 8, 16, 32]:
        freqs, _ = cantilever_modes(**beam, n_elements=n, n_modes=4)
        assert np.all(freqs >= exact * (1 - 1e-9))
        errors.append(freqs - exact)
    errors = np.array(errors)
    assert np.all(np.diff(errors, axis=0) <= 1e-9)


def test_mode_n_has_n_minus_one_zero_crossings(beam):
    """Mode 1 never crosses zero, mode 2 crosses once, mode 3 twice..."""
    _, shapes = cantilever_modes(**beam, n_elements=32, n_modes=4)
    deflections = shapes[2::2, :]                 # skip the clamped node
    for mode in range(4):
        signs = np.sign(deflections[:, mode])
        crossings = np.count_nonzero(np.diff(signs) != 0)
        assert crossings == mode


def test_stiffer_beam_vibrates_faster(beam):
    f1, _ = cantilever_modes(**beam, n_elements=8, n_modes=1)
    stiffer = {**beam, "E": 4 * beam["E"]}
    f2, _ = cantilever_modes(**stiffer, n_elements=8, n_modes=1)
    assert np.isclose(f2[0] / f1[0], 2.0)        # frequency goes with sqrt(E)
