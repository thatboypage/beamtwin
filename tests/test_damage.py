"""Checks on damage modelling and its effect on the FEM (Step 3a)."""

import numpy as np
import pytest

from beamtwin.damage.simulate import damage_profile
from beamtwin.fem.assembly import assemble_stiffness
from beamtwin.fem.modal import cantilever_exact_frequencies, cantilever_modes

N = 50
RHO, AREA = 7850.0, 0.05 * 0.01


@pytest.fixture
def beam(steel_strip):
    return {**steel_strip, "rho": RHO, "A": AREA}


def test_no_factors_equals_all_ones(steel_strip):
    K_default = assemble_stiffness(**steel_strip, n_elements=N)
    K_ones = assemble_stiffness(**steel_strip, n_elements=N, stiffness_factors=np.ones(N))
    assert np.allclose(K_default, K_ones)


def test_wrong_number_of_factors_is_rejected(steel_strip):
    with pytest.raises(ValueError):
        assemble_stiffness(**steel_strip, n_elements=N, stiffness_factors=np.ones(N - 1))


def test_zero_severity_is_healthy():
    assert np.allclose(damage_profile(N, location=0.3, severity=0.0), 1.0)


def test_profile_dips_at_the_damage_location():
    # 0.31 is the centre of element 15, so the deepest point lands exactly on an element
    f = damage_profile(N, location=0.31, severity=0.4)
    x_mid = (np.arange(N) + 0.5) / N
    assert np.isclose(x_mid[np.argmin(f)], 0.31)
    assert np.all((f > 0) & (f <= 1))
    assert np.isclose(f.min(), 0.6)


def test_damage_lowers_every_frequency(beam):
    healthy, _ = cantilever_modes(**beam, n_elements=N)
    damaged, _ = cantilever_modes(**beam, n_elements=N,
                                  stiffness_factors=damage_profile(N, 0.4, 0.3))
    assert np.all(damaged < healthy)


def test_healthy_mesh_still_matches_textbook(beam):
    freqs, _ = cantilever_modes(**beam, n_elements=N, stiffness_factors=np.ones(N))
    assert np.allclose(freqs, cantilever_exact_frequencies(**beam), rtol=1e-4)


def test_damage_near_clamp_hurts_mode_1_most(beam):
    """Mode 1 bends most at the clamp, so damage there lowers f1 more than damage near the tip."""
    near_clamp, _ = cantilever_modes(**beam, n_elements=N, n_modes=1,
                                     stiffness_factors=damage_profile(N, 0.1, 0.3))
    near_tip, _ = cantilever_modes(**beam, n_elements=N, n_modes=1,
                                   stiffness_factors=damage_profile(N, 0.9, 0.3))
    assert near_clamp[0] < near_tip[0]
