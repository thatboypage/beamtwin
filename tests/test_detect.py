"""Checks on the damage detector (Step 4). Uses the exact FEM on a coarse grid, so no PyTorch needed."""

import numpy as np
import pytest

from beamtwin.damage.detect import detect, simulate_measurement
from beamtwin.damage.forward import Beam, FEMForward
from beamtwin.surrogate.data_generation import simulate_one

BEAM = Beam(length=1.0, width=0.05, thickness=0.01, E=200e9, rho=7850)
LOCS = np.linspace(0.05, 0.95, 19)       # coarse grid: every 5 %
SEVS = np.linspace(0.0, 0.5, 11)         # every 5 %


@pytest.fixture(scope="module")
def forward():
    return FEMForward()


def test_noiseless_measurement_equals_fem():
    m = simulate_measurement(BEAM, 0.3, 0.2, freq_noise=0.0, shape_noise=0.0)
    y = simulate_one(1.0, 0.05, 0.01, 200e9, 7850, 0.3, 0.2)
    assert np.allclose(m.frequencies, y[:4]) and np.allclose(m.shapes, y[4:])


def test_forward_of_healthy_beam_gives_ratio_one(forward):
    ratios, _ = forward(BEAM, [0.5], [0.0])
    assert np.allclose(ratios, 1.0, atol=1e-5)


def test_finds_damage_that_lies_on_the_grid(forward):
    m = simulate_measurement(BEAM, 0.35, 0.30, freq_noise=0.0, shape_noise=0.0)
    r = detect(m, BEAM, forward, locations=LOCS, severities=SEVS)
    assert np.isclose(r.location, 0.35) and np.isclose(r.severity, 0.30)
    assert r.probability_damaged > 0.99


def test_healthy_beam_is_reported_healthy(forward):
    """The best answer is 'no damage'. Some doubt remains (about 12 % here) because small damage
    near the free tip, where a cantilever hardly bends, barely changes the vibration at all."""
    m = simulate_measurement(BEAM, 0.5, 0.0, freq_noise=0.0, shape_noise=0.0)
    r = detect(m, BEAM, forward, locations=LOCS, severities=SEVS)
    assert r.severity == 0.0 and r.probability_damaged < 0.2


def test_posterior_is_a_probability_map(forward):
    rng = np.random.default_rng(0)
    m = simulate_measurement(BEAM, 0.62, 0.27, rng=rng)
    r = detect(m, BEAM, forward, locations=LOCS, severities=SEVS)
    assert r.posterior.shape == (len(LOCS), len(SEVS))
    assert np.isclose(r.posterior.sum(), 1.0) and np.all(r.posterior >= 0)
    assert r.location_interval[0] <= r.location <= r.location_interval[1]
    assert r.severity_interval[0] <= r.severity <= r.severity_interval[1]


def test_noisy_detection_lands_near_the_truth(forward):
    rng = np.random.default_rng(1)
    m = simulate_measurement(BEAM, 0.62, 0.27, freq_noise=0.001, shape_noise=0.01, rng=rng)
    r = detect(m, BEAM, forward, freq_noise=0.001, shape_noise=0.01, locations=LOCS, severities=SEVS)
    assert abs(r.location - 0.62) <= 0.05 and abs(r.severity - 0.27) <= 0.05


def test_frequencies_alone_also_work(forward):
    m = simulate_measurement(BEAM, 0.25, 0.40, freq_noise=0.0, shape_noise=0.0)
    r = detect(m, BEAM, forward, use_shapes=False, locations=LOCS, severities=SEVS)
    assert np.isclose(r.location, 0.25) and np.isclose(r.severity, 0.40)
