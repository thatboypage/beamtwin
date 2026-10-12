"""Checks on the functions behind the web app (Step 5a). Uses the exact FEM, so no trained model needed."""

import numpy as np
import pytest

from beamtwin import service
from beamtwin.damage.forward import FEMForward
from beamtwin.fem.modal import cantilever_exact_frequencies

STEEL = service.make_beam(1.0, 0.05, 0.01, 200e9, 7850)
LOCS = np.linspace(0.05, 0.95, 19)
SEVS = np.linspace(0.0, 0.5, 11)


def test_materials_lie_inside_the_trained_ranges():
    for m in service.MATERIALS.values():
        beam = service.make_beam(1.0, 0.05, 0.01, m["E"], m["rho"])
        assert service.check_beam(beam) == []


def test_out_of_range_beam_is_flagged():
    long_beam = service.make_beam(5.0, 0.05, 0.01, 200e9, 7850)
    rubbery = service.make_beam(1.0, 0.05, 0.01, 200e9, 1000)
    assert any("Length" in w for w in service.check_beam(long_beam))
    assert any("density" in w for w in service.check_beam(rubbery))


def test_analyze_matches_textbook():
    result = service.analyze(STEEL)
    I, A = 0.05 * 0.01**3 / 12, 0.05 * 0.01
    exact = cantilever_exact_frequencies(200e9, I, 7850, A, 1.0)
    assert np.allclose(result["frequencies_hz"], exact, rtol=1e-6)
    assert np.allclose(result["fem_frequencies_hz"], exact, rtol=1e-3)
    assert len(result["mode_shapes"]) == 4 and len(result["mode_shapes"][0]) == len(result["positions"])


def test_simulate_is_repeatable_with_a_seed():
    a = service.simulate(STEEL, 0.4, 0.2, "medium", seed=5)
    b = service.simulate(STEEL, 0.4, 0.2, "medium", seed=5)
    assert a["frequencies_hz"] == b["frequencies_hz"] and len(a["shapes"]) == 36


def test_detect_finds_noiseless_damage():
    m = service.simulate(STEEL, 0.35, 0.30, "low", seed=0)
    # replace the noisy frequencies with exact ones by simulating at zero noise
    from beamtwin.damage.detect import simulate_measurement
    exact = simulate_measurement(STEEL, 0.35, 0.30, 0.0, 0.0)
    r = service.detect_damage(STEEL, exact.frequencies.tolist(), exact.shapes.tolist(), "low",
                              forward=FEMForward(), locations=LOCS, severities=SEVS)
    assert np.isclose(r["location"], 0.35) and np.isclose(r["severity"], 0.30)
    assert r["used_mode_shapes"] and r["probability_damaged"] > 0.99
    assert len(r["posterior"]) == len(LOCS) and len(r["posterior"][0]) == len(SEVS)
    assert max(max(row) for row in r["posterior"]) == 1.0
    assert len(m["shapes"]) == 36


def test_detect_works_with_frequencies_only():
    from beamtwin.damage.detect import simulate_measurement
    exact = simulate_measurement(STEEL, 0.25, 0.40, 0.0, 0.0)
    r = service.detect_damage(STEEL, exact.frequencies.tolist(), None, "low",
                              forward=FEMForward(), locations=LOCS, severities=SEVS)
    assert not r["used_mode_shapes"]
    assert np.isclose(r["location"], 0.25) and np.isclose(r["severity"], 0.40)


def test_frequencies_above_healthy_raise_a_warning():
    too_high = (STEEL.healthy_frequencies() * 1.05).tolist()
    r = service.detect_damage(STEEL, too_high, None, "medium",
                              forward=FEMForward(), locations=LOCS, severities=SEVS)
    assert any("higher than the healthy" in w for w in r["warnings"])


def test_wrong_number_of_inputs_is_rejected():
    with pytest.raises(ValueError):
        service.detect_damage(STEEL, [1.0, 2.0, 3.0], forward=FEMForward())
    with pytest.raises(ValueError):
        service.detect_damage(STEEL, [1.0, 2.0, 3.0, 4.0], [0.0] * 10, forward=FEMForward())
