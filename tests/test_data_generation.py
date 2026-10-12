"""Checks on the surrogate training-data generator (Step 3a)."""

import numpy as np

from beamtwin.fem.modal import cantilever_exact_frequencies
from beamtwin.surrogate.data_generation import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    PARAMETER_RANGES,
    generate_dataset,
    sample_parameters,
    simulate_one,
)


def test_shapes_match_names():
    X, Y = generate_dataset(20, seed=1)
    assert X.shape == (20, len(INPUT_NAMES))
    assert Y.shape == (20, len(OUTPUT_NAMES)) == (20, 40)


def test_same_seed_gives_same_data():
    X1, Y1 = generate_dataset(10, seed=3)
    X2, Y2 = generate_dataset(10, seed=3)
    assert np.array_equal(X1, X2) and np.array_equal(Y1, Y2)


def test_samples_stay_inside_their_ranges():
    X = sample_parameters(500, seed=0)
    for col in [0, 1, 2, 3, 5, 6]:                      # rho (col 4) is derived
        _, low, high, _ = PARAMETER_RANGES[col]
        assert X[:, col].min() >= low and X[:, col].max() <= high


def test_materials_are_realistic():
    """E / rho must stay in the range of real structural metals."""
    X = sample_parameters(500, seed=0)
    specific = X[:, 3] / X[:, 4]
    assert specific.min() >= 24e6 * (1 - 1e-12) and specific.max() <= 27e6 * (1 + 1e-12)


def test_healthy_sample_matches_textbook():
    length, width, thickness, E, rho = 1.2, 0.04, 0.008, 70e9, 2700
    out = simulate_one(length, width, thickness, E, rho, damage_location=0.5, damage_severity=0.0)
    I, A = width * thickness**3 / 12, width * thickness
    assert np.allclose(out[:4], cantilever_exact_frequencies(E, I, rho, A, length), rtol=1e-4)


def test_frequencies_are_sorted_and_positive():
    _, Y = generate_dataset(50, seed=2)
    f = Y[:, :4]
    assert np.all(f > 0) and np.all(np.diff(f, axis=1) > 0)
