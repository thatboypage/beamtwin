"""Checks on surrogate inputs/targets and evaluation (Step 3b). No PyTorch needed."""

import numpy as np
import pytest

from beamtwin.fem.modal import cantilever_exact_frequencies
from beamtwin.surrogate.data_generation import generate_dataset
from beamtwin.surrogate.evaluation import evaluate, split_indices
from beamtwin.surrogate.features import (
    decode_outputs,
    encode_inputs,
    encode_targets,
    healthy_frequencies,
)


@pytest.fixture(scope="module")
def data():
    return generate_dataset(30, seed=5)


def test_healthy_frequencies_match_textbook(data):
    X, _ = data
    row = X[0]
    length, width, thickness, E, rho = row[:5]
    I, A = width * thickness**3 / 12, width * thickness
    expected = cantilever_exact_frequencies(E, I, rho, A, length)
    assert np.allclose(healthy_frequencies(X[:1])[0], expected)


@pytest.mark.parametrize("kind", ["plain", "hybrid"])
def test_encode_then_decode_gives_back_the_data(data, kind):
    X, Y = data
    assert np.allclose(decode_outputs(X, encode_targets(X, Y, kind), kind), Y)


def test_hybrid_uses_only_the_two_damage_inputs(data):
    X, _ = data
    assert encode_inputs(X, "hybrid").shape == (len(X), 2)
    assert encode_inputs(X, "plain").shape == (len(X), 7)


def test_hybrid_frequency_targets_are_ratios_at_most_one(data):
    """Damage can only lower frequencies, so f_damaged / f_healthy <= 1."""
    X, Y = data
    ratios = encode_targets(X, Y, "hybrid")[:, :4]
    assert np.all(ratios <= 1 + 1e-6) and np.all(ratios > 0.8)


def test_unknown_kind_is_rejected(data):
    X, _ = data
    with pytest.raises(ValueError):
        encode_inputs(X, "magic")


def test_split_covers_everything_once():
    tr, va, te = split_indices(1000, seed=0)
    assert (len(tr), len(va), len(te)) == (700, 150, 150)
    assert len(np.unique(np.concatenate([tr, va, te]))) == 1000


def test_perfect_prediction_scores_zero_error(data):
    X, Y = data
    m = evaluate(X, Y, Y.copy())
    assert np.allclose(m["freq_error_pct_mean"], 0)
    assert np.allclose(m["drop_error_pct_pts_mean"], 0)
    assert m["shape_mae"] == 0
