"""Checks on serving the network with NumPy only (Step 5a)."""

import numpy as np
import pytest

from beamtwin.surrogate.data_generation import generate_dataset
from beamtwin.surrogate.export import NumpySurrogate, silu


def tiny_network(kind="hybrid", seed=0):
    rng = np.random.default_rng(seed)
    n_in = 2 if kind == "hybrid" else 7
    sizes = [n_in, 5, 5, 40]
    weights = [rng.normal(size=(o, i)) * 0.1 for i, o in zip(sizes[:-1], sizes[1:])]
    biases = [rng.normal(size=o) * 0.1 for o in sizes[1:]]
    return NumpySurrogate(kind, weights, biases, np.zeros(n_in), np.ones(n_in), np.zeros(40), np.ones(40))


def test_silu_matches_its_definition():
    x = np.linspace(-5, 5, 11)
    assert np.allclose(silu(x), x / (1 + np.exp(-x)))


def test_forward_pass_matches_a_hand_calculation():
    net = tiny_network("hybrid")
    X = generate_dataset(3, seed=0)[0]
    h = X[:, 5:7]
    h = silu(h @ net.weights[0].T + net.biases[0])
    h = silu(h @ net.weights[1].T + net.biases[1])
    t = h @ net.weights[2].T + net.biases[2]
    from beamtwin.surrogate.features import decode_outputs
    assert np.allclose(net.predict(X), decode_outputs(X, t, "hybrid"))


def test_save_and_load_round_trip(tmp_path):
    net = tiny_network("plain")
    X = generate_dataset(4, seed=1)[0]
    net.save(tmp_path / "m.npz")
    loaded = NumpySurrogate.load(tmp_path / "m.npz")
    assert loaded.kind == "plain"
    assert np.array_equal(loaded.predict(X), net.predict(X))


def test_numpy_copy_matches_pytorch(tmp_path):
    pytest.importorskip("torch")
    from beamtwin.surrogate.evaluation import split_indices
    from beamtwin.surrogate.export import from_torch
    from beamtwin.surrogate.train import train_surrogate

    X, Y = generate_dataset(120, seed=3)
    tr, va, te = split_indices(len(X), seed=0)
    torch_model, _ = train_surrogate(X[tr], Y[tr], X[va], Y[va], "hybrid", hidden=(16, 16), max_epochs=5)
    numpy_model = from_torch(torch_model)
    assert np.allclose(numpy_model.predict(X[te]), torch_model.predict(X[te]), rtol=1e-5, atol=1e-6)
