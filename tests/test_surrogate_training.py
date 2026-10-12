"""Checks on the PyTorch surrogate (Step 3b). Skipped automatically if PyTorch is missing."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beamtwin.surrogate.data_generation import generate_dataset  # noqa: E402
from beamtwin.surrogate.evaluation import split_indices  # noqa: E402
from beamtwin.surrogate.model import Surrogate  # noqa: E402
from beamtwin.surrogate.train import train_surrogate  # noqa: E402


@pytest.fixture(scope="module")
def small_split():
    X, Y = generate_dataset(200, seed=7)
    tr, va, te = split_indices(len(X), seed=0)
    return X[tr], Y[tr], X[va], Y[va], X[te], Y[te]


@pytest.mark.parametrize("kind", ["plain", "hybrid"])
def test_training_reduces_validation_loss(small_split, kind):
    X_tr, Y_tr, X_va, Y_va, _, _ = small_split
    _, history = train_surrogate(X_tr, Y_tr, X_va, Y_va, kind, hidden=(32, 32), max_epochs=60)
    assert min(history["val_loss"]) < history["val_loss"][0]


@pytest.mark.parametrize("kind", ["plain", "hybrid"])
def test_prediction_shape(small_split, kind):
    X_tr, Y_tr, X_va, Y_va, X_te, _ = small_split
    surrogate, _ = train_surrogate(X_tr, Y_tr, X_va, Y_va, kind, hidden=(16,), max_epochs=5)
    assert surrogate.predict(X_te).shape == (len(X_te), 40)
    assert surrogate.predict(X_te[0]).shape == (1, 40)          # a single beam also works


def test_save_and_load_give_identical_predictions(small_split, tmp_path):
    X_tr, Y_tr, X_va, Y_va, X_te, _ = small_split
    surrogate, _ = train_surrogate(X_tr, Y_tr, X_va, Y_va, "hybrid", hidden=(16,), max_epochs=5)
    path = tmp_path / "s.pt"
    surrogate.save(path)
    loaded = Surrogate.load(path)
    assert loaded.kind == "hybrid"
    assert np.array_equal(loaded.predict(X_te), surrogate.predict(X_te))
