"""Serve the trained network without PyTorch (Step 5a).

A trained network is just a few tables of numbers (weights and biases) plus a
simple rule for combining them. Training needs PyTorch, but using the finished
network does not: this module saves those tables to a small NumPy file and redoes
the calculation in plain NumPy. The web app then needs no PyTorch at all, so it
is about 200 MB smaller and starts faster.

Export after training (needs PyTorch):

    python -m beamtwin.surrogate.export

which writes models/surrogate_hybrid.npz next to models/surrogate_hybrid.pt.
"""

import argparse
from pathlib import Path

import numpy as np

from beamtwin.surrogate.features import decode_outputs, encode_inputs


def silu(x: np.ndarray) -> np.ndarray:
    """The activation used in the network: x * sigmoid(x). Same as torch.nn.SiLU."""
    return x / (1.0 + np.exp(-x))


class NumpySurrogate:
    """The trained surrogate, run with NumPy only. Same predict() as the PyTorch Surrogate."""

    def __init__(self, kind, weights, biases, x_mean, x_std, t_mean, t_std):
        self.kind = str(kind)
        self.weights = [np.asarray(w, dtype=np.float64) for w in weights]   # each (out, in), like torch
        self.biases = [np.asarray(b, dtype=np.float64) for b in biases]
        self.x_mean, self.x_std = np.asarray(x_mean, float), np.asarray(x_std, float)
        self.t_mean, self.t_std = np.asarray(t_mean, float), np.asarray(t_std, float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Raw inputs (n, 7) -> outputs (n, 40): f1..f4 in Hz, then 36 sensor values."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        h = (encode_inputs(X, self.kind) - self.x_mean) / self.x_std
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W.T + b                          # weighted sum, exactly like nn.Linear
            if i < len(self.weights) - 1:            # every layer except the last gets the bend
                h = silu(h)
        return decode_outputs(X, h * self.t_std + self.t_mean, self.kind)

    def save(self, path: Path) -> None:
        arrays = {"kind": np.array(self.kind), "n_layers": np.array(len(self.weights)),
                  "x_mean": self.x_mean, "x_std": self.x_std, "t_mean": self.t_mean, "t_std": self.t_std}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "NumpySurrogate":
        d = np.load(Path(path))
        n = int(d["n_layers"])
        return cls(str(d["kind"]), [d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)],
                   d["x_mean"], d["x_std"], d["t_mean"], d["t_std"])


def from_torch(surrogate) -> NumpySurrogate:
    """Copy the weights out of a trained PyTorch Surrogate."""
    from torch import nn

    linears = [m for m in surrogate.model.net if isinstance(m, nn.Linear)]
    return NumpySurrogate(
        surrogate.kind,
        [m.weight.detach().numpy() for m in linears],
        [m.bias.detach().numpy() for m in linears],
        surrogate.x_scaler.mean, surrogate.x_scaler.std,
        surrogate.t_scaler.mean, surrogate.t_scaler.std,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/surrogate_hybrid.pt"))
    args = parser.parse_args()

    from beamtwin.surrogate.model import Surrogate      # needs PyTorch
    torch_model = Surrogate.load(args.model)
    numpy_model = from_torch(torch_model)
    out = args.model.with_suffix(".npz")
    numpy_model.save(out)

    # Check the two give the same answers on random beams inside the training ranges.
    from beamtwin.surrogate.data_generation import sample_parameters
    X = sample_parameters(500, seed=99)
    diff = np.abs(numpy_model.predict(X) - torch_model.predict(X)).max()
    print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KB). "
          f"Largest difference from PyTorch on 500 beams: {diff:.2e}")


if __name__ == "__main__":
    main()
