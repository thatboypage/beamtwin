"""The neural network and a small wrapper that makes it easy to use (Step 3b)."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from beamtwin.surrogate.features import decode_outputs, encode_inputs


class Standardizer:
    """Shift and scale each column to mean 0, standard deviation 1.

    Networks train best when every input and target is roughly the same size.
    Without this, a column measured in pascals (about 1e11) would swamp one
    that is between 0 and 1.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, data: np.ndarray) -> "Standardizer":
        std = data.std(axis=0)
        return cls(data.mean(axis=0), np.where(std > 0, std, 1.0))

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def state(self) -> dict:
        return {"mean": torch.tensor(self.mean), "std": torch.tensor(self.std)}

    @classmethod
    def from_state(cls, state: dict) -> "Standardizer":
        return cls(state["mean"].numpy(), state["std"].numpy())


class MLP(nn.Module):
    """A multilayer perceptron: layers of weighted sums, each followed by a bend (activation).

    n_in -> 64 -> 64 -> 64 -> n_out  (with the default hidden sizes)
    """

    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...] = (64, 64, 64)):
        super().__init__()
        layers, width = [], n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.SiLU()]
            width = h
        layers.append(nn.Linear(width, n_out))          # last layer: no bend, any value allowed
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Surrogate:
    """A trained network plus everything needed to use it on raw beam data."""

    def __init__(self, kind: str, model: MLP, x_scaler: Standardizer, t_scaler: Standardizer,
                 hidden: tuple[int, ...]):
        self.kind = kind
        self.model = model
        self.x_scaler = x_scaler
        self.t_scaler = t_scaler
        self.hidden = tuple(hidden)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Raw inputs (n, 7) -> predicted outputs (n, 40): f1..f4 in Hz, then 36 sensor values."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        x = self.x_scaler.transform(encode_inputs(X, self.kind))
        self.model.eval()
        t = self.model(torch.tensor(x, dtype=torch.float32)).numpy().astype(np.float64)
        return decode_outputs(X, self.t_scaler.inverse(t), self.kind)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "kind": self.kind,
            "hidden": list(self.hidden),
            "n_in": self.model.net[0].in_features,
            "n_out": self.model.net[-1].out_features,
            "weights": self.model.state_dict(),
            "x_scaler": self.x_scaler.state(),
            "t_scaler": self.t_scaler.state(),
        }, path)

    @classmethod
    def load(cls, path: Path) -> "Surrogate":
        saved = torch.load(Path(path), weights_only=True)
        model = MLP(saved["n_in"], saved["n_out"], tuple(saved["hidden"]))
        model.load_state_dict(saved["weights"])
        return cls(saved["kind"], model,
                   Standardizer.from_state(saved["x_scaler"]),
                   Standardizer.from_state(saved["t_scaler"]),
                   tuple(saved["hidden"]))
