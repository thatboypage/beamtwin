"""Forward models: given a beam and candidate damage, predict what the sensors would read (Step 4).

A damage detector is a search: try many candidate cracks, predict the vibration
for each, and keep the ones that match the measurement. The "forward model" is
the predictor inside that search. Two are provided, with the same interface:

FEMForward        exact but slow (about 2.5 ms per candidate); used for testing
SurrogateForward  the trained hybrid network; thousands of candidates in one call

Both return, for each candidate:
    ratios : f_damaged / f_healthy for modes 1-4             shape (n, 4)
    shapes : mode shapes 1-4 at the 9 sensors (tip = 1)       shape (n, 36)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beamtwin.surrogate.features import healthy_frequencies


@dataclass(frozen=True)
class Beam:
    """The known properties of the beam being monitored (SI units)."""
    length: float
    width: float
    thickness: float
    E: float
    rho: float

    def inputs(self, locations, severities) -> np.ndarray:
        """Build the 7-column input array used across the project, one row per candidate."""
        locations, severities = np.broadcast_arrays(np.asarray(locations, float), np.asarray(severities, float))
        n = locations.size
        fixed = np.tile([self.length, self.width, self.thickness, self.E, self.rho], (n, 1))
        return np.column_stack([fixed, locations.ravel(), severities.ravel()])

    def healthy_frequencies(self) -> np.ndarray:
        """Exact natural frequencies (Hz) of this beam without damage, modes 1-4."""
        return healthy_frequencies(self.inputs([0.5], [0.0]))[0]


class FEMForward:
    """Exact forward model: runs the FEM once per candidate."""

    def __call__(self, beam: Beam, locations, severities):
        from beamtwin.surrogate.data_generation import simulate_one   # here to avoid a circular import

        X = beam.inputs(locations, severities)
        Y = np.array([simulate_one(*row) for row in X])
        return Y[:, :4] / beam.healthy_frequencies(), Y[:, 4:]


class SurrogateForward:
    """Fast forward model: the trained hybrid surrogate from Step 3.

    model_path ending in .pt  -> the PyTorch model (needs PyTorch)
    model_path ending in .npz -> the exported NumPy copy (no PyTorch needed; used by the web app)
    """

    def __init__(self, model_path: Path = Path("models/surrogate_hybrid.pt")):
        model_path = Path(model_path)
        if model_path.suffix == ".npz":
            from beamtwin.surrogate.export import NumpySurrogate
            self.surrogate = NumpySurrogate.load(model_path)
        else:
            from beamtwin.surrogate.model import Surrogate  # imported here so numpy-only users never need torch
            self.surrogate = Surrogate.load(model_path)
        if self.surrogate.kind != "hybrid":
            raise ValueError("SurrogateForward expects the hybrid surrogate")

    def __call__(self, beam: Beam, locations, severities):
        Y = self.surrogate.predict(beam.inputs(locations, severities))
        return Y[:, :4] / beam.healthy_frequencies(), Y[:, 4:]
