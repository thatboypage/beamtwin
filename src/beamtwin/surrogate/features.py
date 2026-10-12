"""Turn raw beam data into what the network sees, and back again (Step 3b).

Two designs are supported, so they can be compared fairly:

plain
    The network gets all 7 inputs and must predict everything itself.
    Inputs:  log(length, width, thickness, E, rho), damage location, damage severity
    Targets: log(f1..f4), 36 mode-shape sensor values

hybrid  (physics + learning)
    The healthy frequencies come from the exact formula. The network only
    learns what damage changes.
    Inputs:  damage location, damage severity
    Targets: f_damaged / f_healthy for modes 1-4, 36 mode-shape sensor values

Why can the hybrid ignore length, width, thickness, E and rho? The damage
multiplies EI by the same profile whatever the beam's size or material, and
frequency scales with sqrt(EI / (rho A L^4)) for every mode. Those common
factors cancel in the ratio f_damaged / f_healthy, and the mode shapes
(expressed along 0..1 of the length) do not depend on them either. So
physics shrinks the learning problem from 7 inputs to 2.

This module needs only numpy, so it can be tested without PyTorch.
"""

import numpy as np

from beamtwin.fem.modal import CANTILEVER_BETA_L

KINDS = ("plain", "hybrid")
N_FREQS = 4


def healthy_frequencies(X: np.ndarray) -> np.ndarray:
    """Exact natural frequencies (Hz) of the undamaged beams, shape (n, 4).

    X columns: length, width, thickness, E, rho, damage_location, damage_severity.
    """
    length, width, thickness, E, rho = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    I = width * thickness**3 / 12
    A = width * thickness
    scale = np.sqrt(E * I / (rho * A * length**4)) / (2 * np.pi)
    return CANTILEVER_BETA_L[None, :N_FREQS] ** 2 * scale[:, None]


def encode_inputs(X: np.ndarray, kind: str) -> np.ndarray:
    """Network inputs for each design."""
    _check_kind(kind)
    if kind == "plain":
        # Logs turn multiplications (like E * I / L^4) into additions,
        # and squash ranges that span 10x or more into similar sizes.
        return np.column_stack([np.log(X[:, :5]), X[:, 5:7]])
    return X[:, 5:7].copy()


def encode_targets(X: np.ndarray, Y: np.ndarray, kind: str) -> np.ndarray:
    """Network targets for each design. Y columns: f1..f4, then 36 sensor values."""
    _check_kind(kind)
    freqs, shapes = Y[:, :N_FREQS], Y[:, N_FREQS:]
    if kind == "plain":
        return np.column_stack([np.log(freqs), shapes])
    return np.column_stack([freqs / healthy_frequencies(X), shapes])


def decode_outputs(X: np.ndarray, T: np.ndarray, kind: str) -> np.ndarray:
    """Undo encode_targets: network outputs -> frequencies (Hz) and sensor values."""
    _check_kind(kind)
    head, shapes = T[:, :N_FREQS], T[:, N_FREQS:]
    if kind == "plain":
        freqs = np.exp(head)
    else:
        freqs = head * healthy_frequencies(X)
    return np.column_stack([freqs, shapes])


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
