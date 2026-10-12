"""Generate a training dataset for the neural surrogate by running the FEM many times (Step 3a).

Each sample is one beam with one damage zone:

    inputs  (7): length, width, thickness, E, rho, damage location, damage severity
    outputs (40): first 4 natural frequencies (Hz)
                  + mode shapes 1-4 measured at 9 "sensor" points along the beam

Run from the project root:

    python -m beamtwin.surrogate.data_generation --n-samples 4000

The dataset is saved to data/processed/ (not tracked by Git). Anyone can
recreate it exactly, because the random sampling uses a fixed seed.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from beamtwin.damage.simulate import damage_profile
from beamtwin.fem.modal import cantilever_modes

N_ELEMENTS = 50            # FEM mesh; nodes every 2% of the length
N_MODES = 4
DAMAGE_WIDTH = 0.04        # fraction of the length (fixed for this dataset)

# Sensors at 10%, 20%, ..., 90% of the length. They sit exactly on nodes
# (node k is at k/50 of the length). The tip is left out: every mode shape is
# scaled so the tip equals 1, so a tip sensor would carry no information.
SENSOR_POSITIONS = np.linspace(0.1, 0.9, 9)
SENSOR_NODES = np.rint(SENSOR_POSITIONS * N_ELEMENTS).astype(int)

# Ranges to sample. Each row: name, low, high, unit.
PARAMETER_RANGES = [
    ("length",            0.5,    2.0,    "m"),
    ("width",             0.02,   0.10,   "m"),
    ("thickness",         0.005,  0.020,  "m"),
    ("E",                 70e9,   210e9,  "Pa"),
    ("specific_stiffness", 24e6,  27e6,   "J/kg"),    # E / rho, see note below
    ("damage_location",   0.05,   0.95,   "-"),
    ("damage_severity",   0.0,    0.5,    "-"),
]
# Note on E and rho: for common structural metals (steel, aluminium, titanium)
# E / rho is almost the same, about 25 MJ/kg. Sampling E and rho independently
# would invent impossible materials (as stiff as steel, as light as aluminium).
# So we sample E and E/rho, then compute rho = E / (E/rho).

INPUT_NAMES = ["length", "width", "thickness", "E", "rho", "damage_location", "damage_severity"]
OUTPUT_NAMES = (
    [f"f{m}" for m in range(1, N_MODES + 1)]
    + [f"mode{m}_x{p:.1f}" for m in range(1, N_MODES + 1) for p in SENSOR_POSITIONS]
)


def sample_parameters(n_samples: int, seed: int = 0) -> np.ndarray:
    """Spread n_samples points evenly over the parameter ranges (Latin hypercube).

    Returns an (n_samples, 7) array in the order of INPUT_NAMES.
    """
    lows = np.array([r[1] for r in PARAMETER_RANGES])
    highs = np.array([r[2] for r in PARAMETER_RANGES])
    unit_cube = qmc.LatinHypercube(d=len(PARAMETER_RANGES), seed=seed).random(n_samples)
    raw = qmc.scale(unit_cube, lows, highs)

    params = raw.copy()
    params[:, 4] = raw[:, 3] / raw[:, 4]     # rho = E / (E/rho)
    return params


def simulate_one(length, width, thickness, E, rho, damage_location, damage_severity):
    """Run the FEM for one beam. Returns the 40 outputs in the order of OUTPUT_NAMES."""
    I = width * thickness**3 / 12
    A = width * thickness
    factors = damage_profile(N_ELEMENTS, damage_location, damage_severity, DAMAGE_WIDTH)
    freqs, shapes = cantilever_modes(E, I, rho, A, length, N_ELEMENTS, N_MODES, factors)
    sensor_values = shapes[2 * SENSOR_NODES, :]       # deflection rows at sensor nodes
    return np.concatenate([freqs, sensor_values.T.ravel()])


def generate_dataset(n_samples: int, seed: int = 0):
    """Return (X, Y): inputs (n, 7) and FEM outputs (n, 40)."""
    X = sample_parameters(n_samples, seed)
    Y = np.array([simulate_one(*row) for row in X])
    return X, Y


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/processed/beam_dataset.npz"))
    args = parser.parse_args()

    start = time.perf_counter()
    X, Y = generate_dataset(args.n_samples, args.seed)
    seconds = time.perf_counter() - start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, Y=Y,
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        sensor_positions=SENSOR_POSITIONS, n_elements=N_ELEMENTS,
        damage_width=DAMAGE_WIDTH, seed=args.seed,
    )
    print(f"Generated {len(X)} samples in {seconds:.1f} s "
          f"({seconds / len(X) * 1000:.2f} ms per FEM run) -> {args.out}")


if __name__ == "__main__":
    main()
