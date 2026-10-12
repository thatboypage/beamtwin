"""Find damage from vibration measurements: the inverse problem (Step 4).

Forward problem (Steps 1-3): damage -> vibration.
Inverse problem (here):      vibration -> damage.

Method (a grid search with a probability map):

1. Lay out a grid of candidate damage: every location from 5 % to 95 % of the
   length, every severity from 0 to 50 %.
2. Ask the forward model what each candidate would make the sensors read.
3. Score each candidate by how far its prediction is from the measurement,
   counting each difference in units of the expected sensor noise ("misfit").
4. Turn misfits into probabilities: P(candidate) is proportional to
   exp(-misfit / 2). This is Bayes' rule with Gaussian sensor noise and no prior
   preference for any candidate.
5. Report the most probable candidate, plus a 90 % range for location and
   severity, so the answer comes with an honest uncertainty.
"""

import time
from dataclasses import dataclass

import numpy as np

from beamtwin.damage.forward import Beam
from beamtwin.surrogate.data_generation import simulate_one

DEFAULT_LOCATIONS = np.linspace(0.05, 0.95, 181)      # every 0.5 % of the length
DEFAULT_SEVERITIES = np.linspace(0.0, 0.5, 101)       # every 0.5 % of stiffness


@dataclass
class Measurement:
    """What the sensors report: four natural frequencies and four mode shapes at nine sensors."""
    frequencies: np.ndarray      # Hz, shape (4,)
    shapes: np.ndarray           # shape (36,): mode 1 at 9 sensors, then mode 2, ...


def simulate_measurement(beam: Beam, location: float, severity: float,
                         freq_noise: float = 0.002, shape_noise: float = 0.02,
                         rng: np.random.Generator | None = None) -> Measurement:
    """Create a realistic measurement from the exact FEM, with sensor noise added.

    freq_noise  : relative error of each measured frequency (0.002 = 0.2 %)
    shape_noise : absolute error of each mode-shape value (mode shapes scaled so the tip is 1)

    The measurement comes from the FEM, NOT from the surrogate used to detect the
    damage. Generating and inverting data with the same model (the "inverse crime")
    makes a method look better than it really is.
    """
    rng = np.random.default_rng() if rng is None else rng
    y = simulate_one(beam.length, beam.width, beam.thickness, beam.E, beam.rho, location, severity)
    freqs = y[:4] * (1 + freq_noise * rng.standard_normal(4))
    shapes = y[4:] + shape_noise * rng.standard_normal(36)
    return Measurement(freqs, shapes)


@dataclass
class DetectionResult:
    location: float                     # most probable damage centre (fraction of length)
    severity: float                     # most probable stiffness loss (0..1)
    location_interval: tuple            # 90 % range for location
    severity_interval: tuple            # 90 % range for severity
    probability_damaged: float          # P(severity >= damage_threshold)
    posterior: np.ndarray               # probability of every candidate, shape (n_loc, n_sev)
    locations: np.ndarray
    severities: np.ndarray
    seconds: float

    def summary(self) -> str:
        lo, hi = self.location_interval
        slo, shi = self.severity_interval
        return (f"Damage at {self.location * 100:.1f} % of the length (90 % range {lo * 100:.1f}-{hi * 100:.1f} %), "
                f"severity {self.severity * 100:.1f} % (90 % range {slo * 100:.1f}-{shi * 100:.1f} %), "
                f"P(damaged) = {self.probability_damaged:.2f}")


def detect(measurement: Measurement, beam: Beam, forward,
           freq_noise: float = 0.002, shape_noise: float = 0.02, use_shapes: bool = True,
           locations: np.ndarray = DEFAULT_LOCATIONS, severities: np.ndarray = DEFAULT_SEVERITIES,
           damage_threshold: float = 0.05) -> DetectionResult:
    """Locate and size damage by comparing the measurement with every candidate on a grid."""
    start = time.perf_counter()
    loc_grid, sev_grid = np.meshgrid(locations, severities, indexing="ij")
    pred_ratios, pred_shapes = forward(beam, loc_grid.ravel(), sev_grid.ravel())

    measured_ratios = measurement.frequencies / beam.healthy_frequencies()
    misfit = (((pred_ratios - measured_ratios) / freq_noise) ** 2).sum(axis=1)
    if use_shapes:
        misfit += (((pred_shapes - measurement.shapes) / shape_noise) ** 2).sum(axis=1)
    misfit = misfit.reshape(loc_grid.shape)

    posterior = np.exp(-0.5 * (misfit - misfit.min()))      # subtract the minimum to avoid underflow
    posterior /= posterior.sum()

    i, j = np.unravel_index(np.argmax(posterior), posterior.shape)
    return DetectionResult(
        location=float(locations[i]),
        severity=float(severities[j]),
        location_interval=_interval(locations, posterior.sum(axis=1)),
        severity_interval=_interval(severities, posterior.sum(axis=0)),
        probability_damaged=float(posterior[:, severities >= damage_threshold].sum()),
        posterior=posterior,
        locations=np.asarray(locations),
        severities=np.asarray(severities),
        seconds=time.perf_counter() - start,
    )


def _interval(values: np.ndarray, weights: np.ndarray, level: float = 0.90) -> tuple:
    """Smallest and largest grid value inside the central `level` share of the probability."""
    cumulative = np.cumsum(weights)
    low = values[np.searchsorted(cumulative, (1 - level) / 2)]
    high = values[min(np.searchsorted(cumulative, 1 - (1 - level) / 2), len(values) - 1)]
    return float(low), float(high)
