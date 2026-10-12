"""Everything the web app can do, as plain Python functions (Step 5a).

The web API (api/main.py) is a thin layer that receives requests, calls these
functions and sends back their results. Keeping the real work here means it can
be tested without a web server, and the same functions could later sit behind a
different interface (a command line tool, another website, a notebook).

Every function takes and returns plain numbers, lists and dictionaries, which
translate directly to and from JSON, the text format websites exchange.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from beamtwin.damage.detect import (
    DEFAULT_LOCATIONS,
    DEFAULT_SEVERITIES,
    Measurement,
    detect,
    simulate_measurement,
)
from beamtwin.damage.forward import Beam, SurrogateForward
from beamtwin.fem.modal import cantilever_modes
from beamtwin.surrogate.data_generation import PARAMETER_RANGES

# Typical values for structural metals. E / rho is about 25-26 MJ/kg for all three,
# which is why the surrogate's training data could tie density to stiffness.
MATERIALS = {
    "steel":     {"label": "Steel",                    "E": 200e9, "rho": 7850},
    "aluminium": {"label": "Aluminium alloy",          "E": 70e9,  "rho": 2700},
    "titanium":  {"label": "Titanium alloy (Ti-6Al-4V)", "E": 114e9, "rho": 4430},
}

# Sensor noise presets: (relative frequency noise, absolute mode-shape noise).
NOISE_LEVELS = {
    "low":    {"label": "Low (careful lab test)",  "freq": 0.001, "shape": 0.01},
    "medium": {"label": "Medium (good field test)", "freq": 0.002, "shape": 0.02},
    "high":   {"label": "High (noisy field test)",  "freq": 0.005, "shape": 0.05},
}

_RANGES = {name: (low, high) for name, low, high, _ in PARAMETER_RANGES}
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "surrogate_hybrid.npz"


@lru_cache(maxsize=2)
def get_forward(model_path: str = str(DEFAULT_MODEL)) -> SurrogateForward:
    """Load the surrogate once and reuse it for every request (loading is the slow part)."""
    return SurrogateForward(Path(model_path))


def make_beam(length: float, width: float, thickness: float, E: float, rho: float) -> Beam:
    return Beam(float(length), float(width), float(thickness), float(E), float(rho))


def check_beam(beam: Beam) -> list[str]:
    """Warnings for any property outside the range the surrogate was trained on."""
    warnings = []
    checks = [("length", beam.length, "m", 1), ("width", beam.width, "mm", 1000),
              ("thickness", beam.thickness, "mm", 1000), ("E", beam.E, "GPa", 1e-9)]
    for name, value, unit, scale in checks:
        low, high = _RANGES[name]
        if not low <= value <= high:
            warnings.append(f"{name.capitalize()} {value * scale:g} {unit} is outside the trained range "
                            f"{low * scale:g}-{high * scale:g} {unit}; results are extrapolations.")
    specific = beam.E / beam.rho
    low, high = _RANGES["specific_stiffness"]
    if not low <= specific <= high:
        warnings.append(f"E / density = {specific / 1e6:.1f} MJ/kg is outside the trained range "
                        f"{low / 1e6:.0f}-{high / 1e6:.0f} MJ/kg (typical of structural metals); "
                        f"results are extrapolations.")
    return warnings


def analyze(beam: Beam, n_points: int = 51) -> dict:
    """Healthy-beam vibration: natural frequencies and mode shapes along the beam (for plotting)."""
    I = beam.width * beam.thickness**3 / 12
    A = beam.width * beam.thickness
    freqs, shapes = cantilever_modes(beam.E, I, beam.rho, A, beam.length, n_elements=n_points - 1, n_modes=4)
    return {
        "frequencies_hz": [round(float(f), 4) for f in beam.healthy_frequencies()],
        "fem_frequencies_hz": [round(float(f), 4) for f in freqs],
        "positions": np.linspace(0, 1, n_points).round(4).tolist(),
        "mode_shapes": [shapes[0::2, m].round(5).tolist() for m in range(4)],
        "mass_kg": round(beam.rho * A * beam.length, 4),
        "warnings": check_beam(beam),
    }


def simulate(beam: Beam, location: float, severity: float, noise: str = "medium",
             seed: int | None = None) -> dict:
    """A realistic noisy measurement of a beam with known damage (for demos and testing)."""
    level = NOISE_LEVELS[noise]
    m = simulate_measurement(beam, location, severity, level["freq"], level["shape"],
                             np.random.default_rng(seed))
    return {"frequencies_hz": m.frequencies.round(5).tolist(), "shapes": m.shapes.round(5).tolist(),
            "noise": noise, "warnings": check_beam(beam)}


def detect_damage(beam: Beam, frequencies: list[float], shapes: list[float] | None = None,
                  noise: str = "medium", forward=None,
                  locations: np.ndarray = DEFAULT_LOCATIONS,
                  severities: np.ndarray = DEFAULT_SEVERITIES) -> dict:
    """Find damage from measured frequencies (and, if given, mode shapes at the 9 sensors)."""
    if len(frequencies) != 4:
        raise ValueError("exactly 4 natural frequencies are needed")
    if shapes is not None and len(shapes) != 36:
        raise ValueError("mode shapes need 36 values (4 modes x 9 sensors)")
    level = NOISE_LEVELS[noise]
    forward = get_forward() if forward is None else forward
    use_shapes = shapes is not None
    measurement = Measurement(np.asarray(frequencies, float),
                              np.asarray(shapes, float) if use_shapes else np.zeros(36))
    r = detect(measurement, beam, forward, level["freq"], level["shape"], use_shapes=use_shapes,
               locations=locations, severities=severities)

    healthy = beam.healthy_frequencies()
    drops = (1 - np.asarray(frequencies, float) / healthy) * 100
    warnings = check_beam(beam)
    if np.any(drops < -3 * level["freq"] * 100):
        warnings.append("Some measured frequencies are higher than the healthy beam's. Damage only lowers "
                        "frequencies, so check the beam properties and the measurement.")
    return {
        "location": r.location, "severity": r.severity,
        "location_interval": list(r.location_interval), "severity_interval": list(r.severity_interval),
        "probability_damaged": round(r.probability_damaged, 4),
        "summary": r.summary(),
        "used_mode_shapes": use_shapes,
        "frequency_drop_pct": drops.round(3).tolist(),
        "healthy_frequencies_hz": healthy.round(4).tolist(),
        # Probability map scaled so the most likely candidate = 1, rounded to keep the reply small.
        "posterior": (r.posterior / r.posterior.max()).round(4).tolist(),
        "locations": np.asarray(locations).round(4).tolist(),
        "severities": np.asarray(severities).round(4).tolist(),
        "seconds": round(r.seconds, 4),
        "warnings": warnings,
    }
