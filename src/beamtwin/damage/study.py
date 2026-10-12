"""How well does the detector find damage? A test on many simulated beams (Step 4).

For each test case: pick a random beam (inside the surrogate's training ranges)
and random damage, create a noisy measurement with the exact FEM, run the
detector with the surrogate, and compare the answer with the truth. Some cases
are healthy beams, to measure false alarms.

Run from the project root (needs the trained hybrid surrogate from Step 3):

    python -m beamtwin.damage.study

Results are printed and saved to data/processed/damage_study.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from beamtwin.damage.detect import detect, simulate_measurement
from beamtwin.damage.forward import Beam, SurrogateForward
from beamtwin.surrogate.data_generation import sample_parameters

# Sensor noise levels: (relative frequency noise, absolute mode-shape noise)
NOISE_LEVELS = {
    "low":    (0.001, 0.01),     # careful lab measurement
    "medium": (0.002, 0.02),     # good field measurement
    "high":   (0.005, 0.05),     # noisy field measurement
}


def run_case(beam, location, severity, forward, freq_noise, shape_noise, rng, use_shapes=True):
    measurement = simulate_measurement(beam, location, severity, freq_noise, shape_noise, rng)
    result = detect(measurement, beam, forward, freq_noise, shape_noise, use_shapes=use_shapes)
    lo, hi = result.location_interval
    slo, shi = result.severity_interval
    return {
        "true_location": location, "true_severity": severity,
        "location": result.location, "severity": result.severity,
        "location_error": abs(result.location - location),
        "severity_error": abs(result.severity - severity),
        "location_in_interval": bool(lo <= location <= hi),
        "severity_in_interval": bool(slo <= severity <= shi),
        "probability_damaged": result.probability_damaged,
        "seconds": result.seconds,
    }


def summarise(cases, damaged_threshold=0.05):
    damaged = [c for c in cases if c["true_severity"] >= damaged_threshold]
    healthy = [c for c in cases if c["true_severity"] == 0.0]
    loc_err = np.array([c["location_error"] for c in damaged]) * 100
    sev_err = np.array([c["severity_error"] for c in damaged]) * 100
    return {
        "n_damaged": len(damaged), "n_healthy": len(healthy),
        "location_error_median_pct": float(np.median(loc_err)),
        "location_error_p90_pct": float(np.percentile(loc_err, 90)),
        "severity_error_median_pts": float(np.median(sev_err)),
        "severity_error_p90_pts": float(np.percentile(sev_err, 90)),
        "location_interval_coverage": float(np.mean([c["location_in_interval"] for c in damaged])),
        "severity_interval_coverage": float(np.mean([c["severity_in_interval"] for c in damaged])),
        "detection_rate": float(np.mean([c["probability_damaged"] > 0.5 for c in damaged])),
        "false_alarm_rate": float(np.mean([c["probability_damaged"] > 0.5 for c in healthy])) if healthy else None,
        "seconds_per_detection": float(np.mean([c["seconds"] for c in cases])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-damaged", type=int, default=200)
    parser.add_argument("--n-healthy", type=int, default=50)
    parser.add_argument("--model", type=Path, default=Path("models/surrogate_hybrid.pt"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("data/processed/damage_study.json"))
    args = parser.parse_args()

    forward = SurrogateForward(args.model)
    rng = np.random.default_rng(args.seed)

    # Random beams inside the training ranges (a different seed from the training data).
    beams = sample_parameters(args.n_damaged + args.n_healthy, seed=args.seed)[:, :5]
    locations = rng.uniform(0.05, 0.95, args.n_damaged + args.n_healthy)
    severities = np.concatenate([rng.uniform(0.05, 0.5, args.n_damaged), np.zeros(args.n_healthy)])

    report, all_cases = {}, {}
    for level, (fn, sn) in NOISE_LEVELS.items():
        start = time.perf_counter()
        cases = [run_case(Beam(*beams[k]), locations[k], severities[k], forward, fn, sn, rng)
                 for k in range(len(beams))]
        report[level] = summarise(cases)
        all_cases[level] = cases
        print(f"{level:>6} noise done in {time.perf_counter() - start:.0f} s")

    print(f"\n{'':40s}" + "".join(f"{lvl:>10s}" for lvl in NOISE_LEVELS))
    print(f"{'Frequency noise (%)':40s}" + "".join(f"{v[0] * 100:>10.1f}" for v in NOISE_LEVELS.values()))
    print(f"{'Mode-shape noise':40s}" + "".join(f"{v[1]:>10.2f}" for v in NOISE_LEVELS.values()))
    rows = [
        ("Location error, median (% of length)", "location_error_median_pct", "{:.1f}"),
        ("Location error, 90th pct (% of length)", "location_error_p90_pct", "{:.1f}"),
        ("Severity error, median (% points)", "severity_error_median_pts", "{:.1f}"),
        ("Severity error, 90th pct (% points)", "severity_error_p90_pts", "{:.1f}"),
        ("Truth inside 90 % location range", "location_interval_coverage", "{:.0%}"),
        ("Truth inside 90 % severity range", "severity_interval_coverage", "{:.0%}"),
        ("Damaged beams flagged (P > 0.5)", "detection_rate", "{:.0%}"),
        ("Healthy beams flagged (false alarms)", "false_alarm_rate", "{:.0%}"),
        ("Time per detection (s)", "seconds_per_detection", "{:.3f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:40s}" + "".join(f"{fmt.format(report[lvl][key]):>10s}" for lvl in NOISE_LEVELS))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"noise_levels": NOISE_LEVELS, "summary": report, "cases": all_cases}, indent=1))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
