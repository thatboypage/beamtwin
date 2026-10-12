"""Data splitting and accuracy measures for the surrogate (Step 3b). Needs only numpy."""

import numpy as np

from beamtwin.surrogate.features import N_FREQS, healthy_frequencies


def split_indices(n: int, seed: int = 0, fractions=(0.70, 0.15, 0.15)):
    """Shuffle 0..n-1 and cut into train / validation / test index arrays.

    train      : the network learns from these
    validation : checked during training to decide when to stop
    test       : locked away until the end, for an honest final score
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must add up to 1")
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def evaluate(X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Accuracy of predicted outputs against FEM, one value per mode where relevant.

    freq_error_pct      : |predicted - true| / true frequency, in %
    drop_true_pct       : how much damage lowered each frequency (the signal), in %
    drop_error_pct_pts  : error in that drop, in percentage points
    signal_to_error     : drop_true / drop_error; above about 10 the damage
                          effect is clearly visible through the surrogate's error
    shape_mae           : mean absolute error of the mode-shape sensor values
    """
    f_true, f_pred = Y_true[:, :N_FREQS], Y_pred[:, :N_FREQS]
    healthy = healthy_frequencies(X)
    rel_err = np.abs(f_pred - f_true) / f_true * 100
    drop_true = (1 - f_true / healthy) * 100
    drop_pred = (1 - f_pred / healthy) * 100
    drop_err = np.abs(drop_pred - drop_true)
    with np.errstate(divide="ignore", invalid="ignore"):     # a perfect model would divide by zero
        signal_to_error = drop_true.mean(axis=0) / drop_err.mean(axis=0)
    return {
        "freq_error_pct_mean": rel_err.mean(axis=0),
        "freq_error_pct_max": rel_err.max(axis=0),
        "drop_true_pct_mean": drop_true.mean(axis=0),
        "drop_error_pct_pts_mean": drop_err.mean(axis=0),
        "signal_to_error": signal_to_error,
        "shape_mae": float(np.abs(Y_pred[:, N_FREQS:] - Y_true[:, N_FREQS:]).mean()),
    }


def format_report(name: str, metrics: dict) -> str:
    """A small text table of the metrics for one model."""
    rows = [
        ("Frequency error, mean (%)", "freq_error_pct_mean", "{:.3f}"),
        ("Frequency error, worst (%)", "freq_error_pct_max", "{:.3f}"),
        ("Damage drop, true mean (%)", "drop_true_pct_mean", "{:.3f}"),
        ("Damage drop error (% points)", "drop_error_pct_pts_mean", "{:.3f}"),
        ("Signal-to-error ratio", "signal_to_error", "{:.1f}"),
    ]
    lines = [f"\n{name}", f"{'':32s}" + "".join(f"{'Mode ' + str(m + 1):>10s}" for m in range(N_FREQS))]
    for label, key, fmt in rows:
        lines.append(f"{label:32s}" + "".join(f"{fmt.format(v):>10s}" for v in metrics[key]))
    lines.append(f"{'Mode-shape error (mean abs)':32s}{metrics['shape_mae']:>10.5f}")
    return "\n".join(lines)
