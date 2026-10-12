"""Train and compare the plain and hybrid surrogates (Step 3b).

Run from the project root, after generating the dataset (Step 3a):

    python -m beamtwin.surrogate.train

This trains both designs on the same data split, prints their test scores,
and saves each model to models/ (not tracked by Git).
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from beamtwin.surrogate.evaluation import evaluate, format_report, split_indices
from beamtwin.surrogate.features import KINDS, encode_inputs, encode_targets
from beamtwin.surrogate.model import MLP, Standardizer, Surrogate


def train_surrogate(X_train, Y_train, X_val, Y_val, kind: str,
                    hidden=(64, 64, 64), max_epochs: int = 3000, batch_size: int = 128,
                    lr: float = 1e-3, patience: int = 200, seed: int = 0, log_every: int = 0):
    """Train one surrogate. Returns (Surrogate, history).

    Training loop, in words:
      1. Show the network a small batch of beams.
      2. Measure how wrong its answers are (mean squared error = the "loss").
      3. Nudge every weight slightly in the direction that reduces the loss (Adam).
      4. Repeat over all batches = one "epoch".
      5. After each epoch, score the validation set. Keep the best version seen.
         Stop when validation has not improved for `patience` epochs (early stopping).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Scalers are fitted on training data only, so no information leaks from validation/test.
    x_scaler = Standardizer.fit(encode_inputs(X_train, kind))
    t_scaler = Standardizer.fit(encode_targets(X_train, Y_train, kind))

    def to_tensors(X, Y):
        x = x_scaler.transform(encode_inputs(X, kind))
        t = t_scaler.transform(encode_targets(X, Y, kind))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)

    x_tr, t_tr = to_tensors(X_train, Y_train)
    x_va, t_va = to_tensors(X_val, Y_val)

    model = MLP(x_tr.shape[1], t_tr.shape[1], tuple(hidden))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Halve the step size whenever validation stalls for a while: big steps early, fine steps later.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience // 4)
    loss_fn = nn.MSELoss()

    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    n = len(x_tr)

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(n))
        running = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[idx]), t_tr[idx])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_va), t_va).item()
        scheduler.step(val_loss)

        history["train_loss"].append(running / n)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val * (1 - 1e-4):
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:5d}  train {running / n:.2e}  val {val_loss:.2e}  "
                  f"lr {optimizer.param_groups[0]['lr']:.1e}")
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["epochs_run"] = epoch
    return Surrogate(kind, model, x_scaler, t_scaler, tuple(hidden)), history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/processed/beam_dataset.npz"))
    parser.add_argument("--kind", choices=[*KINDS, "both"], default="both")
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    data = np.load(args.data)
    X, Y = data["X"], data["Y"]
    tr, va, te = split_indices(len(X), seed=args.seed)
    print(f"Data: {len(X)} beams  ->  train {len(tr)}, validation {len(va)}, test {len(te)}")

    kinds = KINDS if args.kind == "both" else (args.kind,)
    summary = {}
    for kind in kinds:
        print(f"\nTraining the {kind} surrogate...")
        start = time.perf_counter()
        surrogate, history = train_surrogate(X[tr], Y[tr], X[va], Y[va], kind,
                                             max_epochs=args.max_epochs, seed=args.seed, log_every=250)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        Y_pred = surrogate.predict(X[te])
        predict_ms = (time.perf_counter() - start) / len(te) * 1000

        metrics = evaluate(X[te], Y[te], Y_pred)
        print(f"  stopped after {history['epochs_run']} epochs (best at {history['best_epoch']}), "
              f"{train_seconds:.0f} s; prediction {predict_ms:.4f} ms per beam")
        print(format_report(f"{kind.upper()} surrogate on the test set", metrics))

        surrogate.save(args.out_dir / f"surrogate_{kind}.pt")
        (args.out_dir / f"history_{kind}.json").write_text(json.dumps(history))
        summary[kind] = {k: np.round(np.atleast_1d(v), 5).tolist() for k, v in metrics.items()}
        summary[kind].update(train_seconds=round(train_seconds, 1), predict_ms_per_beam=predict_ms,
                             epochs_run=history["epochs_run"], best_epoch=history["best_epoch"])

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved models and summary to {args.out_dir}/")


if __name__ == "__main__":
    main()
