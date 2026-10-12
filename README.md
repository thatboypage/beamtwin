# BeamTwin

**An FEM-based digital twin for vibration-based damage detection in beam structures**

BeamTwin combines a finite element model of a beam, written from scratch, with
machine learning. The finite element model predicts how the beam bends and
vibrates. A neural surrogate learns to give the same answers in milliseconds,
and a damage detector works backwards from vibration readings to locate and size
stiffness loss.

> Status: work in progress. Step 1 (static FEM solver) is complete and validated.

## Roadmap

| Step | What it does | Status |
|---|---|---|
| 1. FEM solver | Euler-Bernoulli beam elements, assembly, cantilever solution | Done |
| 2. Modal analysis | Natural frequencies and mode shapes | Planned |
| 3. Neural surrogate | Fast network trained on FEM results | Planned |
| 4. Damage detection | Locate and size stiffness loss from vibration data | Planned |
| 5. Deployment | FastAPI back end, web app, Docker | Planned |
| 6. Real-data validation | Test on measured data (Los Alamos frame) | Planned |

## Validation so far

For a steel cantilever (1 m long, 50 x 10 mm section, 10 N tip load):

| Quantity | FEM | Textbook | Formula |
|---|---|---|---|
| Tip deflection | 4.000 mm | 4.000 mm | PL³ / 3EI |
| Tip rotation | 0.006 rad | 0.006 rad | PL² / 2EI |

The FEM result matches the textbook to rounding error for any number of
elements, because the cubic shape functions contain the exact solution for a
tip load. See [docs/theory.md](docs/theory.md).

## Project layout

```
src/beamtwin/
    fem/          finite element core (Steps 1-2)
    surrogate/    neural surrogate (Step 3)
    damage/       damage simulation and detection (Step 4)
    validation/   comparison with measured data
api/              FastAPI back end (Step 5)
app/              web front end (Step 5)
tests/            automated checks, run on every commit
notebooks/        worked examples with plots
docs/             theory and figures
```

## Getting started

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

Example:

```python
from beamtwin.fem import solve_cantilever

u = solve_cantilever(E=200e9, I=4.1667e-9, length=1.0, n_elements=8, tip_load=-10.0)
print("Tip deflection (m):", u[-2])
```

## Limitations

- Linear, small-deflection Euler-Bernoulli beam theory (no shear deformation).
- Uniform beams only, so far.
- Damage detection will first be trained on simulated data. Validation on
  measured data is a planned later stage.

## License

MIT
