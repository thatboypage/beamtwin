"""BeamTwin web API (Step 5a).

A thin web layer over beamtwin.service: it checks incoming requests, calls the
service functions and returns JSON. It also serves the web page in app/static.

Run locally from the project root:

    uvicorn api.main:app --reload

then open http://127.0.0.1:8000 for the app, or http://127.0.0.1:8000/docs for
interactive API documentation that FastAPI generates automatically.
"""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from beamtwin import __version__, service

STATIC_DIR = Path(__file__).resolve().parents[1] / "app" / "static"

app = FastAPI(
    title="BeamTwin API",
    version=__version__,
    description="Vibration analysis and damage detection for cantilever beams: "
                "FEM, a physics-informed neural surrogate and Bayesian damage localisation.",
)

Noise = Literal["low", "medium", "high"]


# ---------- What requests must look like (FastAPI checks these automatically) ----------

class BeamIn(BaseModel):
    length: float = Field(gt=0, le=20, description="Length in metres", examples=[1.0])
    width: float = Field(gt=0, le=1, description="Width in metres", examples=[0.05])
    thickness: float = Field(gt=0, le=1, description="Thickness in metres", examples=[0.01])
    E: float = Field(gt=0, description="Young's modulus in pascals", examples=[200e9])
    rho: float = Field(gt=0, description="Density in kg/m^3", examples=[7850])

    def to_beam(self):
        return service.make_beam(self.length, self.width, self.thickness, self.E, self.rho)


class SimulateIn(BaseModel):
    beam: BeamIn
    location: float = Field(ge=0.05, le=0.95, description="Damage centre, fraction of length from the clamp")
    severity: float = Field(ge=0.0, le=0.5, description="Stiffness lost at the damage centre (0-0.5)")
    noise: Noise = "medium"
    seed: int | None = None


class DetectIn(BaseModel):
    beam: BeamIn
    frequencies: list[float] = Field(min_length=4, max_length=4, description="Measured f1-f4 in Hz")
    shapes: list[float] | None = Field(default=None, min_length=36, max_length=36,
                                       description="Optional: modes 1-4 at the 9 sensors, tip-normalised")
    noise: Noise = "medium"


# ---------- Endpoints ----------

@app.get("/api/health")
def health():
    """Quick check that the server is up and the model loads."""
    service.get_forward()
    return {"status": "ok", "version": __version__}


@app.get("/api/options")
def options():
    """Materials, noise presets and the surrogate's valid ranges, for building the form."""
    return {"materials": service.MATERIALS, "noise_levels": service.NOISE_LEVELS}


@app.post("/api/analyze")
def analyze(body: BeamIn):
    """Natural frequencies and mode shapes of the healthy beam."""
    return service.analyze(body.to_beam())


@app.post("/api/simulate")
def simulate(body: SimulateIn):
    """A noisy measurement of the beam with the given damage (for trying the detector)."""
    return service.simulate(body.beam.to_beam(), body.location, body.severity, body.noise, body.seed)


@app.post("/api/detect")
def detect(body: DetectIn):
    """Locate and size damage from measured frequencies (and optionally mode shapes)."""
    try:
        return service.detect_damage(body.beam.to_beam(), body.frequencies, body.shapes, body.noise)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err


# The web page. Mounted last so the /api routes above take priority.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="app")
