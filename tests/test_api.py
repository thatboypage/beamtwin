"""Checks on the web API plumbing (Step 5a). Skipped if FastAPI is not installed.

The science is tested in test_service.py; here a simple stand-in replaces the
surrogate so the tests are fast and do not need a trained model.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from beamtwin import service  # noqa: E402

BEAM = {"length": 1.0, "width": 0.05, "thickness": 0.01, "E": 200e9, "rho": 7850}


def fake_forward(beam, locations, severities):
    """Damage lowers every frequency by a tenth of its severity; mode shapes unchanged."""
    severities = np.asarray(severities, float)
    return 1 - 0.1 * severities[:, None] * np.ones(4), np.zeros((len(severities), 36))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "get_forward", lambda *args, **kwargs: fake_forward)
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_options_list_materials_and_noise(client):
    body = client.get("/api/options").json()
    assert "steel" in body["materials"] and "medium" in body["noise_levels"]


def test_analyze_returns_four_frequencies(client):
    body = client.post("/api/analyze", json=BEAM).json()
    assert len(body["frequencies_hz"]) == 4 and body["warnings"] == []


def test_simulate_then_detect(client):
    sim = client.post("/api/simulate", json={"beam": BEAM, "location": 0.4, "severity": 0.2, "seed": 1}).json()
    r = client.post("/api/detect", json={"beam": BEAM, "frequencies": sim["frequencies_hz"]})
    assert r.status_code == 200
    body = r.json()
    for key in ["location", "severity", "location_interval", "probability_damaged", "posterior", "summary"]:
        assert key in body


def test_bad_requests_are_rejected(client):
    assert client.post("/api/detect", json={"beam": BEAM, "frequencies": [1, 2, 3]}).status_code == 422
    assert client.post("/api/simulate", json={"beam": BEAM, "location": 2.0, "severity": 0.2}).status_code == 422
    bad_beam = {**BEAM, "length": -1}
    assert client.post("/api/analyze", json=bad_beam).status_code == 422


def test_web_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "BeamTwin" in r.text
    for asset in ["/style.css", "/app.js"]:
        assert client.get(asset).status_code == 200
