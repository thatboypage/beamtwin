"""Shared test settings: the steel strip used throughout Step 1."""

import pytest


@pytest.fixture
def steel_strip():
    """A steel strip 1 m long, 50 mm wide and 10 mm thick."""
    b, h = 0.05, 0.01
    return {"E": 200e9, "I": b * h**3 / 12, "length": 1.0}
