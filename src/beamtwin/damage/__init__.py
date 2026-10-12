"""Damage simulation and detection from vibration data (Steps 3-4)."""

from beamtwin.damage.detect import DetectionResult, Measurement, detect, simulate_measurement
from beamtwin.damage.forward import Beam, FEMForward, SurrogateForward
from beamtwin.damage.simulate import damage_profile

__all__ = [
    "damage_profile", "Beam", "FEMForward", "SurrogateForward",
    "Measurement", "simulate_measurement", "detect", "DetectionResult",
]
