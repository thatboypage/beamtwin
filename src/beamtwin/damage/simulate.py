"""Describe damage as a local loss of bending stiffness (used by Steps 3 and 4)."""

import numpy as np


def damage_profile(n_elements: int, location: float, severity: float, width: float = 0.04) -> np.ndarray:
    """Return one stiffness factor per element for a single smooth damage zone.

    The damage is a dip in stiffness shaped like a bell curve:

        factor(x) = 1 - severity * exp(-0.5 * ((x - location) / width)^2)

    Positions are fractions of the beam length (0 = clamp, 1 = tip), so the
    same damage description works for a beam of any length.

    n_elements : number of elements in the FEM mesh
    location   : centre of the damage, 0..1 along the beam
    severity   : stiffness lost at the centre, 0 (healthy) to <1
                 (0.3 means the worst point keeps 70% of its stiffness)
    width      : how spread out the damage is, as a fraction of the length

    Why a bell curve instead of weakening one element? A real crack affects the
    stiffness around it, and a smooth profile means a small move of the damage
    gives a small change in the result. Neural networks learn smooth
    relationships much more easily than jumpy ones.
    """
    if not 0.0 <= severity < 1.0:
        raise ValueError("severity must be in [0, 1)")
    x_mid = (np.arange(n_elements) + 0.5) / n_elements      # element centres, 0..1
    return 1.0 - severity * np.exp(-0.5 * ((x_mid - location) / width) ** 2)
