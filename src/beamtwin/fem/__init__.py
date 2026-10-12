"""Finite element core: element matrices, assembly and solvers (Steps 1-2)."""

from beamtwin.fem.assembly import assemble_stiffness
from beamtwin.fem.element import element_stiffness
from beamtwin.fem.modal import (
    assemble_mass,
    cantilever_exact_frequencies,
    cantilever_modes,
    element_mass,
)
from beamtwin.fem.solver import solve_cantilever

__all__ = [
    "element_stiffness",
    "assemble_stiffness",
    "solve_cantilever",
    "element_mass",
    "assemble_mass",
    "cantilever_modes",
    "cantilever_exact_frequencies",
]
