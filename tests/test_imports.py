"""Every module must import on its own, in a fresh Python, whatever order it is loaded in.

This catches circular imports: two files that each need the other before either
has finished loading. They can hide for a long time, because they only fail when
the modules happen to be loaded in a particular order.
"""

import subprocess
import sys

import pytest

MODULES = [
    "beamtwin.fem",
    "beamtwin.damage",
    "beamtwin.damage.simulate",
    "beamtwin.damage.forward",
    "beamtwin.damage.detect",
    "beamtwin.damage.study",
    "beamtwin.surrogate.data_generation",
    "beamtwin.surrogate.features",
    "beamtwin.surrogate.evaluation",
    "beamtwin.surrogate.export",
    "beamtwin.service",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_on_its_own(module):
    result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
