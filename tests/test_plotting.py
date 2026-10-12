"""Checks on the shared plotting helpers."""

from beamtwin.plotting import project_root


def test_project_root_finds_pyproject():
    assert (project_root() / "pyproject.toml").exists()
