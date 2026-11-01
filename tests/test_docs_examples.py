"""Run every script that the documentation pages include."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "docs" / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_docs_example_runs(script, tmp_path, monkeypatch):
    pytest.importorskip("hartreefock._backends.native._native")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")
