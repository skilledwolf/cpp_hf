from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture
def x64():
    """Run the test with ``jax_enable_x64`` on (when JAX is installed)."""
    try:
        import jax
    except ImportError:
        yield
        return
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)
