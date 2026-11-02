from __future__ import annotations

import contextlib
import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture
def x64():
    """Run the test with ``jax_enable_x64`` on."""
    import jax

    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def enable_x64():
    """Context-manager factory that sets ``jax_enable_x64`` and restores it.

    Replaces ``jax.experimental.enable_x64``, removed in jax 0.12.
    """
    import jax

    @contextlib.contextmanager
    def _scope(flag: bool = True):
        previous = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", bool(flag))
        try:
            yield
        finally:
            jax.config.update("jax_enable_x64", previous)

    return _scope
