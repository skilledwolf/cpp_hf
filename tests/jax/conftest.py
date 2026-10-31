from __future__ import annotations

import contextlib
import os

import pytest

# Force CPU backend for stability in test environments.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


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
