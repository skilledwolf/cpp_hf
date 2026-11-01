"""The dense kernel's exchange runs on NumPy input and traces under JAX."""

from __future__ import annotations

import numpy as np
import pytest

import hartreefock as hf


def _kernel(nk=6, nb=3, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(nk, nk, nb, nb)) + 1j * rng.normal(size=(nk, nk, nb, nb))
    h = h + np.conj(np.swapaxes(h, -1, -2))
    V = rng.uniform(0.5, 2.0, size=(nk, nk, nb, nb))
    weights = rng.uniform(0.5, 1.5, size=(nk, nk))
    g = 0.7
    Oi = np.diag(rng.normal(size=nb)).astype(complex)
    return hf.HartreeFockKernel(weights, h, V, T=0.1, contact_terms=[(g, Oi, Oi)])


def test_exchange_traces_under_jax_and_matches_numpy():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    kernel = _kernel()
    rng = np.random.default_rng(1)
    x = rng.normal(size=kernel.h.shape) + 1j * rng.normal(size=kernel.h.shape)

    ref = kernel.exchange(x)
    traced = np.asarray(jax.jit(kernel.exchange)(jnp.asarray(x)))
    np.testing.assert_allclose(traced, ref, atol=1e-12, rtol=1e-12)

