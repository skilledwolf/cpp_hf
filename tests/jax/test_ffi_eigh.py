"""The native batched eigh the JAX backend calls through jax.ffi."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hartreefock._backends.jax import linalg


def _hermitian(shape, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return jnp.asarray(a + np.conj(np.swapaxes(a, -1, -2)))


@pytest.mark.skipif(not linalg._ffi_eigh_ready(), reason="built without the XLA FFI headers")
def test_ffi_eigh_matches_xla_under_jit_and_vmap():
    h = _hermitian((3, 4, 6, 6))
    w_ref, v_ref = jnp.linalg.eigh(h)
    w, v = jax.jit(linalg.hermitian_eigh)(h)
    np.testing.assert_allclose(w, w_ref, atol=1e-12)
    # Eigenvectors agree up to a phase: compare the projectors.
    proj = jnp.einsum("...in,...jn->...nij", v, v.conj())
    proj_ref = jnp.einsum("...in,...jn->...nij", v_ref, v_ref.conj())
    np.testing.assert_allclose(proj, proj_ref, atol=1e-10)
    wv, _ = jax.vmap(linalg.hermitian_eigh)(h)
    np.testing.assert_allclose(wv, w_ref, atol=1e-12)


def test_other_dtypes_and_single_matrices_use_xla():
    h = _hermitian((5, 5)).real
    w, _ = linalg.hermitian_eigh(h)
    np.testing.assert_allclose(w, jnp.linalg.eigh(h)[0])
