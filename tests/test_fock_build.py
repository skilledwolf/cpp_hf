"""The Fock build matches the kernel's linear maps and is the energy gradient."""

from __future__ import annotations

import numpy as np
import pytest
from _meshes import mesh_for

from hartreefock._jax import fock_operators
from hartreefock.kernel import HartreeFockKernel


def _herm(rng, shape):
    X = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))


def _kernel(rng, *, orbital_V, embed, nk=5, nb=3):
    h = _herm(rng, (nk, nk, nb, nb))
    w = np.full((nk, nk), 1.0 / nk**2)
    V_shape = (nk, nk, nb, nb) if orbital_V else (nk, nk, 1, 1)
    V = rng.uniform(0.5, 1.5, V_shape)
    O = np.diag(rng.normal(size=nb)).astype(complex)
    return HartreeFockKernel(mesh_for(w), h, V, hartree=0.2 + 0.1 * np.eye(nb), T=0.1, reference=0.3 * _herm(rng, h.shape), contact_terms=[(0.4, O, O)], embed_reference=embed)


@pytest.mark.parametrize("orbital_V", [False, True])
@pytest.mark.parametrize("embed", [False, True])
def test_fock_matches_the_kernel_linear_maps(orbital_V, embed, x64):
    rng = np.random.default_rng(7)
    kernel = _kernel(rng, orbital_V=orbital_V, embed=embed)
    P = _herm(rng, kernel.h.shape)

    F, _ = fock_operators(kernel).evaluate(P)
    np.testing.assert_allclose(F, kernel.h + kernel.potential(P - kernel.refP), atol=1e-12)


def test_fock_is_energy_gradient(x64):
    rng = np.random.default_rng(3)
    kernel = _kernel(rng, orbital_V=False, embed=False)
    P = _herm(rng, kernel.h.shape)
    dP = _herm(rng, kernel.h.shape)

    ops = fock_operators(kernel)
    eps = 1e-6
    E_plus = ops.evaluate(P + eps * dP)[1]
    E_minus = ops.evaluate(P - eps * dP)[1]
    F = ops.evaluate(P)[0]
    directional = np.sum(np.real(np.einsum("...ij,...ji->...", kernel.weights_b * F, dP)))

    np.testing.assert_allclose((E_plus - E_minus) / (2 * eps), directional, rtol=1e-7)
