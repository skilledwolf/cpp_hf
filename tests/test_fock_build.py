"""The Fock build matches the kernel's linear maps and is the energy gradient."""

from __future__ import annotations

import numpy as np
import pytest
from _meshes import mesh_for

import hartreefock as hf
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

    F, _ = hf.fock_operators(kernel).evaluate(P)
    np.testing.assert_allclose(F, kernel.h + kernel.potential(P - kernel.refP), atol=1e-12)


def test_fock_is_energy_gradient(x64):
    rng = np.random.default_rng(3)
    kernel = _kernel(rng, orbital_V=False, embed=False)
    P = _herm(rng, kernel.h.shape)
    dP = _herm(rng, kernel.h.shape)

    ops = hf.fock_operators(kernel)
    eps = 1e-6
    E_plus = ops.evaluate(P + eps * dP)[1]
    E_minus = ops.evaluate(P - eps * dP)[1]
    F = ops.evaluate(P)[0]
    directional = np.sum(np.real(np.einsum("...ij,...ji->...", kernel.weights_b * F, dP)))

    np.testing.assert_allclose((E_plus - E_minus) / (2 * eps), directional, rtol=1e-7)


def test_oda_moves_a_start_with_the_wrong_particle_count_onto_it(x64):
    """ODA must not keep an infeasible start whose free energy is lower."""
    nk = 4
    ks = (np.arange(nk) - nk // 2) * (2 * np.pi / nk)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    h = np.zeros((nk, nk, 2, 2), complex)
    h[..., 0, 0], h[..., 1, 1] = 0.3, -0.3
    h[..., 0, 1], h[..., 1, 0] = KX - 1j * KY, KX + 1j * KY
    kernel = HartreeFockKernel(mesh_for(np.full((nk, nk), 1.0 / nk**2)), h, np.full((nk, nk, 1, 1), 0.7), T=0.1)
    n_e = 0.25
    # The filled lower band (N = 1): its free energy is below every density with N = 1/4.
    _, U = np.linalg.eigh(h)
    P0 = U[..., :, :1] @ np.conj(U[..., :, :1]).swapaxes(-1, -2)
    common = dict(max_iter=400, density_tol=1e-8, comm_tol=1e-7, mixing=0.5)
    oda = hf.solve_scf(kernel, P0, n_e, config=hf.SCFConfig(acceleration="oda", **common))
    diis = hf.solve_scf(kernel, P0, n_e, config=hf.SCFConfig(acceleration="rdiis", **common))
    w = np.full((nk, nk), 1.0 / nk**2)
    count = np.sum(w * np.real(np.trace(np.asarray(oda.density), axis1=-2, axis2=-1)))
    np.testing.assert_allclose(count, n_e, atol=1e-8)
    np.testing.assert_allclose(float(oda.energy), float(diis.energy), atol=1e-6)
