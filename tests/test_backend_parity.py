"""The native and JAX backends evaluate the same Fock matrix and energy."""

from __future__ import annotations

import numpy as np
import pytest

from hartreefock.kernel import HartreeFockKernel

native_fock = pytest.importorskip("hartreefock._backends.native.fock")
jax = pytest.importorskip("jax")


def _herm(rng, shape):
    X = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))


def _kernel(rng, *, uniform, orbital_V, embed, nk=5, nb=3):
    h = _herm(rng, (nk, nk, nb, nb))
    w = np.full((nk, nk), 1.0 / nk**2) if uniform else rng.uniform(0.2, 1.0, (nk, nk))
    V_shape = (nk, nk, nb, nb) if orbital_V else (nk, nk, 1, 1)
    V = rng.uniform(0.5, 1.5, V_shape)
    O = np.diag(rng.normal(size=nb)).astype(complex)
    return HartreeFockKernel(
        w, h, V, T=0.1,
        include_hartree=True,
        reference_density=0.3 * _herm(rng, h.shape),
        hartree_matrix=0.2 + 0.1 * np.eye(nb),
        contact_terms=[(0.4, O, O)],
        embed_reference_potential=embed,
    )


def _jax_fock_and_energy(kernel, P):
    from hartreefock._backends.jax.fock import build_fock, hf_energy

    a = kernel._as_args()
    sigma, H, F = build_fock(
        P, h=a["h"], VR=a["VR"], refP=a["refP"], HH=a["HH"], w2d=kernel.w2d,
        include_exchange=a["include_exchange"], include_hartree=a["include_hartree"],
        exchange_hermitian_channel_packing=a["exchange_hermitian_channel_packing"],
        contact_g=a["contact_g"], contact_Oi=a["contact_Oi"], contact_Oj=a["contact_Oj"],
    )
    E = hf_energy(P, h=a["h"], Sigma=sigma, H=H, weights_b=a["weights_b"], refP=a["refP"])
    return np.asarray(F), float(E)


@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("orbital_V", [False, True])
@pytest.mark.parametrize("embed", [False, True])
def test_native_and_jax_fock_agree(uniform, orbital_V, embed, x64):
    rng = np.random.default_rng(7)
    kernel = _kernel(rng, uniform=uniform, orbital_V=orbital_V, embed=embed)
    P = _herm(rng, kernel.h.shape)

    _sigma, _H, F_native, E_native = native_fock.build_fock(kernel, P)
    F_jax, E_jax = _jax_fock_and_energy(kernel, P)

    np.testing.assert_allclose(F_native, F_jax, atol=1e-12)
    np.testing.assert_allclose(E_native, E_jax, rtol=1e-12)
    # The kernel's NumPy linear maps reproduce the backends' Fock matrix.
    np.testing.assert_allclose(F_native, kernel.h + kernel.potential(P - kernel.refP), atol=1e-12)


def test_native_fock_is_energy_gradient():
    rng = np.random.default_rng(3)
    kernel = _kernel(rng, uniform=False, orbital_V=False, embed=False)
    P = _herm(rng, kernel.h.shape)
    dP = _herm(rng, kernel.h.shape)

    eps = 1e-6
    E_plus = native_fock.build_fock(kernel, P + eps * dP)[3]
    E_minus = native_fock.build_fock(kernel, P - eps * dP)[3]
    F = native_fock.build_fock(kernel, P)[2]
    directional = np.sum(np.real(np.einsum("...ij,...ji->...", kernel.weights_b * F, dP)))

    np.testing.assert_allclose((E_plus - E_minus) / (2 * eps), directional, rtol=1e-7)


def test_native_project_callback_runs_after_gil_release():
    kernel = HartreeFockKernel(
        np.ones((1, 1)), np.eye(2, dtype=complex)[None, None], np.ones((1, 1, 1, 1)), T=0.1
    )
    F = native_fock.build_fock(kernel, np.zeros_like(kernel.h), project_fn=np.zeros_like)[2]
    np.testing.assert_array_equal(F, 0.0)


def test_native_only_options_raise_on_jax():
    pytest.importorskip("jax")
    import hartreefock as hf

    kernel = HartreeFockKernel(
        np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex), np.ones((1, 1, 1, 1)), T=0.1
    )
    P0 = np.eye(2, dtype=complex)[None, None] / 2
    with pytest.raises(NotImplementedError, match="native"):
        hf.solve(kernel, P0, 1.0, config=hf.SolverConfig(optimizer="auto"), backend="jax")
    config = hf.SCFConfig(acceleration="rdiis", preconditioner="hartree")
    with pytest.raises(NotImplementedError, match="native"):
        hf.solve_scf(kernel, P0, 1.0, config=config, backend="jax")
