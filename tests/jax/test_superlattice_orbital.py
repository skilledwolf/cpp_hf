"""Orbital streaming exchange: direct sums, variational response and solver use."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hartreefock import HartreeFockKernel, SCFConfig
from hartreefock.backends.jax import solve_scf
from hartreefock.backends.jax.fock import hf_energy
from hartreefock.backends.jax.superlattice import (
    build_extended_layout, make_superlattice_build_fock_fn,
    make_superlattice_fock_fn,
)


@pytest.fixture(autouse=True)
def double_precision(enable_x64):
    with enable_x64():
        yield


def problem():
    g = np.array([[0, 0], [1, 0], [0, -1]])
    B = np.array([[1.0, .35], [.1, .8]])
    nkx, nky, no, weight = 2, 3, 2, 1 / 6
    C = np.array([[1., .7], [.7, 1.3]])
    distance = np.array([[0., .4], [.4, 0.]])

    def scalar(q):
        return .15 / np.sqrt(np.asarray(q)**2 + .7**2)

    def orbital(q):
        q = np.asarray(q)
        return scalar(q)[..., None, None] * C * np.exp(-q[..., None, None] * distance)

    layout = build_extended_layout(g, B, nkx, nky, scalar, weight,
                                   coulomb_V_orbital=orbital)
    return g, B, nkx, nky, no, weight, scalar, orbital, layout


def hermitian(rng, shape):
    A = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return .5 * (A + A.conj().swapaxes(-1, -2))


def direct_exchange(rho, g, B, no, weight, interaction):
    """Independent momentum sum using p=k-G; no FFT or layout indices."""
    nx, ny = rho.shape[:2]
    ng = len(g)
    blocks = rho.reshape(nx, ny, ng, no, ng, no)
    result = np.zeros_like(blocks)
    for kx, ky, qx, qy in np.ndindex(nx, ny, nx, ny):
        dk = np.array([(kx - qx) / nx, (ky - qy) / ny])
        for i, j, l, m in np.ndindex(ng, ng, ng, ng):
            if np.array_equal(g[i] - g[j], g[l] - g[m]):
                momentum = B @ (dk + g[l] - g[i])
                result[kx, ky, i, :, j, :] -= (
                    weight * interaction(np.linalg.norm(momentum)) * blocks[qx, qy, l, :, m, :]
                )
    return result.reshape(rho.shape)


def test_orbital_exchange_matches_independent_direct_sum_and_cpp():
    g, B, nx, ny, no, w, _, orbital, layout = problem()
    rng = np.random.default_rng(218)
    rho = hermitian(rng, (nx, ny, len(g) * no, len(g) * no))
    fn = make_superlattice_fock_fn(layout, len(g), no, nx, ny)
    result = np.asarray(fn(jnp.asarray(rho)))
    expected = direct_exchange(rho, g, B, no, w, orbital)
    np.testing.assert_allclose(result, expected, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(result, result.conj().swapaxes(-1, -2), atol=2e-14)
    cpp = pytest.importorskip("hartreefock.backends.native.superlattice")
    cpp_result = cpp.superlattice_fock(
        rho, layout, len(g), no, nx, ny, V_lag_fft_orbital=layout.V_lag_fft_orbital,
    )
    np.testing.assert_allclose(result, cpp_result, atol=2e-14, rtol=2e-13)


def test_orbital_blind_tensor_reduces_to_scalar_including_real_inputs():
    g, _, nx, ny, no, _, _, _, layout = problem()
    scalar_layout = replace(layout, V_lag_fft_orbital=None)
    scalar_fn = make_superlattice_fock_fn(scalar_layout, len(g), no, nx, ny)
    orbital_fn = make_superlattice_fock_fn(
        scalar_layout, len(g), no, nx, ny,
        V_lag_fft_orbital=np.broadcast_to(layout.V_lag_fft[..., None, None],
                                         layout.V_lag_fft.shape + (no, no)),
    )
    rng = np.random.default_rng(119)
    rho = hermitian(rng, (nx, ny, len(g)*no, len(g)*no))
    np.testing.assert_allclose(orbital_fn(rho), scalar_fn(rho), atol=2e-14)
    np.testing.assert_allclose(orbital_fn(rho.real), scalar_fn(rho.real.astype(complex)), atol=2e-14)


def test_orbital_response_is_self_adjoint_and_jvp_is_the_exact_linear_map():
    g, _, nx, ny, no, w, _, _, layout = problem()
    fn = make_superlattice_fock_fn(layout, len(g), no, nx, ny)
    rng = np.random.default_rng(71)
    shape = (nx, ny, len(g)*no, len(g)*no)
    P, X, Y = [jnp.asarray(hermitian(rng, shape)) for _ in range(3)]
    _, response = jax.jvp(fn, (P,), (Y,))
    np.testing.assert_allclose(response, fn(Y), atol=2e-14)
    _, zero_response = jax.jvp(fn, (jnp.zeros_like(P),), (Y,))
    np.testing.assert_allclose(zero_response, fn(Y), atol=2e-14)
    inner = lambda A, B: w * jnp.vdot(A, B).real
    np.testing.assert_allclose(inner(X, fn(Y)), inner(Y, fn(X)), atol=2e-13)
    E = lambda A: .5 * inner(A, fn(A))
    step = 1e-5
    derivative = (E(P + step*Y) - E(P - step*Y)) / (2*step)
    np.testing.assert_allclose(derivative, inner(Y, fn(P)), atol=2e-10)


def test_custom_fock_adapter_keeps_orbital_exchange_hartree_and_scf():
    g, B, nx, ny, no, w, _, orbital, layout = problem()
    ng, D = len(g), len(g)*no
    rng = np.random.default_rng(55)
    h = hermitian(rng, (nx, ny, D, D))
    ref = np.broadcast_to(.3*np.eye(D), h.shape).astype(complex).copy()
    weights = np.full((nx, ny), w)
    HH_orb = orbital(np.linalg.norm((g[:, None] - g[None, :]) @ B.T, axis=-1))
    with pytest.warns(UserWarning, match="q=0"):
        builder = make_superlattice_build_fock_fn(
            layout, ng, no, nx, ny, np.zeros((ng, ng)), 1., HH_GG_orbital=HH_orb,
        )
    kernel = HartreeFockKernel(weights, h, np.zeros((nx, ny, 1, 1)), T=.7,
                              include_hartree=True, reference_density=ref,
                              hartree_matrix=np.zeros((D, D)))
    args = kernel.as_args()
    kw = {key: args[key] for key in ("h", "VR", "refP", "HH", "include_exchange",
          "include_hartree", "exchange_hermitian_channel_packing", "contact_g", "contact_Oi", "contact_Oj")}
    kw["w2d"] = kernel.w2d
    P = jnp.asarray(ref + .01*hermitian(rng, h.shape))
    Y = jnp.asarray(hermitian(rng, h.shape))
    Sigma, H, F = builder(P, **kw)
    np.testing.assert_allclose(Sigma, Sigma.conj().swapaxes(-1, -2), atol=2e-14)
    E = lambda A: hf_energy(A, h=kernel.h, Sigma=builder(A, **kw)[0], H=H,
                            weights_b=kernel.weights_b, refP=kernel.refP)
    step = 1e-5
    np.testing.assert_allclose((E(P+step*Y)-E(P-step*Y))/(2*step),
                               w*jnp.vdot(Y, F).real, atol=3e-10)
    Fmap = lambda A: builder(A, **kw)[2]
    _, LY = jax.jvp(Fmap, (P,), (Y,))
    np.testing.assert_allclose(LY, Fmap(P+Y)-Fmap(P), atol=3e-14)
    result = solve_scf(kernel, ref, 3., config=SCFConfig(
        acceleration="rdiis", diis_size=8, max_iter=100,
        density_tol=1e-8, comm_tol=1e-8), fock_build_fn=builder)
    assert result.converged, result.message
    np.testing.assert_allclose(result.fock, Fmap(result.density), atol=2e-13)
    np.testing.assert_allclose(result.energy, E(result.density), atol=2e-13)


def test_orbital_kernel_shape_and_finiteness_validation():
    g, B, nx, ny, no, w, scalar, _, layout = problem()
    with pytest.raises(ValueError, match="coulomb_V_orbital.*shape"):
        build_extended_layout(g, B, nx, ny, scalar, w, coulomb_V_orbital=lambda q: q)
    with pytest.raises(ValueError, match="finite"):
        build_extended_layout(g, B, nx, ny, scalar, w,
                              coulomb_V_orbital=lambda q: np.full(q.shape+(no,no), np.nan))
    with pytest.raises(ValueError, match="V_lag_fft_orbital.*shape"):
        make_superlattice_fock_fn(layout, len(g), no+1, nx, ny)
    fn = make_superlattice_fock_fn(layout, len(g), no, nx, ny)
    with pytest.raises(ValueError, match="density must have shape"):
        fn(jnp.zeros((nx, ny, 1, 1)))
