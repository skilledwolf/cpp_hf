"""Superlattice kernel: exchange and Hartree against direct sums, the ffi kernel,
reference embedding and spectral upsampling."""

from __future__ import annotations

import numpy as np
import pytest

import hartreefock as hf
from hartreefock._jax import fock_operators
from hartreefock.superlattice import upsample_density

G = np.array([[0, 0], [1, 0], [0, -1], [-1, 1]])
B = np.array([[1.0, 0.35], [0.1, 0.8]])
NX = NY = 4
NO = 2
MESH = hf.Mesh.bz(B, NX, gamma=True)
W = MESH.weight
C = np.array([[1.0, 0.7], [0.7, 1.3]])
DIST = np.array([[0.0, 0.4], [0.4, 0.0]])


def scalar(q):
    return 0.15 / np.sqrt(np.asarray(q) ** 2 + 0.7**2)


def orbital(q):
    q = np.asarray(q)
    return scalar(q)[..., None, None] * C * np.exp(-q[..., None, None] * DIST)


def hermitian(rng, shape):
    A = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return 0.5 * (A + A.conj().swapaxes(-1, -2))


def same_delta(i, j, l, m):
    return np.array_equal(G[i] - G[j], G[l] - G[m])


def direct_exchange(rho, interaction):
    """Plain momentum sum with p = k - G; no FFT or layout indices."""
    blocks = rho.reshape(NX, NY, len(G), NO, len(G), NO)
    out = np.zeros_like(blocks)
    for kx, ky, qx, qy in np.ndindex(NX, NY, NX, NY):
        dk = np.array([(kx - qx) / NX, (ky - qy) / NY])
        for i, j, l, m in np.ndindex(*(len(G),) * 4):
            if same_delta(i, j, l, m):
                V = interaction(np.linalg.norm(B @ (dk + G[l] - G[i])))
                out[kx, ky, i, :, j, :] -= W * V * blocks[qx, qy, l, :, m, :]
    return out.reshape(rho.shape)


def direct_hartree(rho, HH, degeneracy):
    """σ[(A α), (B α)] = g Σ_γ HH[A, B, α, γ] Σ_{ΔG(A'B') = ΔG(AB)} ρ̄[(A' γ), (B' γ)]."""
    nG = len(G)
    rho_bar = (W * rho.sum(axis=(0, 1))).reshape(nG, NO, nG, NO)
    out = np.zeros((nG, NO, nG, NO), dtype=complex)
    for i, j, l, m in np.ndindex(nG, nG, nG, nG):
        if same_delta(i, j, l, m):
            for a, c in np.ndindex(NO, NO):
                coupling = HH[i, j] if HH.ndim == 2 else HH[i, j, a, c]
                out[i, a, j, a] += degeneracy * coupling * rho_bar[l, c, m, c]
    return out.reshape(nG * NO, nG * NO)


def band_hamiltonian(rng):
    """Free dispersion |B(k - G)|² per orbital plus a weak random coupling."""
    D = len(G) * NO
    kx = (np.arange(NX) / NX - 0.5)[:, None]
    ky = (np.arange(NY) / NY - 0.5)[None, :]
    h = 0.05 * hermitian(rng, (NX, NY, D, D))
    for a, g in enumerate(G):
        p = np.stack(np.broadcast_arrays(kx - g[0], ky - g[1]), axis=-1) @ B.T
        for o in range(NO):
            h[..., a * NO + o, a * NO + o] += 0.3 * np.sum(p**2, axis=-1) + 0.2 * o
    return h


def make_kernel(h, V=scalar, **kw):
    kw.setdefault("T", 0.1)
    return hf.HartreeFockKernel(MESH, h, V, plane_waves=G, **kw)


@pytest.mark.parametrize("orb", [False, True], ids=["scalar", "orbital"])
def test_exchange_and_hartree_match_direct_sums(orb):
    rng = np.random.default_rng(3)
    D = len(G) * NO
    kernel = make_kernel(
        hermitian(rng, (NX, NY, D, D)),
        orbital if orb else scalar,
        hartree=True,
        reference=np.zeros((NX, NY, D, D)),
        degeneracy=2.0,
    )
    rho = hermitian(rng, (NX, NY, D, D))
    np.testing.assert_allclose(kernel.exchange(rho), direct_exchange(rho, orbital if orb else scalar), atol=1e-13)
    # Scalar Hartree drops q = 0; orbital Hartree keeps the gate-screened q = 0 blocks.
    HH = kernel.HH
    assert (np.abs(np.diagonal(HH, axis1=0, axis2=1)).max() > 0) == orb
    np.testing.assert_allclose(kernel.hartree(rho), direct_hartree(rho, HH, 2.0), atol=1e-13)

    # The off-mesh plane-wave sum reproduces the mesh potential on mesh points.
    off_mesh = kernel.potential_at(rho, MESH.frac_grid.reshape(-1, 2)).reshape(rho.shape)
    np.testing.assert_allclose(off_mesh, kernel.potential(rho), atol=1e-13)


@pytest.mark.parametrize(
    "case",
    [dict(), dict(V=orbital), dict(embed_reference=True)],
    ids=["scalar", "orbital", "embedded"],
)
def test_fock_matches_the_kernel_linear_maps(x64, case):
    import jax
    import jax.numpy as jnp

    from hartreefock._jax.fock import fock_builder

    rng = np.random.default_rng(5)
    D = len(G) * NO
    ref = np.broadcast_to(0.5 * np.eye(D), (NX, NY, D, D))
    kernel = make_kernel(
        hermitian(rng, (NX, NY, D, D)), hartree=True, reference=ref, degeneracy=2.0, **case,
    )
    P = hermitian(rng, (NX, NY, D, D))
    F, _ = fock_operators(kernel).evaluate(P)
    np.testing.assert_allclose(F, kernel.h + kernel.potential(P - kernel.refP), atol=1e-13)

    args = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in kernel._as_args().items()}
    build = fock_builder(kernel)
    fock_args = dict(
        h=args["h"], VR=args["VR"], refP=args["refP"], HH=args["HH"], w2d=jnp.asarray(kernel.w2d),
        include_exchange=True, include_hartree=True, exchange_hermitian_channel_packing=False,
        contact_g=args["contact_g"], contact_Oi=args["contact_Oi"], contact_Oj=args["contact_Oj"],
    )
    # The builder is affine: its JVP is the self-adjoint interaction response.
    X, Y = (jnp.asarray(hermitian(rng, P.shape)) for _ in range(2))
    response = lambda dP: jax.jvp(lambda A: build(A, **fock_args)[2], (jnp.asarray(P),), (dP,))[1]
    inner = lambda A, Bm: W * jnp.vdot(A, Bm).real
    np.testing.assert_allclose(float(inner(X, response(Y))), float(inner(Y, response(X))), atol=1e-13)


def test_embedding_solves_standard_hf(x64):
    """An embedded kernel converges to the unreferenced HF state and reports its energy."""
    rng = np.random.default_rng(11)
    h = band_hamiltonian(rng)
    D = h.shape[-1]
    ref = np.broadcast_to(0.5 * np.eye(D), h.shape)
    common = dict(V=orbital, hartree=True, degeneracy=2.0)
    plain = make_kernel(h, reference=np.zeros_like(ref), **common)
    embedded = make_kernel(h, reference=ref, embed_reference=True, **common)
    n = 0.3 * D * W * NX * NY
    config = hf.SolverConfig(max_iter=300, density_tol=1e-10, comm_tol=1e-10, mixing=0.3, method="rdiis")
    r_plain = hf.solve(plain, ref, n, config=config)
    r_embedded = hf.solve(embedded, ref, n, config=config)
    assert r_plain.converged and r_embedded.converged
    np.testing.assert_allclose(r_embedded.energy, r_plain.energy, atol=1e-9)
    np.testing.assert_allclose(r_embedded.mu, r_plain.mu, atol=1e-9)
    np.testing.assert_allclose(r_embedded.density, r_plain.density, atol=1e-7)


def test_upsample_density_is_exact_for_band_limited_densities():
    """Spectral upsampling between Monkhorst-Pack meshes with nk-dependent origins."""
    rng = np.random.default_rng(2)
    D = len(G) * NO
    A0, A1 = hermitian(rng, (D, D)), rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))

    def density(n):
        origin = -0.5 + 0.5 / n
        k = origin + np.arange(n) / n
        phase = np.exp(2j * np.pi * k)[:, None, None, None] * np.ones((1, n, 1, 1))
        P = A0 + phase * A1
        return 0.5 * (P + P.conj().swapaxes(-1, -2)), (origin, origin)

    (Pc, oc), (Pf, of) = density(4), density(8)
    coarse, fine = hf.Mesh.bz(B, 4), hf.Mesh.bz(B, 8)
    assert np.isclose(coarse.frac[0], oc[0]) and np.isclose(fine.frac[0], of[0])
    np.testing.assert_allclose(upsample_density(Pc, coarse, fine), Pf, atol=1e-13)


@pytest.mark.parametrize("hermitian", [False, True])
@pytest.mark.parametrize("kind", ["diagonal", "tiny_coherence"])
def test_sparse_channels_preserve_exact_convolution(x64, hermitian, kind):
    """The channel stream must not drop sparse or tiny ΔG channels."""
    rng = np.random.default_rng(14)
    D = len(G) * NO
    kernel = make_kernel(band_hamiltonian(rng))
    rho = np.zeros((NX, NY, D, D), complex)
    if kind == "diagonal":
        for g in range(len(G)):
            sl = slice(g * NO, (g + 1) * NO)
            rho[..., sl, sl] = rng.normal(size=(NX, NY, NO, NO)) + 1j * rng.normal(size=(NX, NY, NO, NO))
    else:
        rho[0, 1, 0, NO] = 1e-30 + 2e-30j  # far below any SCF tolerance, still nonzero
    if hermitian:
        rho = 0.5 * (rho + rho.conj().swapaxes(-2, -1))
    actual = kernel.exchange(rho)
    expected = direct_exchange(rho, scalar)
    scale = np.max(np.abs(expected))
    assert scale > 0
    assert np.max(np.abs(actual - expected)) / scale < 1e-12


@pytest.mark.parametrize("orb", [False, True], ids=["scalar", "orbital"])
def test_ffi_exchange_matches_pure_jax(x64, orb):
    import jax
    import jax.numpy as jnp

    from hartreefock._jax import ffi
    from hartreefock._jax import superlattice as sl

    if not ffi.available(ffi.SUPERLATTICE_FOCK):
        pytest.skip("built without the XLA FFI headers")
    rng = np.random.default_rng(11)
    D = len(G) * NO
    kernel = make_kernel(hermitian(rng, (NX, NY, D, D)), orbital if orb else scalar)
    layout = kernel.layout
    kernel_fn = sl._make_ffi_superlattice_fock_fn(layout, len(G), NO, NX, NY)
    if orb:
        pure = sl._make_orbital_superlattice_fock_fn(
            layout, len(G), NO, NX, NY, layout.V_lag_fft_orbital
        )
    else:
        pure = sl._make_scalar_superlattice_fock_fn(layout, len(G), NO, NX, NY)
    rho = jnp.asarray(hermitian(rng, (NX, NY, D, D)))
    np.testing.assert_allclose(kernel_fn(rho), pure(rho), atol=1e-13)
    # Non-Hermitian tangents go through the kernel without the ΔG halving.
    drho = jnp.asarray(rng.normal(size=rho.shape) + 1j * rng.normal(size=rho.shape))
    _, t_kernel = jax.jvp(kernel_fn, (rho,), (drho,))
    np.testing.assert_allclose(t_kernel, pure(drho), atol=1e-13)
    batch = jnp.stack([rho, 2.0 * rho])
    np.testing.assert_allclose(jax.vmap(kernel_fn)(batch)[1], 2.0 * pure(rho), atol=1e-13)
