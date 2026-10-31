"""Superlattice kernel: exchange and Hartree against direct sums, backend parity,
reference embedding and spectral upsampling."""

from __future__ import annotations

import numpy as np
import pytest

import hartreefock as hf
from hartreefock.superlattice import SuperlatticeHartreeFockKernel, upsample_density

G = np.array([[0, 0], [1, 0], [0, -1], [-1, 1]])
B = np.array([[1.0, 0.35], [0.1, 0.8]])
NX, NY, NO = 3, 4, 2
W = 1.0 / (NX * NY)
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


def make_kernel(h, **kw):
    kw.setdefault("T", 0.1)
    return SuperlatticeHartreeFockKernel(np.full((NX, NY), W), h, scalar, G, B, **kw)


@pytest.mark.parametrize("orb", [False, True], ids=["scalar", "orbital"])
def test_exchange_and_hartree_match_direct_sums(orb):
    rng = np.random.default_rng(3)
    D = len(G) * NO
    kernel = make_kernel(
        hermitian(rng, (NX, NY, D, D)),
        include_hartree=True,
        coulomb_V_orbital=orbital if orb else None,
        hartree_degeneracy=2.0,
    )
    rho = hermitian(rng, (NX, NY, D, D))
    np.testing.assert_allclose(kernel.exchange(rho), direct_exchange(rho, orbital if orb else scalar), atol=1e-13)
    # Scalar Hartree drops q = 0; orbital Hartree keeps the gate-screened q = 0 blocks.
    HH = kernel.HH_GG_orbital if orb else kernel.HH_GG
    assert (np.abs(np.diagonal(HH, axis1=0, axis2=1)).max() > 0) == orb
    np.testing.assert_allclose(kernel.hartree(rho), direct_hartree(rho, HH, 2.0), atol=1e-13)

    # The off-mesh plane-wave sum reproduces the mesh potential on mesh points.
    k_mesh = np.stack(np.meshgrid(np.arange(NX) / NX - 0.5, np.arange(NY) / NY - 0.5, indexing="ij"), -1)
    off_mesh = kernel.potential_at(rho, k_mesh.reshape(-1, 2)).reshape(rho.shape)
    np.testing.assert_allclose(off_mesh, kernel.potential(rho), atol=1e-13)


@pytest.mark.parametrize(
    "case",
    [dict(), dict(coulomb_V_orbital=orbital), dict(embed_reference_potential=True, center_embedded_hartree=False)],
    ids=["scalar", "orbital", "embedded"],
)
def test_jax_fock_matches_native(x64, case):
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from hartreefock.backends.jax.fock import fock_builder, hf_energy
    from hartreefock.backends.native import build_fock

    rng = np.random.default_rng(5)
    D = len(G) * NO
    ref = np.broadcast_to(0.5 * np.eye(D), (NX, NY, D, D))
    kernel = make_kernel(
        hermitian(rng, (NX, NY, D, D)), include_hartree=True, reference_density=ref,
        hartree_degeneracy=2.0, **case,
    )
    P = hermitian(rng, (NX, NY, D, D))
    _, _, F, E = build_fock(kernel, P)
    np.testing.assert_allclose(F, kernel.h + kernel.potential(P - kernel.refP), atol=1e-13)

    args = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in kernel.as_args().items()}
    build = fock_builder(kernel)
    fock_args = dict(
        h=args["h"], VR=args["VR"], refP=args["refP"], HH=args["HH"], w2d=jnp.asarray(kernel.w2d),
        include_exchange=True, include_hartree=True, exchange_hermitian_channel_packing=False,
        contact_g=args["contact_g"], contact_Oi=args["contact_Oi"], contact_Oj=args["contact_Oj"],
    )
    Sj, Hj, Fj = build(jnp.asarray(P), **fock_args)
    Ej = hf_energy(jnp.asarray(P), h=args["h"], Sigma=Sj, H=Hj, weights_b=args["weights_b"], refP=args["refP"])
    np.testing.assert_allclose(np.asarray(Fj), F, atol=1e-13)
    np.testing.assert_allclose(float(Ej), float(E), atol=1e-13)

    # The builder is affine: its JVP is the self-adjoint interaction response.
    X, Y = (jnp.asarray(hermitian(rng, P.shape)) for _ in range(2))
    response = lambda dP: jax.jvp(lambda A: build(A, **fock_args)[2], (jnp.asarray(P),), (dP,))[1]
    inner = lambda A, Bm: W * jnp.vdot(A, Bm).real
    np.testing.assert_allclose(float(inner(X, response(Y))), float(inner(Y, response(X))), atol=1e-13)


def test_embedding_solves_standard_hf(x64):
    """An embedded kernel converges to the unreferenced HF state, its energy offset by a constant."""
    rng = np.random.default_rng(11)
    h = band_hamiltonian(rng)
    D = h.shape[-1]
    ref = np.broadcast_to(0.5 * np.eye(D), h.shape)
    common = dict(include_hartree=True, coulomb_V_orbital=orbital, hartree_degeneracy=2.0)
    plain = make_kernel(h, **common)
    embedded = make_kernel(
        h, reference_density=ref, embed_reference_potential=True, center_embedded_hartree=False, **common,
    )
    n = 0.3 * D * W * NX * NY
    config = hf.SCFConfig(max_iter=300, density_tol=1e-10, comm_tol=1e-10, mixing=0.3, acceleration="diis")
    results = {}
    for backend in hf.available_backends():
        for name, kernel in (("plain", plain), ("embedded", embedded)):
            r = hf.solve_scf(kernel, ref, n, config=config, backend=backend)
            assert r.converged, (backend, name, r.message)
            results[backend, name] = r
    E_plain = float(results["native", "plain"].energy)
    for (backend, name), r in results.items():
        offset = embedded.embedded_energy_offset if name == "embedded" else 0.0
        np.testing.assert_allclose(float(r.energy) - offset, E_plain, atol=1e-9)
        np.testing.assert_allclose(r.density, results["native", "plain"].density, atol=1e-7)


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
    h_c, h_f = np.zeros((4, 4, D, D)), np.zeros((8, 8, D, D))
    coarse = SuperlatticeHartreeFockKernel(np.full((4, 4), 1 / 16), h_c, scalar, G, B, T=0.1, k_origin=oc)
    fine = SuperlatticeHartreeFockKernel(np.full((8, 8), 1 / 64), h_f, scalar, G, B, T=0.1, k_origin=of)
    np.testing.assert_allclose(upsample_density(Pc, coarse, fine), Pf, atol=1e-13)
