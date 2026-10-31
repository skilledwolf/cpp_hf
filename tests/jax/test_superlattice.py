"""Tests for jax_hf.superlattice — streaming superlattice Fock function.

Covers:
  - layout indexing self-consistency (g_a_off, CSR pair tables)
  - JIT'd JAX Fock vs. a NumPy reference scatter/FFT/gather implementation
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax
# x64 is enabled in contimod's deployed configuration; mirror that here so the
# tests use the same double-precision FFT path that downstream callers see.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from hartreefock.backends.jax.superlattice import (
    ExtendedGridLayout,
    build_extended_layout,
    make_superlattice_build_fock_fn,
    make_superlattice_fock_fn,
)


def _hex_g_basis(g_cut: int):
    """Tiny hexagonal G-basis (gamma + first ``g_cut`` shells)."""
    a = 1.0
    b0 = 4 * np.pi / (np.sqrt(3) * a)
    b1 = np.array([np.sqrt(3) / 2, -0.5]) * b0
    b2 = np.array([0.0, 1.0]) * b0
    B = np.stack([b1, b2], axis=1)

    cart_norms, fracs = [], []
    for n1 in range(-g_cut - 1, g_cut + 2):
        for n2 in range(-g_cut - 1, g_cut + 2):
            v_cart = B @ np.array([n1, n2])
            fracs.append((n1, n2))
            cart_norms.append(np.linalg.norm(v_cart))
    cart_norms = np.array(cart_norms)
    fracs = np.array(fracs, dtype=np.int64)
    sorted_unique = np.sort(np.unique(np.round(cart_norms, 6)))
    cutoff = sorted_unique[min(g_cut, len(sorted_unique) - 1)] + 1e-3
    mask = cart_norms <= cutoff
    return fracs[mask], B


def _np_streaming_fock(rho, layout, n_G, dim_orb, nkx, nky):
    N_ext_x = layout.N_ext_x
    N_ext_y = layout.N_ext_y
    g_a_off = layout.g_a_off
    dpi = layout.delta_pair_i
    dpj = layout.delta_pair_j
    dps = layout.delta_pair_start
    V_lag_fft = layout.V_lag_fft
    rho_r = rho.reshape(nkx, nky, n_G, dim_orb, n_G, dim_orb)
    sigma = np.empty((nkx, nky, n_G, dim_orb, n_G, dim_orb), dtype=complex)
    rho_dg = np.empty((N_ext_x, N_ext_y, dim_orb, dim_orb), dtype=complex)
    for d in range(int(layout.n_delta)):
        rho_dg.fill(0)
        a, b = int(dps[d]), int(dps[d + 1])
        for k in range(a, b):
            i, j = int(dpi[k]), int(dpj[k])
            ox = nkx * int(g_a_off[i, 0])
            oy = nky * int(g_a_off[i, 1])
            rho_dg[ox:ox + nkx, oy:oy + nky, :, :] = rho_r[:, :, i, :, j, :]
        rho_fft = np.fft.fftn(rho_dg, axes=(0, 1))
        rho_fft *= V_lag_fft[:, :, None, None]
        sigma_dg = -np.fft.ifftn(rho_fft, axes=(0, 1))
        for k in range(a, b):
            i, j = int(dpi[k]), int(dpj[k])
            ox = nkx * int(g_a_off[i, 0])
            oy = nky * int(g_a_off[i, 1])
            sigma[:, :, i, :, j, :] = sigma_dg[ox:ox + nkx, oy:oy + nky, :, :]
    return sigma.reshape(nkx, nky, n_G * dim_orb, n_G * dim_orb)


@pytest.fixture
def small_layout():
    g_frac, B = _hex_g_basis(g_cut=1)
    n_G = int(g_frac.shape[0])
    nk = 6
    Vfunc = lambda q: 1.0 / np.sqrt(q ** 2 + 0.5 ** 2)
    layout = build_extended_layout(
        g_basis_fractional=g_frac, g_basis_B=B,
        nkx=nk, nky=nk, coulomb_V=Vfunc, w_scalar=1.0 / (nk * nk),
    )
    return dict(layout=layout, g_frac=g_frac, B=B, n_G=n_G, nk=nk)


def test_layout_csr_consistency(small_layout):
    layout = small_layout["layout"]
    n_G = small_layout["n_G"]
    seen = set()
    for d in range(layout.n_delta):
        a, b = int(layout.delta_pair_start[d]), int(layout.delta_pair_start[d + 1])
        for k in range(a, b):
            i, j = int(layout.delta_pair_i[k]), int(layout.delta_pair_j[k])
            assert int(layout.pair_to_delta[i, j]) == d
            seen.add((i, j))
    assert len(seen) == n_G * n_G


def test_jit_streaming_fock_matches_numpy(small_layout):
    layout = small_layout["layout"]
    n_G = small_layout["n_G"]
    nk = small_layout["nk"]
    dim_orb = 3
    D = n_G * dim_orb
    rng = np.random.default_rng(0)
    rho = (rng.standard_normal((nk, nk, D, D))
           + 1j * rng.standard_normal((nk, nk, D, D))) * 1e-3

    fock_fn = make_superlattice_fock_fn(layout, n_G, dim_orb, nk, nk)
    sigma_jax = np.asarray(fock_fn(jnp.asarray(rho)).block_until_ready())
    sigma_ref = _np_streaming_fock(rho, layout, n_G, dim_orb, nk, nk)

    rel = (np.max(np.abs(sigma_jax - sigma_ref))
           / max(np.max(np.abs(sigma_ref)), 1e-30))
    assert rel < 1e-12, f"JIT'd JAX vs NumPy streaming mismatch: {rel}"


def test_jit_streaming_fock_zero_density(small_layout):
    layout = small_layout["layout"]
    n_G = small_layout["n_G"]
    nk = small_layout["nk"]
    dim_orb = 2
    D = n_G * dim_orb
    fock_fn = make_superlattice_fock_fn(layout, n_G, dim_orb, nk, nk)
    sigma = np.asarray(fock_fn(jnp.zeros((nk, nk, D, D), dtype=jnp.complex128))
                        .block_until_ready())
    assert np.max(np.abs(sigma)) < 1e-12


def test_build_fock_adapter_includes_contact_terms():
    g_frac = np.array([[0, 0]], dtype=np.int64)
    layout = build_extended_layout(
        g_basis_fractional=g_frac,
        g_basis_B=np.eye(2),
        nkx=1,
        nky=1,
        coulomb_V=lambda q: np.zeros_like(q),
        w_scalar=1.0,
    )
    build_fock_sl = make_superlattice_build_fock_fn(
        layout,
        n_G=1,
        dim_orb=2,
        nkx=1,
        nky=1,
        HH_GG=np.zeros((1, 1)),
        hartree_degeneracy=1.0,
    )
    P = jnp.zeros((1, 1, 2, 2), dtype=jnp.complex128)
    P = P.at[..., 0, 0].set(0.8).at[..., 1, 1].set(0.2)
    sigma_z = jnp.diag(jnp.array([1.0, -1.0], dtype=jnp.float64)).astype(
        jnp.complex128
    )

    Sigma, _, _ = build_fock_sl(
        P,
        h=jnp.zeros_like(P),
        VR=None,
        refP=jnp.zeros_like(P),
        HH=jnp.zeros((2, 2)),
        w2d=jnp.ones((1, 1)),
        include_exchange=True,
        include_hartree=False,
        exchange_hermitian_channel_packing=False,
        contact_g=jnp.array([1.0]),
        contact_Oi=sigma_z[None, ...],
        contact_Oj=sigma_z[None, ...],
    )

    expected = np.diag([-0.2, -0.8]).reshape(1, 1, 2, 2)
    np.testing.assert_allclose(np.asarray(Sigma), expected, atol=1e-12)


def test_build_fock_orbital_hartree_reduces_to_scalar(small_layout):
    """When ``HH_GG_orbital[A, B, α, γ] = HH_GG[A, B]`` for all α, γ the orbital
    path must reproduce the scalar path identically.  Closes the
    backend-asymmetry gap where the JAX build-fock factory previously had no
    orbital-resolved Hartree at all (only contimod's cpp_hf wrapper did).

    Note the equivalence requires ``HH_GG_orbital`` to be the *full*
    ``δ_{αβ}``-blind product, not a Kronecker delta in orbital — the scalar
    formula traces over γ before applying V, so all (α, γ) entries must be
    equal to recover it.  A Kronecker-δ in orbital would describe a
    *different* physics (Hartree only couples like orbitals).
    """
    from hartreefock.backends.jax.superlattice import make_superlattice_build_fock_fn

    layout = small_layout["layout"]
    n_G = small_layout["n_G"]
    nk = small_layout["nk"]
    dim_orb = 3
    D = n_G * dim_orb
    rng = np.random.default_rng(11)
    HH_GG = rng.standard_normal((n_G, n_G))
    HH_GG = HH_GG + HH_GG.T
    np.fill_diagonal(HH_GG, 0.0)
    # Orbital-blind HH_GG_orbital that recovers the scalar formula.
    HH_orb = np.broadcast_to(
        HH_GG[:, :, None, None], (n_G, n_G, dim_orb, dim_orb),
    ).copy().astype(np.float64)

    h = jnp.zeros((nk, nk, D, D), dtype=jnp.complex128)
    refP = jnp.zeros_like(h)
    rho = (rng.standard_normal((nk, nk, D, D))
           + 1j * rng.standard_normal((nk, nk, D, D))) * 1e-3
    rho = 0.5 * (rho + np.swapaxes(rho.conj(), -1, -2))
    P = jnp.asarray(rho)
    w2d = jnp.full((nk, nk), 1.0 / (nk * nk), dtype=jnp.float64)
    deg = 1.6

    build_scalar = make_superlattice_build_fock_fn(
        layout, n_G, dim_orb, nk, nk,
        HH_GG=HH_GG, hartree_degeneracy=deg,
    )
    build_orbital = make_superlattice_build_fock_fn(
        layout, n_G, dim_orb, nk, nk,
        HH_GG=HH_GG, hartree_degeneracy=deg,
        HH_GG_orbital=HH_orb,
    )

    kw = dict(
        h=h, VR=None, refP=refP,
        HH=jnp.zeros((D, D)), w2d=w2d,
        include_exchange=False, include_hartree=True,
        exchange_hermitian_channel_packing=False,
        contact_g=jnp.zeros((1,)), contact_Oi=jnp.zeros((1, D, D)),
        contact_Oj=jnp.zeros((1, D, D)),
    )
    _, _, F_scalar = build_scalar(P, **kw)
    _, _, F_orbital = build_orbital(P, **kw)
    diff = np.max(np.abs(np.asarray(F_orbital) - np.asarray(F_scalar)))
    assert diff < 1e-12, f"orbital path with broadcast HH disagrees with scalar: {diff}"


def test_build_fock_orbital_hartree_q0_warns_and_is_included(small_layout):
    """A nonzero diagonal-G block is the q=0 layer Hartree: allowed but warned.

    Mirrors cpp_hf relaxing the validator from raise to warn so a layer-resolved
    system at finite displacement field can keep the gate-screened q=0 layer
    Hartree.  The contribution must actually change the Fock.
    """
    from hartreefock.backends.jax.superlattice import make_superlattice_build_fock_fn

    layout = small_layout["layout"]
    n_G = small_layout["n_G"]
    nk = small_layout["nk"]
    dim_orb = 2
    D = n_G * dim_orb
    rng = np.random.default_rng(7)

    HH_off = rng.standard_normal((n_G, n_G, dim_orb, dim_orb)).astype(np.float64)
    HH_off[np.arange(n_G), np.arange(n_G)] = 0.0      # moiré/TBG convention: q=0 dropped
    HH_q0 = HH_off.copy()
    HH_q0[np.arange(n_G), np.arange(n_G), 0, 0] = 0.5  # keep a q=0 layer Hartree

    h = jnp.zeros((nk, nk, D, D), dtype=jnp.complex128)
    refP = jnp.zeros_like(h)
    rho = (rng.standard_normal((nk, nk, D, D))
           + 1j * rng.standard_normal((nk, nk, D, D))) * 1e-3
    rho = 0.5 * (rho + np.swapaxes(rho.conj(), -1, -2))
    P = jnp.asarray(rho)
    w2d = jnp.full((nk, nk), 1.0 / (nk * nk), dtype=jnp.float64)

    build_off = make_superlattice_build_fock_fn(
        layout, n_G, dim_orb, nk, nk,
        HH_GG=np.zeros((n_G, n_G)), hartree_degeneracy=1.0, HH_GG_orbital=HH_off)
    # a nonzero diagonal-G block must WARN (not raise)
    with pytest.warns(UserWarning, match="q=0"):
        build_q0 = make_superlattice_build_fock_fn(
            layout, n_G, dim_orb, nk, nk,
            HH_GG=np.zeros((n_G, n_G)), hartree_degeneracy=1.0, HH_GG_orbital=HH_q0)

    kw = dict(
        h=h, VR=None, refP=refP, HH=jnp.zeros((D, D)), w2d=w2d,
        include_exchange=False, include_hartree=True,
        exchange_hermitian_channel_packing=False,
        contact_g=jnp.zeros((1,)), contact_Oi=jnp.zeros((1, D, D)),
        contact_Oj=jnp.zeros((1, D, D)),
    )
    _, _, F_off = build_off(P, **kw)
    _, _, F_q0 = build_q0(P, **kw)
    # the q=0 layer Hartree is actually computed -> the Fock changes
    assert np.max(np.abs(np.asarray(F_q0) - np.asarray(F_off))) > 1e-9
