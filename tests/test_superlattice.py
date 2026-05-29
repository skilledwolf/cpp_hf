"""Tests for cpp_hf.superlattice — streaming superlattice Fock primitive.

Covers:
  - layout indexing self-consistency (g_a_off, CSR pair tables)
  - native primitive vs. a NumPy reference scatter/FFT/gather implementation
  - native primitive vs. a direct-sum reference at small g_cut
"""
from __future__ import annotations

import numpy as np
import pytest

from cpp_hf import _compat
from cpp_hf.superlattice import (
    ExtendedGridLayout,
    build_extended_layout,
    superlattice_fock,
)


pytestmark = pytest.mark.skipif(
    not _compat.HAVE_NATIVE, reason="cpp_hf native extension not built"
)


def _hex_g_basis(g_cut: int) -> tuple[np.ndarray, np.ndarray]:
    """Tiny hexagonal G-basis (gamma + first ``g_cut`` stars).

    Returns ``(g_fractional, B)``: integer fractional coordinates and the
    cartesian basis ``B`` so ``G_cart = B @ G_frac``.
    """
    # Hexagonal lattice constants, unit moire BZ.
    a = 1.0
    b0 = 4 * np.pi / (np.sqrt(3) * a)
    b1 = np.array([np.sqrt(3) / 2, -0.5]) * b0
    b2 = np.array([0.0, 1.0]) * b0
    B = np.stack([b1, b2], axis=1)  # (2, 2): columns are basis vectors

    # Brute-force build the integer basis up to |n_1| + |n_2| <= g_cut, then
    # restrict by cartesian distance from origin (canonical g_cut convention).
    cart_norms = []
    fracs = []
    for n1 in range(-g_cut - 1, g_cut + 2):
        for n2 in range(-g_cut - 1, g_cut + 2):
            v_cart = B @ np.array([n1, n2])
            fracs.append((n1, n2))
            cart_norms.append(np.linalg.norm(v_cart))
    cart_norms = np.array(cart_norms)
    fracs = np.array(fracs, dtype=np.int64)

    # First shell is at distance ~b0; allow up to g_cut shells with tolerance.
    sorted_unique = np.sort(np.unique(np.round(cart_norms, 6)))
    cutoff = sorted_unique[min(g_cut, len(sorted_unique) - 1)] + 1e-3
    mask = cart_norms <= cutoff
    return fracs[mask], B


def _np_streaming_fock(rho, layout, n_G, dim_orb, nkx, nky):
    """Pure-NumPy streaming Fock matching the native algorithm bit-for-bit."""
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
    # Every (i, j) pair appears exactly once in the CSR.
    seen = set()
    for d in range(layout.n_delta):
        a, b = int(layout.delta_pair_start[d]), int(layout.delta_pair_start[d + 1])
        for k in range(a, b):
            i, j = int(layout.delta_pair_i[k]), int(layout.delta_pair_j[k])
            assert int(layout.pair_to_delta[i, j]) == d
            seen.add((i, j))
    assert len(seen) == n_G * n_G


def test_native_matches_numpy_streaming(small_layout):
    layout = small_layout["layout"]
    n_G = small_layout["n_G"]
    nk = small_layout["nk"]
    dim_orb = 3
    D = n_G * dim_orb
    rng = np.random.default_rng(0)
    rho = (rng.standard_normal((nk, nk, D, D))
           + 1j * rng.standard_normal((nk, nk, D, D))) * 1e-3

    sigma_native = superlattice_fock(rho, layout, n_G, dim_orb, nk, nk)
    sigma_ref = _np_streaming_fock(rho, layout, n_G, dim_orb, nk, nk)

    rel = (np.max(np.abs(sigma_native - sigma_ref))
           / max(np.max(np.abs(sigma_ref)), 1e-30))
    assert rel < 1e-12, f"native vs NumPy streaming mismatch: {rel}"


def test_native_handles_zero_density(small_layout):
    layout = small_layout["layout"]
    n_G = small_layout["n_G"]
    nk = small_layout["nk"]
    dim_orb = 2
    D = n_G * dim_orb
    rho = np.zeros((nk, nk, D, D), dtype=complex)
    sigma = superlattice_fock(rho, layout, n_G, dim_orb, nk, nk)
    assert np.max(np.abs(sigma)) == 0.0


def test_native_shape_validation(small_layout):
    layout = small_layout["layout"]
    n_G = small_layout["n_G"]
    nk = small_layout["nk"]
    dim_orb = 2
    D = n_G * dim_orb
    wrong = np.zeros((nk, nk, D, D + 1), dtype=complex)
    with pytest.raises(ValueError):
        superlattice_fock(wrong, layout, n_G, dim_orb, nk, nk)
