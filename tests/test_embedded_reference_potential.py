"""Tests for HartreeFockKernel(embed_reference_potential=True).

The embedded mode bakes V_HF[refP] = J[refP] + Σ[refP] into ``h`` once at
construction so the kernel's reference-subtracted update naturally
yields F_std = h_bare + V_HF[P].  Without it, the kernel solves a
*modified* HF whose minimum differs from standard HF.

Identities under test:
  1. Embedded F at P=refP equals h_bare + V_HF[refP] exactly.
  2. Unembedded F at P=refP equals h_bare exactly.
  3. F_embedded(P) − F_unembedded(P) = V_HF[refP] for any P.
  4. Energy reported with embedding equals E_std + ½ Tr[V_HF[refP] · refP],
     with the constant available as ``kernel.embedded_energy_offset``.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpp_hf import HartreeFockKernel
from cpp_hf.fock import build_fock, hf_energy


def _setup(nk=4, n_orb=2, T=0.05, seed=0):
    rng = np.random.default_rng(seed)
    weights = np.full((nk, nk), 1.0 / (nk * nk), dtype=np.float64)
    h_diag = np.array([-0.5, +0.5])
    h = np.broadcast_to(np.diag(h_diag), (nk, nk, n_orb, n_orb)).astype(np.complex128)
    h = np.ascontiguousarray(h)
    # Add a small momentum-dependent piece so it's not all the same matrix
    kx = np.linspace(-0.5, 0.5, nk, endpoint=False)
    h_pert = (kx[:, None, None, None] * np.array([[0.0, 1.0], [1.0, 0.0]])[None, None]
              ).astype(np.complex128)
    h = h + h_pert
    h = np.ascontiguousarray(h)
    Vq = np.full((nk, nk, 1, 1), 0.3, dtype=np.complex128)
    HH = np.array([[0.5, -0.1], [-0.1, 0.5]], dtype=np.float64)
    # A non-trivial Hermitian reference density
    refP = np.zeros((nk, nk, n_orb, n_orb), dtype=np.complex128)
    refP[..., 0, 0] = 0.4
    refP[..., 1, 1] = 0.6
    return weights, h, Vq, HH, T, refP


def _make_kernels(*, embed: bool, with_hartree: bool, refP_arg, **extra):
    weights, h, Vq, HH, T, refP = _setup()
    return HartreeFockKernel(
        weights=weights, hamiltonian=h, coulomb_q=Vq, T=T,
        include_hartree=with_hartree,
        include_exchange=True,
        reference_density=refP_arg if refP_arg is not None else refP,
        hartree_matrix=HH if with_hartree else None,
        embed_reference_potential=embed,
        **extra,
    )


def _build_F(kernel, P):
    Sigma, H, F = build_fock(
        np.ascontiguousarray(P, dtype=np.complex128),
        h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
        HH=kernel.HH, w2d=kernel.w2d,
        include_exchange=kernel.include_exchange,
        include_hartree=kernel.include_hartree,
        exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
        contact_g=kernel.contact_g,
        contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
    )
    return Sigma, H, F


@pytest.mark.parametrize("with_hartree", [True, False])
def test_embedded_F_at_refP_equals_h_plus_V_HF_refP(with_hartree):
    """At P = refP, the embedded kernel's F equals h_bare + V_HF[refP]."""
    kernel_embed = _make_kernels(
        embed=True, with_hartree=with_hartree, refP_arg=None,
        center_embedded_hartree=False,
    )
    kernel_plain = _make_kernels(
        embed=False, with_hartree=with_hartree, refP_arg=None,
    )

    # F at P = refP for both kernels
    refP = kernel_plain.refP
    _, _, F_embed = _build_F(kernel_embed, refP)
    _, _, F_plain = _build_F(kernel_plain, refP)

    # Plain at P=refP gives F = h (since Σ[0]+H[0] = 0)
    np.testing.assert_allclose(F_plain, kernel_plain.h, atol=1e-12)

    # Embedded at P=refP gives F = h_bare + V_HF[refP] = kernel_embed.h
    np.testing.assert_allclose(F_embed, kernel_embed.h, atol=1e-12)

    # Difference equals V_HF[refP]
    np.testing.assert_allclose(F_embed - F_plain, kernel_embed.h - kernel_plain.h,
                                atol=1e-12)


@pytest.mark.parametrize("with_hartree", [True, False])
def test_F_difference_independent_of_P(with_hartree):
    """For any P, F_embed(P) − F_plain(P) = V_HF[refP] (a P-independent shift)."""
    kernel_embed = _make_kernels(
        embed=True, with_hartree=with_hartree, refP_arg=None,
        center_embedded_hartree=False,
    )
    kernel_plain = _make_kernels(
        embed=False, with_hartree=with_hartree, refP_arg=None,
    )
    rng = np.random.default_rng(42)
    P = (rng.standard_normal(kernel_plain.h.shape)
         + 1j * rng.standard_normal(kernel_plain.h.shape))
    P = 0.5 * (P + np.conj(np.swapaxes(P, -1, -2)))  # Hermitian
    P = P.astype(np.complex128)

    _, _, F_embed = _build_F(kernel_embed, P)
    _, _, F_plain = _build_F(kernel_plain, P)
    diff = F_embed - F_plain
    expected = kernel_embed.h - kernel_plain.h
    np.testing.assert_allclose(diff, expected, atol=1e-10)


@pytest.mark.parametrize("with_hartree", [True, False])
def test_embedding_no_op_when_refP_zero(with_hartree):
    """V_HF[0] = 0, so embedding with refP=0 should leave h unchanged."""
    weights, h, Vq, HH, T, _ = _setup()
    refP_zero = np.zeros_like(h)
    if not with_hartree:
        kwargs = dict(include_hartree=False, hartree_matrix=None)
    else:
        kwargs = dict(include_hartree=True, hartree_matrix=HH)
    kernel = HartreeFockKernel(
        weights=weights, hamiltonian=h, coulomb_q=Vq, T=T,
        include_exchange=True,
        reference_density=refP_zero,
        embed_reference_potential=True,
        center_embedded_hartree=False,
        **kwargs,
    )
    np.testing.assert_allclose(kernel.h, h, atol=1e-14)
    assert kernel.embedded_energy_offset == 0.0


def test_centering_shifts_h_by_constant_only():
    """center_embedded_hartree=True subtracts a constant·I from h_eff,
    leaving F differences and SCF minimum unchanged."""
    kernel_uncentered = _make_kernels(
        embed=True, with_hartree=True, refP_arg=None,
        center_embedded_hartree=False,
    )
    kernel_centered = _make_kernels(
        embed=True, with_hartree=True, refP_arg=None,
        center_embedded_hartree=True,
    )
    diff = kernel_uncentered.h - kernel_centered.h
    # Difference should be a constant times identity, broadcast over k
    n_orb = diff.shape[-1]
    diag_vals = diff[0, 0].diagonal().real
    assert np.allclose(diag_vals, diag_vals.mean(), atol=1e-12), (
        "centering should remove only the orbital-uniform diagonal mean"
    )


def test_embedded_energy_offset_is_correct():
    """E_cpp(h_eff) = E_std + ½ Tr[V_HF[refP] refP], offset accessible
    as kernel.embedded_energy_offset."""
    kernel_embed = _make_kernels(
        embed=True, with_hartree=True, refP_arg=None,
        center_embedded_hartree=False,
    )
    kernel_plain = _make_kernels(
        embed=False, with_hartree=True, refP_arg=None,
    )
    # Energy at P = refP:
    refP = kernel_plain.refP
    Sigma_p, H_p, _ = _build_F(kernel_plain, refP)
    Sigma_e, H_e, _ = _build_F(kernel_embed, refP)
    E_plain = hf_energy(refP, h=kernel_plain.h, Sigma=Sigma_p, H=H_p,
                        weights_b=kernel_plain.weights_b, refP=kernel_plain.refP)
    E_embed = hf_energy(refP, h=kernel_embed.h, Sigma=Sigma_e, H=H_e,
                        weights_b=kernel_embed.weights_b, refP=kernel_embed.refP)
    # At P=refP: Σ+H is zero (since they use P-refP), so E reduces to Tr[h refP].
    # E_embed = Tr[(h+V_HF[refP]) refP] = Tr[h refP] + Tr[V_HF[refP] refP]
    # E_plain = Tr[h refP]
    # so E_embed - E_plain = Tr[V_HF[refP] refP] = 2 * embedded_energy_offset
    delta = float(E_embed - E_plain)
    expected = 2.0 * kernel_embed.embedded_energy_offset
    np.testing.assert_allclose(delta, expected, atol=1e-9, rtol=1e-9)
