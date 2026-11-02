"""Mirror of cpp_hf/tests/test_embedded_reference_potential.py for jax_hf.

Same identities, same conventions — verifies the JAX implementation
behaves identically to the cpp_hf one for the embedded reference
potential mode.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _meshes import mesh_for

from hartreefock import HartreeFockKernel
from hartreefock._jax.fock import build_fock, hf_energy


def _setup(nk=4, n_orb=2, T=0.05, seed=0):
    weights = np.full((nk, nk), 1.0 / (nk * nk), dtype=np.float64)
    h_diag = np.array([-0.5, +0.5])
    h = np.broadcast_to(np.diag(h_diag), (nk, nk, n_orb, n_orb)).astype(np.complex128)
    h = np.ascontiguousarray(h)
    kx = np.linspace(-0.5, 0.5, nk, endpoint=False)
    h_pert = (kx[:, None, None, None] * np.array([[0.0, 1.0], [1.0, 0.0]])[None, None]
              ).astype(np.complex128)
    h = h + h_pert
    h = np.ascontiguousarray(h)
    Vq = np.full((nk, nk, 1, 1), 0.3, dtype=np.complex128)
    HH = np.array([[0.5, -0.1], [-0.1, 0.5]], dtype=np.float64)
    refP = np.zeros((nk, nk, n_orb, n_orb), dtype=np.complex128)
    refP[..., 0, 0] = 0.4
    refP[..., 1, 1] = 0.6
    return weights, h, Vq, HH, T, refP


def _make_kernels(*, embed: bool, with_hartree: bool, refP_arg, **extra):
    weights, h, Vq, HH, T, refP = _setup()
    return HartreeFockKernel(mesh_for(weights), h, Vq, hartree=HH if with_hartree else False, T=T, exchange=True, reference=refP_arg if refP_arg is not None else refP, embed_reference=embed, **extra)


def _build_F(kernel, P):
    Sigma, H, F = build_fock(
        jnp.asarray(P, dtype=jnp.complex128),
        h=kernel.h, VR=kernel.VR_shifted, refP=kernel.refP,
        HH=kernel.HH, w2d=kernel.w2d,
        include_exchange=kernel.include_exchange,
        include_hartree=kernel.include_hartree,
        exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
        contact_g=kernel.contact_g,
        contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
    )
    return np.asarray(Sigma), np.asarray(H), np.asarray(F)


@pytest.mark.parametrize("with_hartree", [True, False])
def test_embedded_F_at_refP_equals_h_plus_V_HF_refP(with_hartree):
    kernel_embed = _make_kernels(
        embed=True, with_hartree=with_hartree, refP_arg=None
    )
    kernel_plain = _make_kernels(
        embed=False, with_hartree=with_hartree, refP_arg=None,
    )
    refP = np.asarray(kernel_plain.refP)
    _, _, F_embed = _build_F(kernel_embed, refP)
    _, _, F_plain = _build_F(kernel_plain, refP)

    np.testing.assert_allclose(F_plain, np.asarray(kernel_plain.h), atol=1e-6)
    np.testing.assert_allclose(F_embed, np.asarray(kernel_embed.h), atol=1e-6)
    np.testing.assert_allclose(
        F_embed - F_plain,
        np.asarray(kernel_embed.h - kernel_plain.h),
        atol=1e-6,
    )


@pytest.mark.parametrize("with_hartree", [True, False])
def test_F_difference_independent_of_P(with_hartree):
    kernel_embed = _make_kernels(
        embed=True, with_hartree=with_hartree, refP_arg=None
    )
    kernel_plain = _make_kernels(
        embed=False, with_hartree=with_hartree, refP_arg=None,
    )
    rng = np.random.default_rng(42)
    P = (rng.standard_normal(kernel_plain.h.shape)
         + 1j * rng.standard_normal(kernel_plain.h.shape))
    P = 0.5 * (P + np.conj(np.swapaxes(P, -1, -2)))
    P = P.astype(np.complex128)

    _, _, F_embed = _build_F(kernel_embed, P)
    _, _, F_plain = _build_F(kernel_plain, P)
    diff = F_embed - F_plain
    expected = np.asarray(kernel_embed.h - kernel_plain.h)
    np.testing.assert_allclose(diff, expected, atol=1e-5)


@pytest.mark.parametrize("with_hartree", [True, False])
def test_embedding_no_op_when_refP_zero(with_hartree):
    weights, h, Vq, HH, T, _ = _setup()
    refP_zero = np.zeros_like(h)
    kwargs = dict(hartree=HH) if with_hartree else {}
    kernel = HartreeFockKernel(mesh_for(weights), h, Vq, T=T, exchange=True, reference=refP_zero, embed_reference=True, **kwargs)
    np.testing.assert_allclose(np.asarray(kernel.h), h, atol=1e-6)
    assert float(kernel.embedded_energy_offset) == 0.0


@pytest.mark.parametrize("density_scale", [0.6, 1.4])
def test_embedded_energy_differs_by_the_offset(density_scale):
    weights, h, Vq, HH, T, refP = _setup(nk=2)
    weights *= .3  # The gauge correction uses physical, unnormalized counts.
    common = dict(mesh=mesh_for(weights), h=h, V=Vq, T=T, hartree=HH)
    embedded = HartreeFockKernel(reference=refP, embed_reference=True, **common)
    standard = HartreeFockKernel(reference=np.zeros_like(refP), **common)
    P = density_scale * refP

    def evaluate(kernel, density):
        Sigma, H, F = _build_F(kernel, density)
        E = hf_energy(jnp.asarray(density), h=kernel.h, Sigma=jnp.asarray(Sigma),
                      H=jnp.asarray(H), weights_b=kernel.weights_b, refP=kernel.refP)
        return F, float(E)

    F_standard, E_standard = evaluate(standard, P)
    F_embedded, E_embedded = evaluate(embedded, P)
    np.testing.assert_allclose(E_embedded - embedded.embedded_energy_offset, E_standard, atol=1e-6)
    np.testing.assert_allclose(F_embedded, F_standard, atol=1e-6)
    Sigma_ref, H_ref, _ = _build_F(standard, refP)
    expected_offset = .5 * np.einsum(
        "ij,ijab,ijba->", weights, Sigma_ref + H_ref, refP,
    ).real
    np.testing.assert_allclose(embedded.embedded_energy_offset, expected_offset, atol=1e-6)


def test_embedding_includes_contact_reference_potential_and_offset():
    weights = np.ones((1, 1), dtype=np.float32)
    h = np.zeros((1, 1, 2, 2), dtype=np.complex64)
    Vq = np.zeros((1, 1, 1, 1), dtype=np.float32)
    refP = np.zeros_like(h)
    refP[..., 0, 0] = 0.8
    refP[..., 1, 1] = 0.2
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex64)
    contact_terms = [(1.0, sigma_z, sigma_z)]

    kernel_plain = HartreeFockKernel(mesh_for(weights), h, Vq, T=0.1, reference=refP, contact_terms=contact_terms)
    kernel_embed = HartreeFockKernel(mesh_for(weights), h, Vq, T=0.1, reference=refP, contact_terms=contact_terms, embed_reference=True)

    # For O_i = O_j = sigma_z and rho_bar = diag(0.8, 0.2):
    # g O_i tr(O_j rho_bar) - g O_i rho_bar O_j = diag(-0.2, -0.8).
    expected_contact = np.diag([-0.2, -0.8]).astype(np.complex64)
    expected_shift = np.broadcast_to(expected_contact, h.shape)
    np.testing.assert_allclose(
        np.asarray(kernel_embed.h - kernel_plain.h),
        expected_shift,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(kernel_embed.embedded_energy_offset),
        -0.16,
        atol=1e-6,
    )
