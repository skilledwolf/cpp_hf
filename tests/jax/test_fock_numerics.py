"""Focused regression tests for Fock/energy numerical conventions."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from hartreefock.backends.jax.fock import build_fock, hf_energy, occupation_entropy
from hartreefock.backends.jax.problem import HartreeFockKernel


def test_reference_subtracted_energy_gradient_matches_fock():
    """dE/dP must be the weighted Fock matrix when refP is nonzero."""
    weights = jnp.array([[0.25, 0.75]], dtype=jnp.float32)
    weights_b = weights[..., None, None]
    P = jnp.array(
        [[
            [[0.7, 0.0], [0.0, 0.1]],
            [[0.4, 0.0], [0.0, 0.6]],
        ]],
        dtype=jnp.float32,
    )
    refP = jnp.array(
        [[
            [[0.2, 0.0], [0.0, 0.4]],
            [[0.3, 0.0], [0.0, 0.2]],
        ]],
        dtype=jnp.float32,
    )
    h = jnp.array(
        [[
            [[-0.4, 0.0], [0.0, 0.2]],
            [[0.1, 0.0], [0.0, -0.3]],
        ]],
        dtype=jnp.float32,
    )
    HH = jnp.array([[2.0, 0.3], [0.3, 1.0]], dtype=jnp.float32)

    def energy(P_trial):
        Sigma, H, _ = build_fock(
            P_trial,
            h=h,
            VR=jnp.zeros((1, 2, 1, 1), dtype=jnp.complex64),
            refP=refP,
            HH=HH,
            w2d=weights,
            include_exchange=False,
            include_hartree=True,
            exchange_hermitian_channel_packing=False,
        )
        return hf_energy(
            P_trial,
            h=h,
            Sigma=Sigma,
            H=H,
            weights_b=weights_b,
            refP=refP,
        )

    Sigma, H, F = build_fock(
        P,
        h=h,
        VR=jnp.zeros((1, 2, 1, 1), dtype=jnp.complex64),
        refP=refP,
        HH=HH,
        w2d=weights,
        include_exchange=False,
        include_hartree=True,
        exchange_hermitian_channel_packing=False,
    )
    del Sigma, H
    gradient = jax.grad(energy)(P)

    np.testing.assert_allclose(
        np.asarray(gradient),
        np.asarray(weights_b * F),
        atol=2e-6,
        rtol=2e-6,
    )


def test_exchange_applies_nonuniform_weights_to_density():
    """Σ(k) = -sum_q w(q) V(k-q) P(q), including nonuniform w(q)."""
    weights = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    Vq = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    density = np.arange(1, 5, dtype=np.float32).reshape(2, 2)
    h = np.zeros((2, 2, 1, 1), dtype=np.complex64)
    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=h,
        coulomb_q=Vq[..., None, None],
        T=0.1,
    )

    Sigma, _, _ = build_fock(
        jnp.asarray(density[..., None, None], dtype=jnp.complex64),
        h=kernel.h,
        VR=kernel._VR_shifted,
        refP=kernel.refP,
        HH=kernel.HH,
        w2d=kernel.w2d,
        include_exchange=True,
        include_hartree=False,
        exchange_hermitian_channel_packing=(
            kernel.exchange_hermitian_channel_packing
        ),
    )

    expected_fft_order = np.empty((2, 2), dtype=np.float32)
    for kx in range(2):
        for ky in range(2):
            expected_fft_order[kx, ky] = -sum(
                weights[qx, qy]
                * Vq[(kx - qx) % 2, (ky - qy) % 2]
                * density[qx, qy]
                for qx in range(2)
                for qy in range(2)
            )
    expected = np.fft.ifftshift(expected_fft_order, axes=(0, 1))

    np.testing.assert_allclose(
        np.asarray(Sigma[..., 0, 0]).real,
        expected,
        atol=2e-5,
        rtol=2e-6,
    )


def test_float32_entropy_is_finite_and_exact_at_endpoints():
    p = jnp.array([[[0.0, 1.0]]], dtype=jnp.float32)
    weights = jnp.ones((1, 1), dtype=jnp.float32)

    entropy = jax.jit(occupation_entropy)(p, weights)

    assert jnp.isfinite(entropy)
    assert float(entropy) == 0.0
