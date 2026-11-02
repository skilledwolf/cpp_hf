"""Focused regression tests for Fock/energy numerical conventions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hartreefock._jax.fock import build_fock, hf_energy, occupation_entropy


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


def test_float32_entropy_is_finite_and_exact_at_endpoints():
    p = jnp.array([[[0.0, 1.0]]], dtype=jnp.float32)
    weights = jnp.ones((1, 1), dtype=jnp.float32)

    entropy = jax.jit(occupation_entropy)(p, weights)

    assert jnp.isfinite(entropy)
    assert float(entropy) == 0.0
