import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hartreefock._jax.utils import (
    density_matrix_from_fock,
    fermidirac,
    find_chemical_potential,
    selfenergy_fft,
)


def test_find_chemical_potential_hits_target_density():
    # Simple two-level system with a single k-point.
    bands = jnp.array([[[0.0, 1.0]]], dtype=jnp.float32)  # shape (1,1,2)
    weights = jnp.ones((1, 1), dtype=jnp.float32)
    mu = find_chemical_potential(bands, weights, n_electrons=1.0, T=0.1)
    occ = fermidirac(bands - mu, 0.1)
    total = float(jnp.sum(weights[..., None] * occ))
    assert abs(total - 1.0) < 1e-4


def test_find_chemical_potential_rejects_unreachable_density():
    bands = jnp.array([[[0.0]]], dtype=jnp.float32)
    weights = jnp.ones((1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="physically reachable range"):
        find_chemical_potential(bands, weights, n_electrons=2.0, T=0.1)


def test_density_matrix_from_fock_rejects_unreachable_density():
    F = jnp.array([[[[0.0]]]], dtype=jnp.complex64)
    weights = jnp.ones((1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="physically reachable range"):
        density_matrix_from_fock(F, weights, n_electrons=2.0, T=0.1)


def test_selfenergy_fft_single_point_matches_definition():
    # For a single k-point grid, FFT/ifft reduce to identity, so Σ = -VR * P.
    VR = jnp.array([[[[0.3]]]], dtype=jnp.complex64)  # shape (1,1,1,1)
    P = jnp.array([[[[1.0 + 0.5j, 0.2], [0.2, 0.8 - 0.1j]]]], dtype=jnp.complex64)
    sigma = selfenergy_fft(VR, P)
    expected = -VR * P
    np.testing.assert_allclose(np.array(sigma), np.array(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_find_chemical_potential_hits_target_multiband(method):
    """Both solvers hit target density on a multi-k, multi-band system."""
    rng = np.random.RandomState(42)
    nk1, nk2, nb = 4, 4, 6
    bands = jnp.array(rng.randn(nk1, nk2, nb).astype(np.float32))
    weights = jnp.ones((nk1, nk2), dtype=jnp.float32) / (nk1 * nk2)
    n_target = 3.0
    T = 0.05

    mu = find_chemical_potential(bands, weights, n_target, T, method=method)
    occ = fermidirac(bands - mu, T)
    total = float(jnp.sum(weights[..., None] * occ))
    assert abs(total - n_target) < 1e-4


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_find_chemical_potential_cold_limit(method):
    """At low T, the chemical-potential solver must still find the correct mu."""
    bands = jnp.array([[[0.0, 1.0, 2.0]]])
    weights = jnp.ones((1, 1))

    mu = find_chemical_potential(bands, weights, n_electrons=2.0, T=1e-5, method=method)
    occ = fermidirac(bands - mu, 1e-5)
    total = float(jnp.sum(weights[..., None] * occ))
    assert abs(total - 2.0) < 1e-4


@pytest.mark.parametrize(("n_electrons", "expected_mu"), [(0.0, -jnp.inf), (3.0, jnp.inf)])
def test_find_chemical_potential_exact_endpoint_fillings(n_electrons, expected_mu):
    bands = jnp.array([[[0.0, 1.0, 2.0]]])
    weights = jnp.ones((1, 1))
    mu = find_chemical_potential(bands, weights, n_electrons, T=0.1)
    assert mu == expected_mu
    occupations = fermidirac(bands - mu, 0.1)
    np.testing.assert_array_equal(
        np.asarray(occupations),
        np.full((1, 1, 3), n_electrons / 3.0),
    )


def test_fermi_utilities_reject_negative_temperature():
    with pytest.raises(ValueError, match="non-negative"):
        fermidirac(jnp.array([0.0]), -0.1)
    with pytest.raises(ValueError, match="non-negative"):
        find_chemical_potential(
            jnp.array([[[0.0, 1.0]]]), jnp.ones((1, 1)), 1.0, -0.1,
        )


@pytest.mark.parametrize("method", ["bisection", "newton"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("target", [1e-25, 1e-6, 2.0 - 1e-3])
@pytest.mark.parametrize("bandwidth", [0.0, 0.2])
def test_jitted_chemical_bounds_cover_dilute_and_full_fillings(
    method, dtype, target, bandwidth, enable_x64,
):
    with enable_x64(dtype == jnp.float64):
        bands = jnp.asarray([[[0.0, bandwidth]]], dtype=dtype)
        weights = jnp.ones((1, 1), dtype=dtype)
        target = jnp.asarray(target, dtype=dtype)
        find_mu = jax.jit(lambda eps, w, n: find_chemical_potential(eps, w, n, 0.1, method=method))
        mu = find_mu(bands, weights, target)
        assert mu.dtype == dtype
        if target > 1.0:
            count = jnp.sum(fermidirac(mu - bands, 0.1))
            expected = 2.0 - target
        else:
            count = jnp.sum(fermidirac(bands - mu, 0.1))
            expected = target
        tolerance = 3e-4 if dtype == jnp.float32 else 2e-10
        np.testing.assert_allclose(count, expected, rtol=tolerance, atol=0.0)


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_jitted_chemical_solver_masks_exact_endpoints(method):
    bands = jnp.zeros((1, 1, 2))
    weights = jnp.ones((1, 1))
    find_mu = jax.jit(jax.vmap(lambda n: find_chemical_potential(bands, weights, n, 0.1, method=method)))
    np.testing.assert_array_equal(find_mu(jnp.array([0.0, 2.0])), [-np.inf, np.inf])
