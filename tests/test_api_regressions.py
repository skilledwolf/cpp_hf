"""Regressions: exchange weighting, embedding, validation, chemical potential."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from _meshes import mesh_for
from scipy.special import expit

from hartreefock import (
    HartreeFockKernel,
    Mesh,
    SolverConfig,
    fock_operators,
    solve,
)
from hartreefock._jax.utils import find_chemical_potential as _find_mu


def find_chemical_potential(bands, weights, n_electrons, T):
    return float(_find_mu(bands, weights, n_electrons, T))


def _exchange_expected(weights, interaction, density):
    nkx, nky = weights.shape
    expected_fft_order = np.empty((nkx, nky), dtype=float)
    for kx in range(nkx):
        for ky in range(nky):
            expected_fft_order[kx, ky] = -sum(
                weights[qx, qy]
                * interaction[(kx - qx) % nkx, (ky - qy) % nky]
                * density[qx, qy]
                for qx in range(nkx)
                for qy in range(nky)
            )
    return np.fft.ifftshift(expected_fft_order, axes=(0, 1))


def _exchange_problem():
    weights = np.full((2, 2), 0.25)
    interaction = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    density = np.arange(1, 5, dtype=np.float64).reshape(2, 2)
    h = np.zeros((2, 2, 1, 1), dtype=np.complex128)
    kernel = HartreeFockKernel(mesh_for(weights), h, interaction[..., None, None], T=0.1)
    P = density[..., None, None].astype(np.complex128)
    expected = _exchange_expected(weights, interaction, density)
    return kernel, P, expected


def test_exchange_weights_the_integrated_density():
    kernel, P, expected = _exchange_problem()
    F, _ = fock_operators(kernel).evaluate(P)
    np.testing.assert_allclose(
        F[..., 0, 0].real,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 1.5},
        {"tol_E": np.nan},
        {"tol_grad": np.inf},
        {"cg_restart": 0},
        {"bt_shrink": 1.0},
        {"mu_maxiter": 0},
        {"method": "bfgs"},
        {"block_sizes": (1.5, 0.5)},
        {"project_fn": 3},
    ],
)
def test_solver_config_rejects_invalid_domains(kwargs):
    with pytest.raises(ValueError):
        SolverConfig(**{"method": "cg", **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 1.5},
        {"density_tol": np.nan},
        {"comm_tol": np.inf},
        {"mixing": -0.1},
        {"method": "bfgs"},
        {"diis_size": 0},
        {"diis_start": True},
        {"diis_damping": 1.1},
        {"trust_radius": -0.1},
        {"project_fn": 3},
        {"method": "rdiis", "safeguard": 1},
    ],
)
def test_scf_config_rejects_invalid_domains(kwargs):
    with pytest.raises(ValueError):
        SolverConfig(**kwargs)


@pytest.fixture
def two_band_kernel():
    h = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    return HartreeFockKernel(mesh_for(np.full((2, 2), 0.25)), h, np.zeros((2, 2, 1, 1)), T=0.1)


@pytest.mark.parametrize("method", ["cg", "rdiis"])
def test_solvers_reject_wrong_initial_density_shape(two_band_kernel, method):
    with pytest.raises(ValueError, match="P0 must have shape"):
        solve(two_band_kernel, np.zeros((2, 2, 3, 3)), 1.0, config=SolverConfig(method=method))


@pytest.mark.parametrize("method", ["cg", "rdiis"])
def test_solvers_reject_unreachable_electron_count(two_band_kernel, method):
    with pytest.raises(ValueError, match="physically reachable range"):
        solve(two_band_kernel, np.zeros_like(two_band_kernel.h), 100.0, config=SolverConfig(method=method))


def test_chemical_potential_exact_empty_and_full_endpoints():
    bands = np.array([[[0.0, 1.0]]], dtype=np.float64)
    weights = np.ones((1, 1), dtype=np.float64)
    empty = find_chemical_potential(bands, weights, 0.0, 0.1)
    full = find_chemical_potential(bands, weights, 2.0, 0.1)
    assert np.isneginf(empty)
    assert np.isposinf(full)


def test_float64_bisection_default_reaches_double_precision():
    bands = np.array([[[-0.31, 0.17, 1.24]]], dtype=np.float64)
    weights = np.ones((1, 1), dtype=np.float64)
    target = 1.37
    temperature = 0.19
    mu = find_chemical_potential(bands, weights, target, temperature)
    count = float(np.sum(weights[..., None] * expit((mu - bands) / temperature)))
    assert abs(count - target) < 1e-12


@pytest.mark.parametrize("target", [1e-25, 1e-6, 2.0 - 1e-6])
@pytest.mark.parametrize("bandwidth", [0.0, 0.2])
def test_chemical_potential_bounds_cover_low_and_high_fillings(target, bandwidth):
    bands = np.array([[[0.0, bandwidth]]])
    T = 0.1
    mu = find_chemical_potential(bands, np.ones((1, 1)), target, T)
    # Count holes directly near full filling so the relative error is visible.
    if target > 1.0:
        count, expected = np.sum(expit((bands - mu) / T)), 2.0 - target
    else:
        count, expected = np.sum(expit((mu - bands) / T)), target
    np.testing.assert_allclose(count, expected, rtol=2e-8, atol=0.0)


@pytest.mark.parametrize("acceleration", ["linear", "oda", "rdiis"])
def test_zero_temperature_scf_preserves_flat_band_particle_count(acceleration):
    h = np.zeros((1, 1, 2, 2), complex)
    kernel = HartreeFockKernel(mesh_for(np.ones((1, 1))), h, np.zeros((1, 1, 1, 1)), T=0.0)
    config = SolverConfig(
        method=acceleration, max_iter=4, mixing=1.0, density_tol=1e-12, comm_tol=1e-12
    )
    result = solve(kernel, np.eye(2)[None, None] * 0.05, 0.1, config=config)
    assert result.converged
    np.testing.assert_allclose(np.trace(result.density[0, 0]), 0.1, atol=1e-14)


@pytest.mark.parametrize("kind", ["regular", "superlattice"])
@pytest.mark.parametrize("density_scale", [0.6, 1.4])
def test_embedded_kernel_reports_standard_hf(kind, density_scale):
    """Away from the reference density too, the embedded Fock matrix and energy are standard."""
    weights = np.full((2, 2), 0.3 / 4)
    kx = np.linspace(-0.5, 0.5, 2, endpoint=False)
    h = np.diag([-0.5, 0.5])[None, None] + kx[:, None, None, None] * np.array([[0, 1], [1, 0]])
    h = np.ascontiguousarray(np.broadcast_to(h, (2, 2, 2, 2)), dtype=complex)
    HH = np.array([[0.5, -0.1], [-0.1, 0.5]])
    refP = np.zeros_like(h)
    refP[..., 0, 0], refP[..., 1, 1] = 0.4, 0.6
    mesh = mesh_for(weights)
    if kind == "regular":
        common = dict(V=np.full((2, 2, 1, 1), 0.3, complex), hartree=HH)
    else:
        mesh = replace(mesh, periodic=True)
        common = dict(V=lambda q: np.full_like(q, 0.3), plane_waves=np.zeros((1, 2)),
                      hartree=HH[None, None])
    common.update(mesh=mesh, h=h, T=0.05)
    embedded = HartreeFockKernel(**common, reference=refP, embed_reference=True)
    standard = HartreeFockKernel(**common, reference=np.zeros_like(refP))
    P = density_scale * refP

    F_standard, E_standard = fock_operators(standard).evaluate(P)
    F_embedded, E_embedded = fock_operators(embedded).evaluate(P)
    np.testing.assert_allclose(E_embedded, E_standard, atol=1e-12)
    np.testing.assert_allclose(F_embedded, F_standard, atol=1e-12)

    # The energy parts sum to the reported energy, with and without a reference.
    referenced = HartreeFockKernel(**common, reference=refP)
    for kernel in (embedded, standard, referenced):
        E = fock_operators(kernel).evaluate(P)[1]
        np.testing.assert_allclose(sum(kernel.energies(P).values()), E, atol=1e-12)
    np.testing.assert_allclose(embedded.energies(P)["kinetic"],
                               standard.energies(P)["kinetic"], atol=1e-12)


def test_noninteracting_start():
    """P0=None starts from the filled bands of h; a field biases the start."""
    mesh = Mesh.square(2.0, 16)
    eps = np.sum(mesh.ks**2, axis=-1)
    kernel = HartreeFockKernel(mesh, eps[..., None, None] * np.eye(2), lambda q: np.full_like(q, 0.5), T=0.05)
    n = 1 / (2 * np.pi)
    P = kernel.noninteracting_density(n)
    np.testing.assert_allclose(np.trace(kernel.mean_density(P)).real, n, rtol=1e-10)
    up, down = np.diag(kernel.mean_density(kernel.noninteracting_density(n, field=np.diag([-1.0, 1.0])))).real
    assert up > 10 * down
    result = solve(kernel, None, n)
    assert result.converged is True and type(result.energy) is float
    assert type(result.density) is np.ndarray
    np.testing.assert_allclose(result.energy, float(solve(kernel, P, n).energy), atol=1e-8)
