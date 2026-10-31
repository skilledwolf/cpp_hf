"""Focused regressions for the public Python/API parity contract."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from cpp_hf import (
    HartreeFockKernel,
    SCFConfig,
    SolverConfig,
    solve,
    solve_scf,
)
from cpp_hf._compat import _native
from cpp_hf.fock import build_fock, occupation_entropy
from cpp_hf.solver import _kernel_args_for_native
from cpp_hf.utils import fermidirac, find_chemical_potential


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


def _nonuniform_exchange_problem():
    weights = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    interaction = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    density = np.arange(1, 5, dtype=np.float64).reshape(2, 2)
    h = np.zeros((2, 2, 1, 1), dtype=np.complex128)
    kernel = HartreeFockKernel(
        weights=weights,
        hamiltonian=h,
        coulomb_q=interaction[..., None, None],
        T=0.1,
    )
    P = density[..., None, None].astype(np.complex128)
    expected = _exchange_expected(weights, interaction, density)
    return kernel, P, expected


@pytest.mark.parametrize(
    "weights,hamiltonian,interaction,match",
    [
        (
            np.ones((2,)),
            np.zeros((2, 2, 1, 1)),
            np.ones((2, 2, 1, 1)),
            "weights must have shape",
        ),
        (
            np.ones((2, 2)),
            np.zeros((2, 2, 1)),
            np.ones((2, 2, 1, 1)),
            "hamiltonian must have shape",
        ),
        (
            np.ones((2, 2)),
            np.zeros((2, 2, 1, 1)),
            np.ones((3, 2, 1, 1)),
            "coulomb_q must have shape",
        ),
    ],
)
def test_kernel_rejects_incompatible_core_shapes(
    weights, hamiltonian, interaction, match
):
    with pytest.raises(ValueError, match=match):
        HartreeFockKernel(
            weights,
            hamiltonian,
            interaction,
            T=0.1,
        )


def test_python_exchange_weights_the_integrated_density():
    kernel, P, expected = _nonuniform_exchange_problem()
    sigma, _, _ = build_fock(
        P,
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
        contact_g=kernel.contact_g,
        contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
    )
    np.testing.assert_allclose(
        sigma[..., 0, 0].real,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_native_exchange_weights_the_integrated_density():
    kernel, P, expected = _nonuniform_exchange_problem()
    sigma, _, _, _ = _native.build_fock_apply(
        _kernel_args_for_native(kernel),
        np.ascontiguousarray(P),
        None,
    )
    np.testing.assert_allclose(
        sigma[..., 0, 0].real,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_embedding_includes_contact_reference_potential_and_offset():
    weights = np.ones((1, 1), dtype=np.float64)
    h = np.zeros((1, 1, 2, 2), dtype=np.complex128)
    interaction = np.zeros((1, 1, 1, 1), dtype=np.float64)
    refP = np.zeros_like(h)
    refP[..., 0, 0] = 0.8
    refP[..., 1, 1] = 0.2
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    contact_terms = [(1.0, sigma_z, sigma_z)]

    plain = HartreeFockKernel(
        weights,
        h,
        interaction,
        T=0.1,
        reference_density=refP,
        contact_terms=contact_terms,
    )
    embedded = HartreeFockKernel(
        weights,
        h,
        interaction,
        T=0.1,
        reference_density=refP,
        contact_terms=contact_terms,
        embed_reference_potential=True,
        center_embedded_hartree=False,
    )

    expected_contact = np.diag([-0.2, -0.8]).astype(np.complex128)
    np.testing.assert_allclose(
        embedded.h - plain.h,
        expected_contact[None, None, ...],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(embedded.embedded_energy_offset, -0.16, atol=1e-12)


@pytest.mark.parametrize("temperature", [-0.1, np.inf, np.nan])
def test_kernel_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        HartreeFockKernel(
            np.ones((1, 1)),
            np.zeros((1, 1, 1, 1), dtype=np.complex128),
            np.zeros((1, 1, 1, 1)),
            T=temperature,
        )


def test_fermi_utilities_reject_negative_temperature():
    with pytest.raises(ValueError, match="non-negative"):
        fermidirac(np.array([0.0]), -0.1)
    with pytest.raises(ValueError, match="non-negative"):
        find_chemical_potential(
            np.array([[[0.0, 1.0]]]),
            np.ones((1, 1)),
            1.0,
            -0.1,
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
        {"optimizer": "bfgs"},
        {"block_sizes": (1.5, 0.5)},
        {"project_fn": 3},
    ],
)
def test_solver_config_rejects_invalid_domains(kwargs):
    with pytest.raises(ValueError):
        SolverConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 1.5},
        {"density_tol": np.nan},
        {"comm_tol": np.inf},
        {"mixing": -0.1},
        {"acceleration": "bfgs"},
        {"diis_size": 0},
        {"diis_start": True},
        {"diis_damping": 1.1},
        {"trust_radius": -0.1},
        {"block_sizes": (1, 0)},
        {"project_fn": 3},
    ],
)
def test_scf_config_rejects_invalid_domains(kwargs):
    with pytest.raises(ValueError):
        SCFConfig(**kwargs)


def test_solver_config_mu_iterations_match_jax_default():
    assert SolverConfig().mu_maxiter == 50


def test_config_positional_field_prefixes_match_jax():
    solver_names = tuple(field.name for field in fields(SolverConfig))
    assert solver_names[:10] == (
        "max_iter",
        "tol_E",
        "tol_grad",
        "denom_scale",
        "max_step",
        "cg_restart",
        "bt_shrink",
        "bt_max",
        "mu_maxiter",
        "plateau_window",
    )
    scf_names = tuple(field.name for field in fields(SCFConfig))
    assert scf_names[:11] == (
        "max_iter",
        "density_tol",
        "comm_tol",
        "mixing",
        "level_shift",
        "acceleration",
        "diis_size",
        "diis_start",
        "diis_damping",
        "trust_radius",
        "project_fn",
    )


@pytest.fixture
def two_band_kernel():
    h = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    return HartreeFockKernel(
        weights=np.full((2, 2), 0.25),
        hamiltonian=h,
        coulomb_q=np.zeros((2, 2, 1, 1)),
        T=0.1,
    )


@pytest.mark.parametrize("solver", [solve, solve_scf])
def test_solvers_reject_wrong_initial_density_shape(two_band_kernel, solver):
    with pytest.raises(ValueError, match="P0 must have shape"):
        solver(two_band_kernel, np.zeros((2, 2, 3, 3)), 1.0)


@pytest.mark.parametrize("solver", [solve, solve_scf])
def test_solvers_reject_unreachable_electron_count(two_band_kernel, solver):
    with pytest.raises(ValueError, match="physically reachable range"):
        solver(two_band_kernel, np.zeros_like(two_band_kernel.h), 100.0)


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_chemical_potential_exact_empty_and_full_endpoints(method):
    bands = np.array([[[0.0, 1.0]]], dtype=np.float64)
    weights = np.ones((1, 1), dtype=np.float64)
    empty = find_chemical_potential(bands, weights, 0.0, 0.1, method=method)
    full = find_chemical_potential(bands, weights, 2.0, 0.1, method=method)
    assert np.isneginf(empty)
    assert np.isposinf(full)


def test_float64_bisection_default_reaches_double_precision():
    bands = np.array([[[-0.31, 0.17, 1.24]]], dtype=np.float64)
    weights = np.ones((1, 1), dtype=np.float64)
    target = 1.37
    temperature = 0.19
    mu = find_chemical_potential(bands, weights, target, temperature)
    count = float(np.sum(weights[..., None] * fermidirac(bands - mu, temperature)))
    assert abs(count - target) < 1e-12


def test_newton_chemical_potential_is_robust_in_cold_gap():
    bands = np.array([[[0.0, 1.0, 2.0]]], dtype=np.float64)
    weights = np.ones((1, 1), dtype=np.float64)
    mu = find_chemical_potential(
        bands,
        weights,
        n_electrons=2.0,
        T=1e-5,
        method="newton",
    )
    count = float(np.sum(weights[..., None] * fermidirac(bands - mu, 1e-5)))
    assert abs(count - 2.0) < 1e-8


def test_float32_entropy_is_finite_and_exact_at_endpoints():
    p = np.array([[[0.0, 1.0]]], dtype=np.float32)
    weights = np.ones((1, 1), dtype=np.float32)
    entropy = occupation_entropy(p, weights)
    assert np.isfinite(entropy)
    assert float(entropy) == 0.0
