"""Finite-temperature solver stationarity and objective regressions."""

from __future__ import annotations

import numpy as np
import pytest

from hartreefock import (
    HartreeFockKernel,
    SCFConfig,
    SolverConfig,
    solve,
    solve_scf,
)
from hartreefock._jax.utils import density_matrix_from_fock



def _repulsive_hartree(*, temperature: float = 0.5) -> HartreeFockKernel:
    h = np.zeros((1, 1, 2, 2), dtype=np.complex128)
    return HartreeFockKernel(
        weights=np.ones((1, 1), dtype=np.float64),
        hamiltonian=h,
        coulomb_q=np.zeros((1, 1, 1, 1), dtype=np.complex128),
        T=temperature,
        include_hartree=True,
        include_exchange=False,
        reference_density=np.zeros_like(h),
        hartree_matrix=np.eye(2, dtype=np.float64),
    )


def _fixed_point_residual(
    result, kernel: HartreeFockKernel, n_electrons: float
) -> float:
    density_fd, _ = density_matrix_from_fock(
        np.asarray(result.fock),
        np.asarray(kernel.w2d),
        n_electrons,
        float(kernel.T),
    )
    diff = np.asarray(result.density) - density_fd
    per_k = np.sum(np.abs(diff) ** 2, axis=(-2, -1))
    return float(
        np.sqrt(
            np.sum(np.asarray(kernel.w2d) * per_k)
            / np.sum(np.asarray(kernel.w2d))
        )
    )


@pytest.mark.parametrize(
    "optimizer,tol,atol",
    [
        ("cg", 2.0e-4, 2.0e-3),
        ("newton", 1.0e-8, 1.0e-7),
    ],
)
def test_zero_orbital_gradient_does_not_hide_wrong_occupations(
    optimizer: str, tol: float, atol: float
) -> None:
    kernel = _repulsive_hartree()
    initial = np.array(
        [[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.complex128
    )
    result = solve(
        kernel,
        initial,
        1.0,
        config=SolverConfig(
            optimizer=optimizer,
            max_iter=300,
            tol_grad=tol,
        ),
    )

    assert bool(result.converged)
    np.testing.assert_allclose(
        np.asarray(result.density[0, 0]), 0.5 * np.eye(2), atol=atol
    )
    assert _fixed_point_residual(result, kernel, 1.0) <= max(tol, atol)

    internal = np.asarray(result.history["E"])
    objective = np.asarray(result.history["free_energy"])
    assert internal.shape == objective.shape
    assert objective.size > 1
    assert np.all(np.diff(objective) <= 1.0e-10)
    assert np.max(np.abs(internal - objective)) > 1.0e-2


def test_finite_temperature_oda_descends_free_energy() -> None:
    h = np.array([[[[-1.0, 0.0], [0.0, 1.0]]]], dtype=np.complex128)
    kernel = HartreeFockKernel(
        weights=np.ones((1, 1), dtype=np.float64),
        hamiltonian=h,
        coulomb_q=np.zeros((1, 1, 1, 1), dtype=np.complex128),
        T=1.0,
        include_hartree=True,
        include_exchange=False,
        reference_density=np.zeros_like(h),
        hartree_matrix=np.zeros((2, 2), dtype=np.float64),
    )
    initial = np.array(
        [[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.complex128
    )
    result = solve_scf(
        kernel,
        initial,
        1.0,
        config=SCFConfig(
            acceleration="oda",
            max_iter=20,
            density_tol=1.0e-6,
            comm_tol=1.0e-6,
        ),
    )

    assert bool(result.converged)
    np.testing.assert_allclose(
        np.asarray(result.density[0, 0]),
        np.diag([0.73105858, 0.26894142]),
        atol=2.0e-6,
    )
    objective = np.asarray(result.history["free_energy"])
    assert objective.size > 1
    assert np.all(np.diff(objective) <= 1.0e-12)


def test_zero_diis_history_is_rejected_defensively() -> None:
    kernel = _repulsive_hartree()
    with pytest.raises(ValueError, match="diis_size"):
        solve_scf(
            kernel,
            np.zeros_like(kernel.h),
            1.0,
            config=SCFConfig(
                acceleration="rdiis",
                diis_size=0,
                max_iter=2,
            ),
        )


@pytest.mark.parametrize("acceleration", ["linear", "oda", "rdiis"])
def test_scf_returns_the_density_whose_stationarity_was_certified(acceleration):
    # The fixed-point map amplifies a density error by 500, so replacing the
    # certified state with its undamped Fermi density would invalidate the stop.
    h = np.zeros((1, 1, 2, 2), dtype=np.complex128)
    kernel = HartreeFockKernel(
        np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=0.5,
        include_hartree=True, include_exchange=False,
        reference_density=h, hartree_matrix=1000.0 * np.eye(2),
    )
    initial = np.diag([0.5 + 1e-10, 0.5 - 1e-10])[None, None]
    tolerance = 1e-7
    result = solve_scf(
        kernel, initial, 1.0,
        config=SCFConfig(acceleration=acceleration, max_iter=1,
                         density_tol=tolerance, comm_tol=tolerance),
    )
    assert result.converged
    density_fd, _ = density_matrix_from_fock(
        result.fock, kernel.w2d, 1.0, kernel.T,
    )
    residual = np.linalg.norm(result.density - density_fd)
    assert residual <= tolerance
    np.testing.assert_allclose(
        result.history["density_residual"][-1], residual, rtol=1e-6, atol=1e-12,
    )


@pytest.mark.parametrize("n_electrons", [0.0, 2.0])
def test_free_energy_has_exact_zero_entropy_at_filling_endpoints(
    n_electrons: float,
) -> None:
    kernel = _repulsive_hartree(temperature=0.7)
    result = solve(
        kernel,
        np.zeros_like(kernel.h),
        n_electrons,
        config=SolverConfig(max_iter=2, tol_grad=1.0e-12),
    )
    np.testing.assert_array_equal(
        np.asarray(result.history["free_energy"]),
        np.asarray(result.history["E"]),
    )


def _wide_band_kernel(nk: int = 6, nb: int = 4, T: float = 0.5, ex: float = 0.6):
    """Random multiband kernel whose linear-mixing SCF crawls along one slow mode."""
    rng = np.random.default_rng(0)
    h = np.zeros((nk, nk, nb, nb), dtype=np.complex128)
    h[..., np.arange(nb), np.arange(nb)] = np.linspace(-20.0, 20.0, nb)
    A = (rng.normal(scale=0.5, size=(nk, nk, nb, nb))
         + 1j * rng.normal(scale=0.5, size=(nk, nk, nb, nb)))
    h = h + 0.5 * (A + np.conj(A.swapaxes(-1, -2)))
    return HartreeFockKernel(
        weights=np.ones((nk, nk), dtype=np.float64) / (nk * nk),
        hamiltonian=h,
        coulomb_q=np.full((nk, nk, 1, 1), ex, dtype=np.complex128),
        T=T,
    )


def test_rdiis_engages_when_the_residual_history_is_nearly_rank_deficient() -> None:
    """DIIS must accelerate an iteration dominated by a single slow mode.

    Successive commutators are then nearly parallel, so the Pulay matrix is
    numerically rank deficient.  Rejecting such solves on a pivot threshold
    (the old ``FullPivLU::isInvertible`` gate) silently degraded DIIS to
    linear mixing: 69 iterations here against 95 for plain mixing, and 100+
    on the bilayer once an exact symmetry projector removed the rounding noise
    that had occasionally let a solve through.  With the normalised
    partial-pivot solve, DIIS converges in about ten.
    """
    kernel = _wide_band_kernel()
    n_e = 1.5 * float(kernel.weight_sum)
    common = dict(max_iter=200, density_tol=1e-9, comm_tol=1e-8, mixing=0.2)
    linear = solve_scf(kernel, np.zeros_like(kernel.h), n_e,
                       config=SCFConfig(acceleration="linear", **common))
    diis = solve_scf(kernel, np.zeros_like(kernel.h), n_e,
                     config=SCFConfig(acceleration="rdiis", **common))
    assert bool(linear.converged) and bool(diis.converged)
    assert int(linear.n_iter) > 60
    assert int(diis.n_iter) < 25, int(diis.n_iter)
    np.testing.assert_allclose(float(diis.energy), float(linear.energy), atol=1e-7)

