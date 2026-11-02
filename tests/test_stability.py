"""Stability analysis and stability-checked minimization."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import xlogy

import hartreefock.stability as module
from hartreefock import (
    HartreeFockKernel,
    SCFConfig,
    fock_operators,
    solve_minimum,
    solve_scf,
    stability_analysis,
)




def saddle_kernel():
    """P = I/2 is exactly self-consistent, but exchange > 4T makes it a saddle."""
    return HartreeFockKernel(
        np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex), np.ones((1, 1, 1, 1)), T=0.1
    )


HALF = np.eye(2, dtype=complex)[None, None] / 2


def omega(kernel, P):
    """Actual free energy E[P] - T S[P] through the Fock build."""
    E = fock_operators(kernel).evaluate(P)[1]
    p = np.clip(np.linalg.eigvalsh(P), 0, 1)
    w = np.asarray(kernel.w2d)[..., None]
    return E + kernel.T * np.sum(w * (xlogy(p, p) + xlogy(1 - p, 1 - p)))


# ----- stability_analysis -------------------------------------------------------


def test_degenerate_paramagnet_has_negative_exchange_curvature():
    s = stability_analysis(saddle_kernel(), HALF, 1.0)
    assert s.stationary and s.eigen_converged and s.stable is False
    np.testing.assert_allclose(s.lowest_eigenvalue, -1.5, atol=1e-10)
    np.testing.assert_allclose(np.trace(s.direction[0, 0]), 0.0, atol=1e-14)


def test_contact_response_matches_its_explicit_tangent():
    # For two flavours the identity contact has tangent response -g D; its
    # scalar Hartree part is removed by the fixed-trace constraint.
    kernel = HartreeFockKernel(
        np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex), np.zeros((1, 1, 1, 1)), T=0.1,
        contact_terms=[(1.0, np.eye(2), np.eye(2))],
    )
    s = stability_analysis(kernel, HALF, 1.0)
    np.testing.assert_allclose(s.lowest_eigenvalue, -1.5, atol=1e-10)


def test_repulsive_hartree_minimum_has_positive_curvature():
    h = np.diag([-0.2, 0.3]).astype(complex)[None, None]
    kernel = HartreeFockKernel(
        np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=0.5, include_exchange=False,
        include_hartree=True, reference_density=np.zeros_like(h), hartree_matrix=3 * np.eye(2),
    )
    config = SCFConfig(acceleration="rdiis", density_tol=1e-9, comm_tol=1e-9)
    r = solve_scf(kernel, HALF, 1.0, config=config)
    s = stability_analysis(kernel, np.asarray(r.density), 1.0)
    assert s.stationary and s.eigen_converged and s.stable is True
    assert s.lowest_eigenvalue > 0


def test_curvature_matches_free_energy_second_difference():
    w = np.array([[0.2, 0.8]])
    h = np.broadcast_to(np.array([[-0.2, 0.1j], [-0.1j, 0.3]]), (1, 2, 2, 2)).copy()
    kernel = HartreeFockKernel(w, h, np.full((1, 2, 1, 1), 0.2), T=0.4)
    config = SCFConfig(acceleration="rdiis", density_tol=1e-11, comm_tol=1e-11)
    r = solve_scf(kernel, np.zeros_like(h), 0.8, config=config)
    P = np.asarray(r.density)
    s = stability_analysis(kernel, P, 0.8)
    assert s.stable
    D = s.direction
    np.testing.assert_allclose(np.sum(w * np.trace(D, axis1=-2, axis2=-1)), 0, atol=1e-14)
    e = 1e-4
    curvature = (
        omega(kernel, P + e * D) + omega(kernel, P - e * D)
        - 2 * omega(kernel, P)
    ) / (e * e * w.sum())
    np.testing.assert_allclose(curvature, s.lowest_eigenvalue, rtol=2e-6, atol=1e-7)


def test_nonstationary_density_does_not_receive_stability_certificate():
    h = np.diag([-1.0, 1.0])[None, None].astype(complex)
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.full((1, 1, 1, 1), 0.1), T=0.2)
    s = stability_analysis(kernel, HALF, 1.0)
    assert not s.stationary and s.stable is None


def test_precision_limited_kernel_is_not_given_a_tight_certificate():
    kernel = HartreeFockKernel(
        np.ones((1, 1), np.float32), np.zeros((1, 1, 2, 2), np.complex64),
        np.zeros((1, 1, 1, 1), np.float32), T=0.1,
    )
    with pytest.raises(ValueError, match="precision"):
        stability_analysis(kernel, HALF, 1.0)


def _flat_nine_band_kernel():
    return HartreeFockKernel(
        np.ones((1, 1)), np.zeros((1, 1, 9, 9), complex), np.zeros((1, 1, 1, 1)), T=0.1
    )


def test_failed_iterative_eigenvalue_solve_has_no_certificate(monkeypatch):
    from scipy.sparse.linalg import ArpackNoConvergence

    def fail(*args, **kwargs):
        raise ArpackNoConvergence("forced nonconvergence", np.empty(0), np.empty((80, 0)))

    monkeypatch.setattr(module, "eigsh", fail)
    s = stability_analysis(_flat_nine_band_kernel(), np.eye(9)[None, None] / 2, 4.5)
    assert s.stationary and not s.eigen_converged and s.stable is None


def test_eigenvalue_requires_an_independently_verified_residual(monkeypatch):
    def wrong_eigenpair(operator, **kwargs):
        size = operator.shape[0]
        return np.array([2.0]), np.ones((size, 1)) / np.sqrt(size)

    monkeypatch.setattr(module, "eigsh", wrong_eigenpair)
    s = stability_analysis(_flat_nine_band_kernel(), np.eye(9)[None, None] / 2, 4.5)
    assert s.stationary and not s.eigen_converged and s.stable is None
    np.testing.assert_allclose(s.eigen_residual, 1.0, atol=1e-12)


@pytest.mark.parametrize("number", [1e-60, 1e-6, 2.0 - 1e-6])
def test_host_fermi_density_preserves_particle_count_in_extreme_tails(number):
    fock = np.zeros((1, 1, 2, 2), complex)
    density, _, _, occupations = module._host_fermi_density(fock, np.ones((1, 1)), number, 0.1)
    np.testing.assert_allclose(np.trace(density[0, 0]).real, number, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(occupations, number / 2, rtol=1e-13, atol=0.0)


# ----- solve_minimum ------------------------------------------------------------


def test_recovery_escapes_exact_degenerate_paramagnetic_saddle():
    r = solve_minimum(saddle_kernel(), HALF, 1.0)
    assert r.converged and r.stability.stable
    assert r.escapes >= 1 and r.total_iterations > r.solution.n_iter
    assert r.fock_builds > r.solution.history["fock_builds"]
    assert np.all(np.diff(r.free_energies) < 0)
    p = np.linalg.eigvalsh(np.asarray(r.solution.density))
    assert p.max() > 0.99 and p.min() < 0.01
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-13)


@pytest.mark.parametrize("number", [1e-6, 2.0 - 1e-6])
def test_flat_band_minimum_is_certified_near_empty_and_full_filling(number):
    kernel = HartreeFockKernel(
        np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex), np.zeros((1, 1, 1, 1)), T=0.1
    )
    r = solve_minimum(kernel, np.eye(2)[None, None] * (number / 2), number)
    assert r.converged and r.stability.stable and r.escapes == 0
    assert r.stability.density_residual < 1e-12


def test_fermi_escape_preserves_unstable_tangent_and_saturated_occupations():
    rng = np.random.default_rng(37)
    Q = np.linalg.qr(rng.normal(size=(1, 2, 4, 4)) + 1j * rng.normal(size=(1, 2, 4, 4)))[0]
    F = (Q * np.array([-4.0, -0.03, 0.03, 4.0])) @ Q.conj().swapaxes(-2, -1)
    w = np.array([[0.2, 0.8]])
    P = module._host_fermi_density(F, w, 2.0, 0.1)[0]
    kernel = HartreeFockKernel(w, F, np.ones((1, 2, 1, 1)), T=0.1, reference_density=P)
    s = stability_analysis(kernel, P, 2.0)
    assert s.stationary and s.stable is False
    ops = fock_operators(kernel)
    response = np.asarray(ops.response(P, s.direction)) / (1 - s.lowest_eigenvalue)

    def retract(step):
        return module._fermi_escape_density(F, response, w, 2.0, kernel.T, step)

    eps = 1e-5
    np.testing.assert_allclose(
        (retract(eps) - retract(-eps)) / (2 * eps), s.direction, rtol=2e-6, atol=1e-8
    )
    energies = []
    for step in (-0.1, 0.1):
        trial = retract(step)
        occupations = np.linalg.eigvalsh(trial)
        assert occupations.min() > -1e-13 and occupations.max() < 1 + 1e-13
        np.testing.assert_allclose(np.sum(w[..., None] * occupations), 2.0, atol=1e-13)
        energies.append(omega(kernel, trial))
    assert min(energies) < omega(kernel, P) - 1e-6


def test_zero_escape_budget_reports_saddle_without_false_success():
    r = solve_minimum(saddle_kernel(), HALF, 1.0, max_escapes=0)
    assert r.solution.converged and not r.converged and r.stability.stable is False
    assert r.escapes == 0


def test_recovery_rejects_conflicting_symmetry_constraint():
    with pytest.raises(ValueError, match="unconstrained"):
        solve_minimum(saddle_kernel(), HALF, 1.0, config=SCFConfig(project_fn=lambda P: P))


def test_nonstationary_iterate_is_not_reported_as_a_minimum():
    h = np.diag([-1.0, 1.0]).astype(complex)[None, None]
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.full((1, 1, 1, 1), 0.1), T=0.1)
    config = SCFConfig(acceleration="rdiis", max_iter=1, density_tol=1e-8, comm_tol=1e-8)
    r = solve_minimum(kernel, HALF, 1.0, config=config)
    assert not r.converged and r.stability is None
    assert r.escapes == 0 and r.total_iterations == 1


def test_near_saddle_local_stall_can_propose_a_validated_escape(monkeypatch):
    engine = module._jax
    actual_solve = engine.solve_scf
    calls = []

    def first_local_budget_is_short(*args, config, **kwargs):
        result = actual_solve(*args, config=replace(config, max_iter=1) if not calls else config)
        calls.append(result)
        return result

    monkeypatch.setattr(engine, "solve_scf", first_local_budget_is_short)
    r = solve_minimum(saddle_kernel(), np.diag([0.5 + 2e-8, 0.5 - 2e-8])[None, None], 1.0)
    residual = max(
        float(calls[0].history["density_residual"][-1]),
        float(calls[0].history["commutator_residual"][-1]),
    )
    assert not calls[0].converged and 1e-8 < residual < 1e-6
    assert r.converged and r.stability.stable and r.escapes >= 1
    assert np.all(np.diff(r.free_energies) < 0)


def test_fermi_retraction_escapes_without_using_density_projection(monkeypatch):
    def forbidden_projection(*args, **kwargs):
        raise AssertionError("Fermi trials already provide a feasible descent")

    monkeypatch.setattr(module, "_feasible_density", forbidden_projection)
    r = solve_minimum(saddle_kernel(), HALF, 1.0)
    assert r.converged and r.stability.stable and r.escapes == 1
    assert np.all(np.diff(r.free_energies) < 0)


def test_projected_density_fallback_remains_available_when_fermi_trials_stall(monkeypatch):
    fermi_calls = []

    def unresolved_fermi_step(*args, **kwargs):
        fermi_calls.append(None)
        return HALF.copy()

    monkeypatch.setattr(module, "_fermi_escape_density", unresolved_fermi_step)
    r = solve_minimum(saddle_kernel(), HALF, 1.0)
    assert len(fermi_calls) == 12
    assert r.converged and r.stability.stable and r.escapes == 1
    assert np.all(np.diff(r.free_energies) < 0)


def test_free_energy_describes_returned_state_not_stale_history(monkeypatch):
    engine = module._jax
    actual_solve = engine.solve_scf

    def stale_history(*args, **kwargs):
        result = actual_solve(*args, **kwargs)
        history = dict(result.history, free_energy=np.asarray(result.history["free_energy"]) + 1)
        return replace(result, history=history)

    monkeypatch.setattr(engine, "solve_scf", stale_history)
    # The saddle is real, but neither escape construction moves the density,
    # so neither may pass the finite-step descent safeguard.
    monkeypatch.setattr(module, "_fermi_escape_density", lambda *a, **k: HALF.copy())
    monkeypatch.setattr(module, "_feasible_density", lambda *a, **k: HALF.copy())
    kernel = saddle_kernel()
    r = solve_minimum(kernel, HALF, 1.0, config=SCFConfig(acceleration="diis"))
    assert r.solution.converged and r.stability.stable is False
    assert not r.converged and r.escapes == 0
    assert "no finite feasible descent" in r.message
    np.testing.assert_allclose(r.free_energies, [omega(kernel, HALF)], atol=1e-14)


def test_escape_requires_descent_above_kernel_precision(monkeypatch):
    """A float32 kernel must not accept float64-only progress."""
    engine = module._jax
    kernel = HartreeFockKernel(
        np.ones((1, 1), np.float32), np.zeros((1, 1, 2, 2), np.complex64),
        np.zeros((1, 1, 1, 1), np.float32), T=1e-9,
    )
    density = HALF.astype(np.complex64)
    local = engine.solve_scf(saddle_kernel(), HALF, 1.0)
    local = replace(local, density=density, fock=np.zeros_like(density), energy=0.0)
    direction = np.diag([1.0, -1.0]).astype(complex)[None, None]
    saddle = module.StabilityResult(-1.0, 0.0, True, True, False, 0.0, 0.0, 1, direction)
    evaluated = []

    def evaluate(P):
        evaluated.append(P.dtype)
        return np.zeros_like(P), -1e-7

    ops = SimpleNamespace(
        evaluate=evaluate, response=lambda P, dP: np.zeros_like(P), dtype=np.dtype(np.complex64)
    )
    monkeypatch.setattr(engine, "solve_scf", lambda *a, **k: local)
    monkeypatch.setattr(engine, "fock_operators", lambda kernel: ops)
    monkeypatch.setattr(module, "stability_analysis", lambda *a, **k: saddle)
    config = SCFConfig(acceleration="rdiis", density_tol=1e-4, comm_tol=1e-4)
    r = solve_minimum(kernel, density, 1.0, eigen_tol=1e-4, config=config)
    assert not r.converged and r.escapes == 0
    assert "no finite feasible descent" in r.message
    # Fermi retraction and projected fallback (2 × 6 × 2 trials) all stay float32.
    assert len(evaluated) == 24 and set(evaluated) == {np.dtype(np.complex64)}

