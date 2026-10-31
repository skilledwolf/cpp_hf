import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hartreefock import HartreeFockKernel, SCFConfig
from hartreefock.backends.jax import solve_minimum

jax.config.update("jax_enable_x64", True)


def saddle_kernel():
    return HartreeFockKernel(np.ones((1, 1)), np.zeros((1, 1, 2, 2), complex),
                            np.ones((1, 1, 1, 1)), T=.1)


def test_recovery_escapes_exact_degenerate_paramagnetic_saddle():
    result = solve_minimum(saddle_kernel(), np.eye(2)[None, None]/2, 1.)
    assert result.converged and result.stability.stable
    assert result.escapes >= 1 and result.total_iterations > result.solution.n_iter
    assert result.fock_builds > result.solution.history["fock_builds"]
    assert np.all(np.diff(result.free_energies) < 0)
    p = np.linalg.eigvalsh(result.solution.density)
    assert p.max() > .99 and p.min() < .01
    np.testing.assert_allclose(p.sum(), 1., atol=1e-13)


def test_zero_escape_budget_reports_saddle_without_false_success():
    result = solve_minimum(saddle_kernel(), np.eye(2)[None, None]/2, 1., max_escapes=0)
    assert result.solution.converged and not result.converged and result.stability.stable is False
    assert result.escapes == 0


def test_recovery_rejects_conflicting_symmetry_constraint():
    with pytest.raises(ValueError, match="unconstrained"):
        solve_minimum(saddle_kernel(), np.eye(2)[None, None]/2, 1.,
                      config=SCFConfig(project_fn=lambda P: P))


def test_nonstationary_iterate_is_not_reported_as_a_minimum():
    h = np.diag([-1., 1.]).astype(complex)[None, None]
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.full((1, 1, 1, 1), .1), T=.1)
    result = solve_minimum(kernel, np.eye(2)[None, None]/2, 1.,
        config=SCFConfig(acceleration="rdiis", max_iter=1, density_tol=1e-8, comm_tol=1e-8))
    assert not result.converged and result.stability is None
    assert result.escapes == 0 and result.total_iterations == 1


def test_near_saddle_local_stall_can_propose_a_validated_escape(monkeypatch):
    from dataclasses import replace
    import hartreefock.backends.jax.minima as module
    actual_solve = module.solve_scf
    calls = []
    def first_local_budget_is_short(*args, config, **kwargs):
        local_config = replace(config, max_iter=1) if not calls else config
        result = actual_solve(*args, config=local_config, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(module, "solve_scf", first_local_budget_is_short)
    initial = np.diag([.5+2e-8, .5-2e-8])[None, None]
    result = solve_minimum(saddle_kernel(), initial, 1.)
    assert not calls[0].converged
    residual = max(float(calls[0].history["density_residual"][-1]),
                   float(calls[0].history["commutator_residual"][-1]))
    assert 1e-8 < residual < 1e-6
    assert result.converged and result.stability.stable and result.escapes >= 1
    assert np.all(np.diff(result.free_energies) < 0)


def test_float32_escape_requires_descent_above_kernel_precision(monkeypatch):
    from hartreefock.backends.jax.reference_scf import SCFResult
    from hartreefock.backends.jax.stability import StabilityResult
    import hartreefock.backends.jax.minima as module

    h = np.zeros((1, 1, 2, 2), np.complex64)
    kernel = HartreeFockKernel(np.ones((1, 1), np.float32), h,
                              np.zeros((1, 1, 1, 1), np.float32), T=1e-9)
    density = np.eye(2, dtype=np.complex64)[None, None]/2
    local = SCFResult(density, h, 0., 0., 1, True, "test stationary state", {
        "free_energy": np.array([0.]), "density_residual": np.array([0.]),
        "commutator_residual": np.array([0.]), "fock_builds": 1,
    })
    direction = np.diag([1., -1.]).astype(complex)[None, None]
    saddle = StabilityResult(-1., 0., True, True, False, 0., 0., 1, direction)
    monkeypatch.setattr(module, "solve_scf", lambda *args, **kwargs: local)
    monkeypatch.setattr(module, "stability_analysis", lambda *args, **kwargs: saddle)
    evaluated = []

    def precision_limited_trial(P, **kwargs):
        evaluated.append(P.dtype)
        return jnp.zeros_like(P), jnp.asarray(-1e-7, dtype=jnp.float32)

    monkeypatch.setattr(module, "_evaluate_fock", precision_limited_trial)
    result = solve_minimum(kernel, density, 1., eigen_tol=1e-4,
        config=SCFConfig(acceleration="rdiis", density_tol=1e-4, comm_tol=1e-4))
    assert not result.converged and result.escapes == 0
    # Both the Fermi retraction and projected fallback fail the significant
    # descent test. Neither may accept float64-only progress.
    assert len(evaluated) == 24 and all(dtype == jnp.complex64 for dtype in evaluated)
    assert "no finite feasible descent" in result.message


def test_fermi_escape_preserves_saturated_coupled_tangent_and_energy_curvature(monkeypatch):
    from scipy.special import expit, xlogy
    from hartreefock.backends.jax import stability_analysis
    from hartreefock.backends.jax.minima import _fermi_escape_density
    from hartreefock.backends.jax import stability as stability_module
    from hartreefock.backends.jax.stability import (
        _evaluate_fock, _fock_kwargs, _host_fermi_density, _response_fock,
    )

    # The outer bands are numerically saturated, while the inner occupations
    # respond to a common chemical potential across nonuniform k weights.
    # A rank-one interaction couples their populations and outer-band coherence.
    w = np.array([[.25, .75]])
    h = np.broadcast_to(np.diag([-20., -.01, .015, 20.]), (1, 2, 4, 4)).astype(complex).copy()
    h[0, 1, 1, 1] = -.02
    temperature = .05
    number = float(np.sum(w[..., None]*expit(-np.diagonal(h, axis1=-2, axis2=-1).real/temperature)))
    P = _host_fermi_density(h, w, number, temperature)[0]
    observable = np.zeros_like(h)
    observable[..., 0, 3] = observable[..., 3, 0] = 1.
    observable[..., 1, 1], observable[..., 2, 2] = .15, -.15
    kernel = HartreeFockKernel(w, h, np.zeros((1, 2, 1, 1)),
                              T=temperature, reference_density=P)
    def interaction(strength):
        def custom(P, *, h, refP, w2d, **kwargs):
            value = jnp.sum(w2d[..., None, None]
                            * (jnp.conj(jnp.asarray(observable))*(P-refP)).real)
            sigma = -strength*jnp.asarray(observable)*value
            return sigma, jnp.zeros_like(P), h+sigma
        return custom
    monkeypatch.setattr(stability_module, "fock_builder", lambda _: interaction(1.))
    unit = stability_analysis(kernel, P, number)
    fock_fn = interaction(1.05/(1-unit.lowest_eigenvalue))
    monkeypatch.setattr(stability_module, "fock_builder", lambda _: fock_fn)
    stability = stability_analysis(kernel, P, number)
    np.testing.assert_allclose(stability.lowest_eigenvalue, -.05, atol=1e-12)
    assert stability.stationary and stability.stable is False
    direction = stability.direction
    norm = np.sqrt(np.sum(w*np.sum(abs(direction)**2, axis=(-2, -1)))/w.sum())
    direction = direction/norm
    args = _fock_kwargs(kernel)
    response = np.asarray(_response_fock(jnp.asarray(P), jnp.asarray(direction), **args))
    response = response/(1-stability.lowest_eigenvalue)
    def trial(step):
        return _fermi_escape_density(h, response, w, number, temperature, step)
    def omega(density):
        E = float(_evaluate_fock(jnp.asarray(density), **args)[1])
        p = np.clip(np.linalg.eigvalsh(density), 0., 1.)
        return E+temperature*np.sum(w[..., None]*(xlogy(p, p)+xlogy(1-p, 1-p)))
    step = 1e-4
    plus, minus = trial(step), trial(-step)
    np.testing.assert_allclose((plus-minus)/(2*step), direction, atol=1e-8)
    curvature = (omega(plus)+omega(minus)-2*omega(P))/step**2
    np.testing.assert_allclose(curvature, stability.lowest_eigenvalue/norm**2,
                               rtol=1e-4, atol=1e-6)
    for step in (-.25, .25):
        candidate = trial(step)
        spectrum = np.linalg.eigvalsh(candidate)
        assert spectrum.min() >= -1e-14 and spectrum.max() <= 1+1e-14
        np.testing.assert_allclose(np.sum(w[..., None]*spectrum), number, atol=1e-13)
        np.testing.assert_allclose(candidate, candidate.conj().swapaxes(-2, -1), atol=1e-14)
    assert omega(trial(.25)) < omega(P)-1e-7


def test_fermi_retraction_escapes_without_using_density_projection(monkeypatch):
    import hartreefock.backends.jax.minima as module
    def forbidden_projection(*args, **kwargs):
        raise AssertionError("Fermi trials already provide a feasible descent")
    monkeypatch.setattr(module, "_feasible_density", forbidden_projection)
    response_calls = []
    original_response = module._response_fock
    def counted_response(*args, **kwargs):
        response_calls.append(None)
        return original_response(*args, **kwargs)
    monkeypatch.setattr(module, "_response_fock", counted_response)
    result = solve_minimum(saddle_kernel(), np.eye(2)[None, None]/2, 1.)
    assert result.converged and result.stability.stable and result.escapes == 1
    assert len(response_calls) == 1
    assert np.all(np.diff(result.free_energies) < 0)


def test_projected_density_fallback_remains_available_when_fermi_trials_stall(monkeypatch):
    import hartreefock.backends.jax.minima as module
    P = np.eye(2)[None, None]/2
    fermi_calls = []
    def unresolved_fermi_step(*args, **kwargs):
        fermi_calls.append(None)
        return P.copy()
    monkeypatch.setattr(module, "_fermi_escape_density", unresolved_fermi_step)
    result = solve_minimum(saddle_kernel(), P, 1.)
    assert len(fermi_calls) == 12
    assert result.converged and result.stability.stable and result.escapes == 1
    assert np.all(np.diff(result.free_energies) < 0)


def test_stale_legacy_history_cannot_make_an_unchanged_trial_look_like_descent(monkeypatch):
    from dataclasses import replace
    from scipy.special import xlogy
    import hartreefock.backends.jax.minima as module
    kernel = saddle_kernel()
    P = np.eye(2)[None, None]/2
    actual_solve = module.solve_scf
    local_results = []
    def stale_history_solve(*args, **kwargs):
        result = actual_solve(*args, **kwargs)
        history = dict(result.history)
        history["free_energy"] = np.asarray(history["free_energy"])+1.
        result = replace(result, history=history)
        local_results.append(result)
        return result
    monkeypatch.setattr(module, "solve_scf", stale_history_solve)
    # The saddle diagnosis is real. Neither escape construction resolves any
    # motion, so neither may pass the actual finite-step descent safeguard.
    monkeypatch.setattr(module, "_fermi_escape_density", lambda *args, **kwargs: P.copy())
    monkeypatch.setattr(module, "_feasible_density", lambda *args, **kwargs: P.copy())
    result = solve_minimum(kernel, P, 1., config=SCFConfig(acceleration="diis"))
    assert len(local_results) == 1
    assert result.solution.converged and result.stability.stable is False
    assert not result.converged and result.escapes == 0
    assert "no finite feasible descent" in result.message
    p = np.linalg.eigvalsh(P)
    actual_omega = float(result.solution.energy)+kernel.T*np.sum(
        np.asarray(kernel.w2d)[..., None]*(xlogy(p, p)+xlogy(1-p, 1-p)))
    np.testing.assert_allclose(result.free_energies, [actual_omega], atol=1e-14)
    assert abs(float(result.solution.history["free_energy"][-1])-actual_omega) > .9
