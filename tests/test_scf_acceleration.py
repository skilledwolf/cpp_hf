"""Tests for SCF acceleration schemes (linear / DIIS / ODA) and trust_radius.

DIIS (Pulay commutator extrapolation) and ODA (optimal damping) mirror
cpp_hf's solver_scf.hpp.  DIIS converges in far fewer iterations than linear
mixing; ODA is a robust energy-monotone line search (it reaches the same
energy, but as an energy method it does not drive the commutator as hard, so
its convergence is asserted on the energy).  These run in the suite's default
float32.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from _meshes import mesh_for

from hartreefock import HartreeFockKernel, SolverConfig, solve


def _comm_rms(F, P, w2d):
    comm = F @ P - P @ F
    per_k = jnp.sum(jnp.abs(comm) ** 2, axis=(-2, -1))
    return float(jnp.sqrt(jnp.sum(w2d * per_k) / jnp.maximum(jnp.sum(w2d), 1e-30)))


def _dirac(nk=4, m=0.3, v=1.0, T=0.1, ex=0.7):
    ks = (np.arange(nk) - nk // 2) * (2 * np.pi / nk)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    h = np.zeros((nk, nk, 2, 2), np.complex64)
    h[..., 0, 0] = m
    h[..., 1, 1] = -m
    h[..., 0, 1] = v * (KX - 1j * KY)
    h[..., 1, 0] = v * (KX + 1j * KY)
    return HartreeFockKernel(mesh_for(jnp.ones((nk, nk), jnp.float32)), jnp.asarray(h), jnp.full((nk, nk, 1, 1), ex, jnp.complex64), T=T)


_NE = 16.0


class TestAccelerationAgreement:
    def test_schemes_agree_on_energy(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        energies = {}
        for acc in ("linear", "oda", "rdiis"):
            r = solve(K, P0, _NE, config=SolverConfig(
                max_iter=800, density_tol=1e-6, comm_tol=1e-5,
                mixing=0.5, method=acc))
            energies[acc] = r.energy
        spread = max(energies.values()) - min(energies.values())
        assert spread < 1e-3, f"schemes disagree on energy: {energies}"


class TestRDIIS:
    def test_converges_far_faster_than_linear(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        cfg = dict(max_iter=800, density_tol=1e-6, comm_tol=1e-5, mixing=0.5)
        lin = solve(K, P0, _NE, config=SolverConfig(method="linear", **cfg))
        diis = solve(K, P0, _NE, config=SolverConfig(method="rdiis", **cfg))
        assert bool(lin.converged) and bool(diis.converged)
        # RDIIS should need far fewer Roothaan steps than plain mixing.
        assert int(diis.n_iter) < int(lin.n_iter) // 3
        np.testing.assert_allclose(float(diis.energy), float(lin.energy), atol=1e-4, rtol=1e-5)
        assert _comm_rms(diis.fock, diis.density, K.w2d) < 1e-4

    def test_trust_radius_converges(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        free = solve(K, P0, _NE, config=SolverConfig(
            max_iter=800, density_tol=1e-6, comm_tol=1e-5, method="rdiis"))
        clipped = solve(K, P0, _NE, config=SolverConfig(
            max_iter=800, density_tol=1e-6, comm_tol=1e-5, method="rdiis",
            trust_radius=0.5))
        assert bool(clipped.converged)
        np.testing.assert_allclose(float(clipped.energy), float(free.energy), atol=1e-4, rtol=1e-5)

    def test_damping_still_converges(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        r = solve(K, P0, _NE, config=SolverConfig(
            max_iter=800, density_tol=1e-6, comm_tol=1e-5,
            method="rdiis", diis_damping=0.7))
        assert r.converged


class TestODA:
    def test_energy_matches_and_is_monotone(self):
        K = _dirac()
        P0 = jnp.zeros_like(K.h)
        cfg = dict(max_iter=800, density_tol=1e-6, comm_tol=1e-5, mixing=0.5)
        lin = solve(K, P0, _NE, config=SolverConfig(method="linear", **cfg))
        oda = solve(K, P0, _NE, config=SolverConfig(method="oda", **cfg))
        # ODA reaches the same energy (its guarantee is energy descent)...
        np.testing.assert_allclose(float(oda.energy), float(lin.energy), atol=1e-3, rtol=1e-5)
        # ...and the energy history is non-increasing (robust line search).
        hO = np.asarray(oda.history["free_energy"])
        assert hO.size > 1
        assert np.all(np.diff(hO) <= 1e-4), "ODA free energy not monotonically decreasing"

    def test_finite_temperature_oda_optimizes_free_energy(self):
        h = jnp.array([[[[-1.0, 0.0], [0.0, 1.0]]]], dtype=jnp.complex64)
        K = HartreeFockKernel(mesh_for(jnp.ones((1, 1), dtype=jnp.float32)), h, jnp.zeros((1, 1, 1, 1), dtype=jnp.complex64), T=1.0)
        P0 = jnp.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=jnp.complex64)
        result = solve(
            K, P0, 1.0,
            config=SolverConfig(
                method="oda", max_iter=20,
                density_tol=1e-6, comm_tol=1e-6,
            ),
        )
        assert result.converged
        expected = np.diag([0.731059, 0.268941])
        np.testing.assert_allclose(
            np.asarray(result.density[0, 0]), expected, atol=2e-5,
        )


class TestSolverConfigValidation:
    def test_invalid_acceleration_raises(self):
        with pytest.raises(ValueError, match="method"):
            SolverConfig(method="bfgs")

    def test_nonpositive_diis_size_raises(self):
        with pytest.raises(ValueError, match="diis_size"):
            SolverConfig(diis_size=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_iter": 1.5},
            {"density_tol": np.nan},
            {"comm_tol": np.inf},
            {"mixing": -0.1},
            {"diis_start": True},
            {"level_shift": np.nan},
        ],
    )
    def test_invalid_domains_raise(self, kwargs):
        with pytest.raises(ValueError):
            SolverConfig(**kwargs)

    def test_scf_validates_count_shape_and_temperature_before_jit(self):
        K = _dirac(nk=2)
        with pytest.raises(ValueError, match="physically reachable range"):
            solve(K, jnp.zeros_like(K.h), 100.0)
        with pytest.raises(ValueError, match="P0 must have shape"):
            solve(K, jnp.zeros((2, 2, 3, 3)), 1.0)
        with pytest.raises(ValueError, match="non-negative"):
            _dirac(nk=2, T=-0.1)

    @pytest.mark.parametrize("n_electrons", [0.0, 2.0])
    def test_scf_exact_empty_and_full_fillings(self, n_electrons):
        h = jnp.array([[[[-1.0, 0.0], [0.0, 1.0]]]], dtype=jnp.complex64)
        K = HartreeFockKernel(mesh_for(jnp.ones((1, 1), dtype=jnp.float32)), h, jnp.zeros((1, 1, 1, 1), dtype=jnp.complex64), T=0.1)
        result = solve(K, jnp.zeros_like(h), n_electrons)
        assert result.converged
        expected = np.eye(2) * (n_electrons / 2.0)
        np.testing.assert_array_equal(
            np.asarray(result.density[0, 0]), expected,
        )


@pytest.mark.parametrize("level_shift,expected_solves", [(0.0, 1), (0.2, 2)])
def test_density_from_fock_only_resolves_mu_for_nonzero_shift(
    monkeypatch, level_shift, expected_solves,
):
    import jax

    from hartreefock._jax import scf

    calls = []
    original = scf.find_chemical_potential

    def counted_solve(*args, **kwargs):
        # Count actual executions inside JIT branches, not tracing calls.
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(scf, "find_chemical_potential", counted_solve)
    fock = jnp.diag(jnp.array([-.3, .4]))[None, None].astype(jnp.complex64)
    evaluate = jax.jit(lambda shift: scf._density_from_fock(
        fock, weights=jnp.ones((1, 1)), n_electrons=1.0,
        temperature=.2, level_shift=shift,
    ))
    density, _ = evaluate(jnp.asarray(level_shift))
    density.block_until_ready()
    jax.effects_barrier()
    assert len(calls) == expected_solves
    np.testing.assert_allclose(jnp.trace(density[0, 0]), 1.0, atol=1e-6)
