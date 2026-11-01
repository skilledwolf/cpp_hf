"""Bilayer graphene energies against the saved SCF reference.

Three densities (hole side, near neutrality, electron side) on both branches.
SCF runs native RDIIS (the reference was made with linear mixing, so this
also checks that RDIIS reaches the same fixed point) and direct minimization
runs on JAX (native when JAX is missing); backend agreement itself is covered
by test_backend_parity.

Needs contimod.  Reference data come from
``tests/generate_bilayer_reference.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

contimod = pytest.importorskip("contimod")
cg_symmetry = pytest.importorskip("contimod.materials.graphene.symmetry")

import contimod as cm  # noqa: E402
from contimod.kdiscrete import kMeshgrid2D  # noqa: E402
from contimod.meanfield import SelfEnergySeed  # noqa: E402
from contimod.utils.spectrum_fermi import FermiParams  # noqa: E402

import hartreefock as hf  # noqa: E402
from hartreefock.symmetry import make_project_fn  # noqa: E402

DATA_PATH = Path(__file__).parent / "data" / "bilayer_reference.npz"
PER_CM = 0.246e-7
E_ATOL = 1e-4
DENSITIES = [-0.60, -0.12, 0.25]
DIRECT_BACKEND = "jax" if "jax" in hf.available_backends() else "native"


@pytest.fixture(scope="module")
def bilayer_setup():
    ref = np.load(DATA_PATH, allow_pickle=False)
    nk, kmax = int(ref["nk"]), float(ref["kmax"])
    temperature = float(ref["temperature"])

    # Replay the saved endpoint quadrature (contimod now uses midpoint cells).
    mesh = kMeshgrid2D(kmax, nk)
    mesh.kx = np.array(ref["kx"], copy=True)
    mesh.ky = np.array(ref["ky"], copy=True)
    mesh.ks = np.stack(np.meshgrid(mesh.kx, mesh.ky, indexing="ij"), axis=-1)
    mesh.dk = np.array([mesh.kx[1] - mesh.kx[0], mesh.ky[1] - mesh.ky[0]])
    mesh.dk_sq = float(np.prod(mesh.dk))
    mesh.weight = float(ref["weight"])
    model = cm.graphene.Bernal(2, valleyful=True, spinful=True, U=float(ref["U_meV"]))
    h = model.discretize(ks=mesh, fermi=FermiParams(T=temperature, mu=0.0))

    Vq = cm.coulomb.dualgate_coulomb(
        h.kmesh.distance_grid, epsilon_r=float(ref["epsilon_r"]), d_gate=float(ref["d_gate"])
    )
    kernel = hf.HartreeFockKernel(
        np.asarray(h.kmesh.weights), np.asarray(h.hs), np.asarray(Vq)[..., None, None],
        T=temperature,
    )

    identity = np.asarray(model.identity)
    s1, s2, s3 = (np.asarray(model.spin_op(i)) for i in (1, 2, 3))
    v1, v3 = np.asarray(model.valley_op(1)), np.asarray(model.valley_op(3))
    projector_sv = 0.25 * (identity + s3) @ (identity + v3)
    sv_contrast = -projector_sv + 3.0 * (identity - projector_sv)
    seeds = {
        "PM": h.get_operator("zero"),
        "SVP": -float(ref["init_scale"]) * h.get_operator(sv_contrast),
    }
    project_fns = {
        "PM": make_project_fn(
            unitary_group=cg_symmetry.make_pm_group(identity, s1, s2, s3, v3),
            time_reversal_U=cg_symmetry.make_time_reversal_U(s2, v1),
            time_reversal_k_convention="flip",
        ),
        "SVP": cg_symmetry.make_svp_project_fn(
            s3=s3, v3=v3, n_orb=4, outlier_sv=(+1, +1), k_convention="flip", k_flip_axes=(0,)
        ),
    }
    ne_cn = float(h.state.density_from_filling(0.5 - 1e-6))
    return ref, h, kernel, ne_cn, seeds, project_fns, temperature


def _initial_state(setup, branch, n_cm12):
    _, h, _, ne_cn, seeds, _, temperature = setup
    h_run = h.copy()
    h_run.fermi = FermiParams(T=temperature, mu=0.0)
    h_run = h_run.with_chemical_potential(density=float(ne_cn + n_cm12 * 1e12 * PER_CM**2))
    n_e = float(h_run.state.density() / float(h_run.degeneracy))
    if branch == "PM":  # the zero seed is the empty density
        P0 = np.zeros_like(np.asarray(h_run.hs))
    else:
        P0 = SelfEnergySeed(seeds[branch]).to_density(h_run, h_run.state.density(), T=temperature)
    return np.asarray(P0), n_e


def _reference_energy(ref, branch, n_cm12):
    key = f"{branch}_{n_cm12:+.2f}"
    if not bool(ref[f"{key}/converged"]):
        pytest.skip(f"reference point {key} did not converge")
    return float(ref[f"{key}/energy"])


@pytest.mark.parametrize("branch", ["PM", "SVP"])
@pytest.mark.parametrize("n_cm12", DENSITIES)
def test_scf_energy_matches_reference(bilayer_setup, branch, n_cm12):
    ref, _, kernel, *_ = bilayer_setup
    E_ref = _reference_energy(ref, branch, n_cm12)
    P0, n_e = _initial_state(bilayer_setup, branch, n_cm12)
    config = hf.SCFConfig(
        acceleration="rdiis", max_iter=500, density_tol=1e-7, comm_tol=1e-6,
        project_fn=bilayer_setup[5][branch],
    )
    result = hf.solve_scf(kernel, P0, n_e, config=config, backend="native")

    np.testing.assert_allclose(float(result.energy), E_ref, atol=E_ATOL)
    if not result.converged:
        last = result.n_iter - 1
        assert float(result.history["density_residual"][last]) < 3e-7, result.message
        assert float(result.history["commutator_residual"][last]) < 1e-6, result.message


@pytest.mark.parametrize("branch", ["PM", "SVP"])
@pytest.mark.parametrize("n_cm12", DENSITIES)
def test_direct_solver_energy_matches_reference(bilayer_setup, branch, n_cm12):
    # Direct minimization reaches the SCF energy even where the gradient
    # stop has not formally fired, so only the energy is checked.
    ref, _, kernel, *_ = bilayer_setup
    E_ref = _reference_energy(ref, branch, n_cm12)
    P0, n_e = _initial_state(bilayer_setup, branch, n_cm12)
    config = hf.SolverConfig(max_iter=200, tol_E=1e-7, project_fn=bilayer_setup[5][branch])
    result = hf.solve(kernel, P0, n_e, config=config, backend=DIRECT_BACKEND)

    np.testing.assert_allclose(float(result.energy), E_ref, atol=E_ATOL)
