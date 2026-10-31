"""Bilayer graphene density scan against the saved SCF reference, per backend.

Needs contimod and contimod_graphene.  Reference data come from
``tests/generate_bilayer_reference.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

contimod = pytest.importorskip("contimod")
cg_symmetry = pytest.importorskip("contimod_graphene.symmetry")

import contimod as cm  # noqa: E402
from contimod.kdiscrete import kMeshgrid2D  # noqa: E402
from contimod.meanfield.init_guess import init_to_density_matrix  # noqa: E402
from contimod.utils.spectrum_fermi import FermiParams  # noqa: E402

import hartreefock as hf  # noqa: E402
from hartreefock.symmetry import make_project_fn  # noqa: E402

DATA_PATH = Path(__file__).parent / "data" / "bilayer_reference.npz"
PER_CM = 0.246e-7
E_ATOL = 1e-4
DENSITIES = [-0.60, -0.42, -0.25, -0.12, -0.05, 0.05, 0.12, 0.25]


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
        np.asarray(h.kmesh.weights), np.asarray(h.hs), np.asarray(Vq.magnitude)[..., None, None],
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
    ne_cn = float(h.state.compute_density_from_filling(0.5 - 1e-6))
    return ref, h, kernel, ne_cn, seeds, project_fns, temperature


def _initial_state(setup, branch, n_cm12):
    _, h, _, ne_cn, seeds, _, temperature = setup
    h_run = h.copy()
    h_run.fermi = FermiParams(T=temperature, mu=0.0)
    h_run.compute_chemicalpotential(density=float(ne_cn + n_cm12 * 1e12 * PER_CM**2))
    n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))
    P0 = init_to_density_matrix(h_run, seeds[branch], density=None, T=temperature, init_kind="auto")
    return np.asarray(P0), n_e


def _reference_energy(ref, branch, n_cm12):
    key = f"{branch}_{n_cm12:+.2f}"
    if not bool(ref[f"{key}/converged"]):
        pytest.skip(f"reference point {key} did not converge")
    return float(ref[f"{key}/energy"])


@pytest.mark.parametrize("backend", hf.available_backends())
@pytest.mark.parametrize("branch", ["PM", "SVP"])
@pytest.mark.parametrize("n_cm12", DENSITIES)
def test_scf_energy_matches_reference(bilayer_setup, backend, branch, n_cm12):
    ref, _, kernel, *_ = bilayer_setup
    E_ref = _reference_energy(ref, branch, n_cm12)
    P0, n_e = _initial_state(bilayer_setup, branch, n_cm12)
    config = hf.SCFConfig(
        max_iter=500, mixing=0.2, density_tol=1e-7, comm_tol=1e-6,
        project_fn=bilayer_setup[5][branch],
    )
    result = hf.solve_scf(kernel, P0, n_e, config=config, backend=backend)

    np.testing.assert_allclose(float(result.energy), E_ref, atol=E_ATOL)
    if not result.converged:
        last = result.n_iter - 1
        assert float(result.history["density_residual"][last]) < 3e-7, result.message
        assert float(result.history["commutator_residual"][last]) < 1e-6, result.message


@pytest.mark.parametrize("backend", hf.available_backends())
@pytest.mark.parametrize("branch", ["PM", "SVP"])
@pytest.mark.parametrize("n_cm12", DENSITIES)
def test_direct_solver_energy_matches_reference(bilayer_setup, backend, branch, n_cm12):
    # Direct minimization reaches the SCF energy even where the gradient
    # stop has not formally fired, so only the energy is checked.
    ref, _, kernel, *_ = bilayer_setup
    E_ref = _reference_energy(ref, branch, n_cm12)
    P0, n_e = _initial_state(bilayer_setup, branch, n_cm12)
    config = hf.SolverConfig(max_iter=200, tol_E=1e-7, project_fn=bilayer_setup[5][branch])
    result = hf.solve(kernel, P0, n_e, config=config, backend=backend)

    np.testing.assert_allclose(float(result.energy), E_ref, atol=E_ATOL)
