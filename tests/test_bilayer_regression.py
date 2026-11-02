"""Bilayer graphene energies against the saved SCF reference.

Three densities (hole side, near neutrality, electron side) on both branches.
SCF runs RDIIS (the reference was made with linear mixing, so this also
checks that RDIIS reaches the same fixed point); direct minimization runs CG.

Needs contimod.  Reference data come from
``tests/generate_bilayer_reference.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

contimod = pytest.importorskip("contimod")
cg_symmetry = pytest.importorskip("contimod.materials.graphene.symmetry")

import contimod as cm
from contimod.meanfield import SelfEnergySeed
from contimod.utils import pauli

import hartreefock as hf
from hartreefock import Mesh
from hartreefock.symmetry import make_project_fn

DATA_PATH = Path(__file__).parent / "data" / "bilayer_reference.npz"
PER_CM = 0.246e-7
E_ATOL = 1e-4
DENSITIES = [-0.60, -0.12, 0.25]


@pytest.fixture(scope="module")
def bilayer_setup():
    ref = np.load(DATA_PATH, allow_pickle=False)
    nk, kmax = int(ref["nk"]), float(ref["kmax"])
    temperature = float(ref["temperature"])

    # The reference was saved on the endpoint grid linspace(-kmax, kmax, nk)
    # (odd nk).  That grid is the odd-nk square Mesh of half side
    # kmax * nk / (nk - 1): same points, same weight.
    mesh = Mesh.square(kmax * nk / (nk - 1), nk)
    np.testing.assert_allclose(mesh.kx, ref["kx"], atol=1e-12)
    np.testing.assert_allclose(mesh.weight, float(ref["weight"]), rtol=1e-12)
    model = cm.graphene.Bernal(2, valley="conserved", spin="conserved", U=float(ref["U_meV"]))
    h = model.discretize(mesh, T=temperature)

    def V(q):
        return cm.coulomb.dualgate_coulomb(
            q, epsilon_r=float(ref["epsilon_r"]), d_gate=float(ref["d_gate"])
        )

    kernel = hf.HartreeFockKernel(h.kmesh, h.matrices, V, T=temperature)

    basis = h.basis
    identity = np.eye(basis.dim)
    s1, s2, s3 = (basis.op("spin", pauli[i]) for i in (1, 2, 3))
    v1, v3 = (basis.op("valley", pauli[i]) for i in (1, 3))
    projector_sv = 0.25 * (identity + s3) @ (identity + v3)
    sv_contrast = -projector_sv + 3.0 * (identity - projector_sv)
    seeds = {
        "PM": np.zeros_like(identity),
        "SVP": -float(ref["init_scale"]) * sv_contrast,
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
    h_run = h.with_chemical_potential(density=float(ne_cn + n_cm12 * 1e12 * PER_CM**2))
    n_e = float(h_run.state.density() / h_run.basis.multiplicity)
    if branch == "PM":  # the zero seed is the empty density
        P0 = np.zeros_like(h_run.matrices)
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
    config = hf.SolverConfig(
        method="rdiis", max_iter=500, density_tol=1e-7, comm_tol=1e-6,
        project_fn=bilayer_setup[5][branch],
    )
    result = hf.solve(kernel, P0, n_e, config=config)

    np.testing.assert_allclose(result.energy, E_ref, atol=E_ATOL)
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
    config = hf.SolverConfig(method="cg", max_iter=200, tol_E=1e-7, project_fn=bilayer_setup[5][branch])
    result = hf.solve(kernel, P0, n_e, config=config)

    np.testing.assert_allclose(result.energy, E_ref, atol=E_ATOL)
