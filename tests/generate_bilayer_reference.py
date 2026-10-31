#!/usr/bin/env python
"""Generate bilayer graphene density scan reference data using SCF solver.

Runs a standard SCF solver on a bilayer graphene model (via contimod) at a
small set of density points for PM and SVP branches, then saves converged
energies as a .npz file for regression testing.

Usage:
    python tests/generate_bilayer_reference.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force CPU for reproducibility.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import jax
import jax.numpy as jnp

import contimod as cm
import contimod_graphene.symmetry as cg_symmetry
from contimod.meanfield.init_guess import init_to_density_matrix
from contimod.utils.spectrum_fermi import FermiParams

import hartreefock

from hartreefock import SCFConfig
from hartreefock.backends.jax import solve_scf
from hartreefock.backends.jax.symmetry import make_project_fn

# ---------------------------------------------------------------------------
# Physical parameters (matching debug/repro_solver_ordering.py)
# ---------------------------------------------------------------------------
NK = 49
KMAX = 0.14
U_MEV = 40.0
TEMPERATURE = 0.5
EPSILON_R = 10.0
D_GATE = 40.0
INIT_SCALE = 45.0
PER_CM = 0.246e-7

# Density points: a small representative set (in 1e12 cm^-2).
# Negative = hole-doped.  Includes CN reference (0.0).
DENSITY_POINTS_CM12 = (-0.60, -0.42, -0.25, -0.12, -0.05, 0.05, 0.12, 0.25)
BRANCHES = ("PM", "SVP")


def build_problem():
    """Set up bilayer graphene HF problem via contimod."""
    model = cm.graphene.Bernal(2, valleyful=True, spinful=True, U=float(U_MEV))
    h_template = model.discretize(
        nk=NK, kmax=KMAX, fermi=FermiParams(T=TEMPERATURE, mu=0.0),
    )

    ne_cn = float(h_template.state.compute_density_from_filling(0.5 - 1e-6))
    weights = np.asarray(h_template.kmesh.weights)

    Vq = cm.coulomb.dualgate_coulomb(
        h_template.kmesh.distance_grid,
        epsilon_r=EPSILON_R,
        d_gate=D_GATE,
    )
    Vq = np.asarray(Vq.magnitude)[..., None, None]

    # Symmetry operators
    identity_op = np.asarray(model.identity)
    s1, s2, s3 = [np.asarray(model.spin_op(i)) for i in (1, 2, 3)]
    v1, v3 = np.asarray(model.valley_op(1)), np.asarray(model.valley_op(3))
    U_tr = cg_symmetry.make_time_reversal_U(s2, v1)

    projector_sv = 0.25 * (identity_op + s3) @ (identity_op + v3)
    sv_contrast = -projector_sv + 3.0 * (identity_op - projector_sv)

    seeds = {
        "PM": h_template.get_operator("zero"),
        "SVP": -float(INIT_SCALE) * h_template.get_operator(sv_contrast),
    }

    G_pm = cg_symmetry.make_pm_group(identity_op, s1, s2, s3, v3)
    project_fn_pm = make_project_fn(
        unitary_group=G_pm,
        time_reversal_U=jnp.asarray(U_tr),
        time_reversal_k_convention="flip",
    )
    project_fn_svp = cg_symmetry.make_svp_project_fn(
        s3=jnp.asarray(s3),
        v3=jnp.asarray(v3),
        n_orb=4,
        outlier_sv=(+1, +1),
        k_convention="flip",
        k_flip_axes=(0,),
    )

    project_fns = {"PM": project_fn_pm, "SVP": project_fn_svp}

    return h_template, ne_cn, weights, Vq, seeds, project_fns


def init_density_from_seed(h_run, seed_op):
    P0 = init_to_density_matrix(
        h_run,
        seed_op,
        density=None,
        T=float(h_run.fermi.T),
        init_kind="auto",
    )
    return np.asarray(P0)


def solve_scf_point(h_template, weights, Vq, n_electrons, P0, project_fn):
    """Solve one HF point with the reference SCF solver."""
    kernel = hartreefock.HartreeFockKernel(
        weights=weights,
        hamiltonian=np.asarray(h_template.hs),
        coulomb_q=Vq,
        T=TEMPERATURE,
        include_hartree=False,
        include_exchange=True,
    )
    config = SCFConfig(
        max_iter=500,
        mixing=0.2,
        density_tol=1e-7,
        comm_tol=1e-6,
        project_fn=project_fn,
    )
    result = solve_scf(kernel, jnp.asarray(P0), n_electrons,
                                  config=config)
    jax.block_until_ready(result.energy)
    return result


def main():
    h_template, ne_cn, weights, Vq, seeds, project_fns = build_problem()

    results = {}

    for branch in BRANCHES:
        print(f"\n=== {branch} ===")
        for n_cm12 in DENSITY_POINTS_CM12:
            dd = n_cm12 * 1e12 * (PER_CM ** 2)
            h_run = h_template.copy()
            h_run.fermi = FermiParams(T=TEMPERATURE, mu=0.0)
            h_run.compute_chemicalpotential(density=float(ne_cn + dd))
            n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))

            P0 = init_density_from_seed(h_run, seeds[branch])
            result = solve_scf_point(
                h_template, weights, Vq, n_e, P0, project_fns[branch],
            )

            E = float(result.energy)
            conv = result.converged

            key = f"{branch}_{n_cm12:+.2f}"
            results[key] = {
                "energy": E,
                "n_iter": result.n_iter,
                "converged": conv,
                "n_electrons": n_e,
                "mu": float(result.mu),
            }

            status = "OK" if conv else "FAIL"
            print(f"  n={n_cm12:+.3f}  it={result.n_iter:3d}  E={E:.6f}  {status}")

    # Save reference data
    output = Path(__file__).parent / "data" / "bilayer_reference.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "nk": NK,
        "kmax": KMAX,
        "kx": np.asarray(h_template.kmesh.kx),
        "ky": np.asarray(h_template.kmesh.ky),
        "weight": float(h_template.kmesh.weight),
        "U_meV": U_MEV,
        "temperature": TEMPERATURE,
        "epsilon_r": EPSILON_R,
        "d_gate": D_GATE,
        "init_scale": INIT_SCALE,
        "density_points_cm12": np.array(DENSITY_POINTS_CM12),
        "branches": np.array(list(BRANCHES)),
    }
    for key, vals in results.items():
        for vk, vv in vals.items():
            save_dict[f"{key}/{vk}"] = np.asarray(vv)

    np.savez(str(output), **save_dict)
    print(f"\nSaved reference to {output}")

    # Verify all converged
    all_conv = all(v["converged"] for v in results.values())
    if not all_conv:
        print("\nWARNING: not all points converged!")
        return 1
    print("\nAll points converged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
