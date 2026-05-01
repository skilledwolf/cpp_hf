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
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_CACHE_ROOT = Path(tempfile.gettempdir()) / "cpp_hf_optional_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

# Ensure src/ is importable when running standalone.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
for optional_src in (
    REPO_ROOT.parent / "contimod" / "src",
    REPO_ROOT.parent / "contimod_graphene" / "src",
):
    if optional_src.exists() and str(optional_src) not in sys.path:
        sys.path.insert(0, str(optional_src))

import numpy as np

import contimod as cm
from contimod.meanfield.init_guess import init_to_density_matrix
from contimod.utils.spectrum_fermi import FermiParams

import cpp_hf
from cpp_hf import SCFConfig, solve_scf
from cpp_hf.symmetry import make_project_fn

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


def _flip_k(
    A: np.ndarray,
    k_convention: str,
    flip_axes: tuple[int, ...],
) -> np.ndarray:
    if k_convention == "flip":
        return np.flip(A, axis=flip_axes)
    if k_convention == "mod":
        nk1, nk2 = A.shape[0], A.shape[1]
        i = (
            (-np.arange(nk1, dtype=np.int32)) % nk1
            if 0 in flip_axes
            else np.arange(nk1, dtype=np.int32)
        )
        j = (
            (-np.arange(nk2, dtype=np.int32)) % nk2
            if 1 in flip_axes
            else np.arange(nk2, dtype=np.int32)
        )
        return A[i[:, None], j[None, :], ...]
    raise ValueError(f"k_convention must be 'mod' or 'flip', got {k_convention!r}")


def _make_time_reversal_U(s2: np.ndarray, v1: np.ndarray) -> np.ndarray:
    return np.asarray(v1) @ (1j * np.asarray(s2))


def _make_pm_group(
    identity: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
    spin_elems = [identity, s1, s2, s3]
    valley_elems = [identity, v3]
    return np.stack([S @ V for S in spin_elems for V in valley_elems], axis=0)


def _make_svp_project_fn(
    *,
    s3: np.ndarray,
    v3: np.ndarray,
    n_orb: int,
    outlier_sv: tuple[int, int] = (+1, +1),
    k_convention: str = "flip",
    k_flip_axes: tuple[int, ...] = (0,),
):
    s3_np = np.asarray(s3)
    v3_np = np.asarray(v3)
    nb = s3_np.shape[0]
    n_blocks = nb // n_orb
    so, vo = float(outlier_sv[0]), float(outlier_sv[1])

    idx_outlier = None
    idx_same_v = None
    idx_other_v: list[int] = []

    for i in range(n_blocks):
        s_val = float(np.sign(np.real(s3_np[i * n_orb, i * n_orb])))
        v_val = float(np.sign(np.real(v3_np[i * n_orb, i * n_orb])))
        if s_val == so and v_val == vo:
            idx_outlier = i
        elif v_val == vo:
            idx_same_v = i
        else:
            idx_other_v.append(i)

    if idx_outlier is None or idx_same_v is None or len(idx_other_v) != 2:
        raise ValueError(
            "Could not identify 4 spin-valley blocks from s3/v3 "
            f"(n_orb={n_orb}, nb={nb}, outlier_sv={outlier_sv})"
        )

    def _sl(i: int) -> slice:
        return slice(i * n_orb, (i + 1) * n_orb)

    sl_same = _sl(idx_same_v)
    sl_ov0 = _sl(idx_other_v[0])
    sl_ov1 = _sl(idx_other_v[1])

    mask = np.zeros((nb, nb), dtype=np.float64)
    for i in range(n_blocks):
        a, b = i * n_orb, (i + 1) * n_orb
        mask[a:b, a:b] = 1.0

    k_conv = str(k_convention)
    k_axes = tuple(k_flip_axes)

    def project(P: np.ndarray) -> np.ndarray:
        P_arr = np.asarray(P)
        out = np.array(P_arr * mask, copy=True)

        P_same = P_arr[..., sl_same, sl_same]
        P_ov0_flip = _flip_k(P_arr[..., sl_ov0, sl_ov0], k_conv, k_axes)
        P_ov1_flip = _flip_k(P_arr[..., sl_ov1, sl_ov1], k_conv, k_axes)

        Q = (P_same + P_ov0_flip + P_ov1_flip) / 3.0
        Q_flip = _flip_k(Q, k_conv, k_axes)

        out[..., sl_same, sl_same] = Q
        out[..., sl_ov0, sl_ov0] = Q_flip
        out[..., sl_ov1, sl_ov1] = Q_flip
        return out

    return project


def build_problem():
    """Set up bilayer graphene HF problem via contimod."""
    model = cm.graphene.MultilayerAB(valleyful=True, spinful=True, U=float(U_MEV))
    h_template = model.discretize(nk=NK, kmax=KMAX)
    h_template.fermi = FermiParams(T=TEMPERATURE, mu=0.0)

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
    U_tr = _make_time_reversal_U(s2, v1)

    projector_sv = 0.25 * (identity_op + s3) @ (identity_op + v3)
    sv_contrast = -projector_sv + 3.0 * (identity_op - projector_sv)

    seeds = {
        "PM": h_template.get_operator("zero"),
        "SVP": -float(INIT_SCALE) * h_template.get_operator(sv_contrast),
    }

    G_pm = _make_pm_group(identity_op, s1, s2, s3, v3)
    project_fn_pm = make_project_fn(
        unitary_group=G_pm,
        time_reversal_U=np.asarray(U_tr),
        time_reversal_k_convention="flip",
    )
    project_fn_svp = _make_svp_project_fn(
        s3=np.asarray(s3),
        v3=np.asarray(v3),
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
    kernel = cpp_hf.HartreeFockKernel(
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
    result = solve_scf(kernel, np.asarray(P0), n_electrons,
                                  config=config)
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
                "n_iter": result.iterations,
                "converged": conv,
                "n_electrons": n_e,
                "mu": float(result.chemical_potential),
            }

            status = "OK" if conv else "FAIL"
            print(f"  n={n_cm12:+.3f}  it={result.iterations:3d}  E={E:.6f}  {status}")

    # Save reference data
    output = Path(__file__).parent / "data" / "bilayer_reference.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "nk": NK,
        "kmax": KMAX,
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
