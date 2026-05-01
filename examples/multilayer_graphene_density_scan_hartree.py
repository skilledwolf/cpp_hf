#!/usr/bin/env python
"""Bilayer graphene density scan with Hartree + Fock (layer-resolved Coulomb).

Same 4 branches as the extended scan (PM, SVP, SP, SVP_flipped), but the
interaction kernel is layer-resolved via ``layer_coulomb_kernel`` and both
Hartree and exchange are included.

The Hartree contribution is computed relative to a self-consistent
charge-neutrality reference density (bootstrapped with a PM-projected SCF
at CN using the non-interacting CN density as the initial reference).

Writes a CSV and a figure of energy-per-carrier vs density for all branches.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


import numpy as np
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import cpp_hf
import _bilayer_common as common


DEFAULT_OUTPUT = REPO_ROOT / "examples" / "outputs" / "direct_minimization_density_scan_hartree.csv"
BRANCHES_EXTENDED = ("PM", "SVP", "SP", "SVP_flipped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branches", nargs="+", default=list(BRANCHES_EXTENDED),
                   choices=list(BRANCHES_EXTENDED))
    p.add_argument("--density-points", nargs="+", type=float,
                   default=list(common.DEFAULT_DENSITY_POINTS),
                   help="Carrier densities in 1e12 cm^-2")
    p.add_argument("--nk", type=int, default=common.NK)
    p.add_argument("--kmax", type=float, default=common.KMAX)
    p.add_argument("--u-mev", type=float, default=common.U_MEV)
    p.add_argument("--temperature", type=float, default=common.TEMPERATURE)
    p.add_argument("--epsilon-zz", type=float, default=common.EPSILON_ZZ,
                   help="Out-of-plane dielectric constant (hBN-like default)")
    p.add_argument("--epsilon-perp", type=float, default=common.EPSILON_PERP,
                   help="In-plane dielectric constant")
    p.add_argument("--lat-nm", type=float, default=common.LAT_NM,
                   help="Graphene lattice constant (nm)")
    p.add_argument("--layer-spacing-nm", type=float, default=common.LAYER_SPACING_NM,
                   help="Interlayer distance (nm)")
    p.add_argument("--init-scale", type=float, default=common.INIT_SCALE)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol-e", type=float, default=1e-7,
                   help="Energy-change tolerance")
    p.add_argument("--max-step", type=float, default=0.6,
                   help="Max orbital rotation norm per step")
    p.add_argument("--seed-mode", choices=("cold", "warm"), default="cold",
                   help="'cold' = fresh seed per density point (robust branch "
                        "separation). 'warm' = reuse previous converged density "
                        "(faster but drift-prone near phase boundaries).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args()


def run_scan(args: argparse.Namespace) -> list[dict]:
    setup = common.build_bilayer_layer_resolved(
        nk=args.nk, kmax=args.kmax, U_meV=args.u_mev,
        temperature=args.temperature,
        epsilon_zz=args.epsilon_zz, epsilon_perp=args.epsilon_perp,
        lat_nm=args.lat_nm, layer_spacing_nm=args.layer_spacing_nm,
        init_scale=args.init_scale,
    )

    # Bootstrap self-consistent CN reference density (PM, Hartree + exchange)
    print("Bootstrapping self-consistent CN reference density ...")
    t0 = time.perf_counter()
    reference_density = common.bootstrap_cn_reference_density(
        setup, temperature=args.temperature,
        max_iter=500, mixing=0.3, density_tol=1e-7, comm_tol=1e-6,
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights,
        hamiltonian=np.asarray(setup.h_template.hs),
        coulomb_q=setup.Vq,
        T=args.temperature,
        include_hartree=True,
        include_exchange=True,
        reference_density=reference_density,
        hartree_matrix=setup.hartree_matrix,
    )


    # Sort density points by branch into ascending order from CN outward.
    # This gives the warmest possible warm-start for each step.
    positive = sorted([n for n in args.density_points if n > 0])
    negative = sorted([n for n in args.density_points if n < 0], reverse=True)
    zero = [n for n in args.density_points if abs(n) < 1e-8]
    # Branch order: first CN then walk outward
    walk = list(zero) + [n for pair in zip(positive, negative) for n in pair]
    if len(positive) > len(negative):
        walk += positive[len(negative):]
    elif len(negative) > len(positive):
        walk += negative[len(positive):]

    rows: list[dict] = []

    for branch in args.branches:
        print(f"\n=== {branch} ({args.seed_mode}-seed) ===")
        project_fn = setup.project_fns[branch]
        config = cpp_hf.SolverConfig(
            max_iter=int(args.max_iter),
            tol_E=float(args.tol_e),
            max_step=float(args.max_step),
            project_fn=project_fn,
        )

        seed_P = None  # warm-seed state (only used in warm mode)

        # Order: CN first if present (to establish baseline), then outward.
        for n_cm12 in walk:
            n_e, h_run = common.n_electrons_for_density(setup, n_cm12, args.temperature)

            if args.seed_mode == "cold" or seed_P is None:
                P0 = common.initial_density_from_seed(
                    h_run, setup.seeds[branch], args.temperature,
                )
                seed_used = np.asarray(P0)
                seed_label = "cold-seed"
            else:
                seed_used = seed_P
                seed_label = "warm-seed"

            t0 = time.perf_counter()
            r = cpp_hf.solve(kernel, seed_used, n_e, config=config)
            t = time.perf_counter() - t0
            row = _make_row(branch, n_cm12, n_e, r, t, seed_label)
            rows.append(row)
            _print_row(row)
            seed_P = r.density

    return rows


def _make_row(branch: str, n_cm12: float, n_electrons: float,
              r: cpp_hf.SolveResult, elapsed: float, seed_kind: str) -> dict:
    k = int(r.n_iter)
    grad_hist = np.asarray(r.history["grad_norm"])
    E_hist = np.asarray(r.history["E"])
    grad_last = float(grad_hist[k - 1]) if k > 0 else float("nan")
    E_last = float(E_hist[k - 1]) if k > 0 else float("nan")
    return {
        "branch": branch,
        "density_cm12": n_cm12,
        "n_electrons": n_electrons,
        "converged": bool(r.converged),
        "iterations": k,
        "energy": float(r.energy),
        "energy_loop_last": E_last,
        "grad_norm_last": grad_last,
        "mu": float(r.mu),
        "time_s": elapsed,
        "seed": seed_kind,
    }


def _print_row(row: dict) -> None:
    print(
        f"  n={row['density_cm12']:+.3f}  it={row['iterations']:3d}  "
        f"E={row['energy']:.6f}  grad={row['grad_norm_last']:.2e}  "
        f"conv={row['converged']}  t={row['time_s']:.2f}s  {row['seed']}"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort rows: branch first, then density
    rows_sorted = sorted(rows, key=lambda r: (r["branch"], float(r["density_cm12"])))
    fieldnames = [
        "branch", "density_cm12", "n_electrons", "converged", "iterations",
        "energy", "energy_loop_last", "grad_norm_last", "mu", "time_s", "seed",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)


def main() -> int:
    args = parse_args()
    rows = run_scan(args)
    write_csv(args.output, rows)
    print(f"\nwrote CSV: {args.output}")
    if not args.no_figure:
        common.write_figure(
            csv_path=args.output, rows=rows,
            title="Bilayer graphene PM/SVP/SP/SVP_flipped (Hartree + Fock, layer-resolved)",
            branches=list(BRANCHES_EXTENDED),
        )
    total_time = sum(r["time_s"] for r in rows)
    n_pts = len(rows)
    print(f"total solver time: {total_time:.1f}s  ({total_time/n_pts:.2f}s/point over {n_pts} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
