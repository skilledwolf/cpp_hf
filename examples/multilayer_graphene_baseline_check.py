#!/usr/bin/env python
"""Bilayer graphene baseline check for the direct-minimization solver.

Solves a small number of (branch, density) points with the v2 solver
(``cpp_hf.solve``) and compares the total energy against a maintained
reference CSV produced by the SCF solver.

Supports an optional ``--continuation`` flag that solves first on a coarse
k-grid and uses the resampled density as a seed for the target grid.
Continuation relies on :func:`cpp_hf.resample_kgrid` for periodic bilinear
interpolation of the density matrix between grid sizes.

Usage:
    python examples/multilayer_graphene_baseline_check.py
    python examples/multilayer_graphene_baseline_check.py --continuation
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any


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


DEFAULT_OUTPUT = REPO_ROOT / "examples" / "outputs" / "baseline_check.csv"
DEFAULT_REFERENCE_CSV = REPO_ROOT / "examples" / "outputs" / "reference_scf_density_scan.csv"
DEFAULT_DENSITY_POINTS = (-0.42,)
DEFAULT_CONTINUATION_SCHEDULE = (17, 33)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branches", nargs="+", default=list(common.BRANCHES),
                   choices=list(common.BRANCHES))
    p.add_argument("--density-points", nargs="+", type=float,
                   default=list(DEFAULT_DENSITY_POINTS),
                   help="Carrier densities in 1e12 cm^-2")
    p.add_argument("--nk", type=int, default=common.NK)
    p.add_argument("--kmax", type=float, default=common.KMAX)
    p.add_argument("--u-mev", type=float, default=common.U_MEV)
    p.add_argument("--temperature", type=float, default=common.TEMPERATURE)
    p.add_argument("--epsilon-r", type=float, default=common.EPSILON_R)
    p.add_argument("--d-gate", type=float, default=common.D_GATE)
    p.add_argument("--init-scale", type=float, default=common.INIT_SCALE)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol-e", type=float, default=1e-7)
    p.add_argument("--max-step", type=float, default=0.6)
    p.add_argument("--continuation", action="store_true",
                   help="Solve on a coarse grid first and use the resampled "
                        "density as a warm seed for the target grid.")
    p.add_argument("--continuation-schedule", nargs="+", type=int,
                   default=list(DEFAULT_CONTINUATION_SCHEDULE),
                   help="Ascending list of coarse nk values used before the "
                        "target nk (requires --continuation).")
    p.add_argument("--energy-tol-mev", type=float, default=0.5,
                   help="Maximum allowed energy-per-particle mismatch vs "
                        "the reference CSV (meV).")
    p.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--fail-on-mismatch", action="store_true")
    return p.parse_args()


def _normalize_density_key(value: float) -> float:
    return round(float(value), 2)


def load_reference_targets(
    reference_csv: Path,
    *,
    branches: tuple[str, ...],
    density_points: tuple[float, ...],
) -> dict[tuple[str, float], dict[str, str]]:
    if not reference_csv.exists():
        raise SystemExit(
            f"reference CSV not found at {reference_csv}\n"
            "generate it first with:\n"
            "    python examples/multilayer_graphene_reference_scf_scan.py"
        )
    with reference_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    targets: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        key = (row["branch"], _normalize_density_key(float(row["density_cm12"])))
        targets[key] = row
    missing = [
        (branch, density_cm12)
        for branch in branches
        for density_cm12 in density_points
        if (branch, _normalize_density_key(density_cm12)) not in targets
    ]
    if missing:
        missing_text = ", ".join(f"{b}@{d:+.2f}" for b, d in missing)
        raise ValueError(
            f"Reference CSV {reference_csv} is missing requested baseline "
            f"points: {missing_text}"
        )
    return targets


def _kernel_for_setup(setup: common.BilayerSetup, temperature: float) -> cpp_hf.HartreeFockKernel:
    return cpp_hf.HartreeFockKernel(
        weights=setup.weights,
        hamiltonian=np.asarray(setup.h_template.hs),
        coulomb_q=setup.Vq,
        T=float(temperature),
        include_hartree=False,
        include_exchange=True,
    )


def _solve_single(
    setup: common.BilayerSetup,
    branch: str,
    n_e: float,
    P0: np.ndarray,
    *,
    temperature: float,
    max_iter: int,
    tol_e: float,
    max_step: float,
) -> cpp_hf.SolveResult:
    kernel = _kernel_for_setup(setup, temperature)
    config = cpp_hf.SolverConfig(
        max_iter=int(max_iter),
        tol_E=float(tol_e),
        max_step=float(max_step),
        project_fn=setup.project_fns[branch],
    )
    return cpp_hf.solve(kernel, P0, n_e, config=config)


def _solve_with_continuation(
    args: argparse.Namespace,
    branch: str,
    n_cm12: float,
    target_nk: int,
) -> tuple[cpp_hf.SolveResult, float]:
    """Run solve on a schedule (coarse nk values) then on target_nk.

    Returns (final_result, total_elapsed_seconds).
    """
    schedule = [int(nk) for nk in args.continuation_schedule if int(nk) < target_nk]
    schedule = sorted(set(schedule)) + [target_nk]

    t0 = time.perf_counter()
    seed_P: np.ndarray | None = None
    result: cpp_hf.SolveResult | None = None
    for nk in schedule:
        setup = common.build_bilayer(
            nk=nk, kmax=args.kmax, U_meV=args.u_mev,
            temperature=args.temperature, epsilon_r=args.epsilon_r,
            d_gate=args.d_gate, init_scale=args.init_scale,
        )
        n_e, h_run = common.n_electrons_for_density(setup, n_cm12, args.temperature)
        if seed_P is None:
            P0 = common.initial_density_from_seed(
                h_run, setup.seeds[branch], args.temperature,
            )
            P0 = np.asarray(P0)
        else:
            P0 = cpp_hf.resample_kgrid(seed_P, nk)
        result = _solve_single(
            setup, branch, n_e, P0,
            temperature=args.temperature,
            max_iter=args.max_iter, tol_e=args.tol_e, max_step=args.max_step,
        )
        seed_P = result.density
    elapsed = time.perf_counter() - t0
    assert result is not None
    return result, elapsed


def _solve_cold(
    args: argparse.Namespace,
    branch: str,
    n_cm12: float,
) -> tuple[cpp_hf.SolveResult, float]:
    t0 = time.perf_counter()
    setup = common.build_bilayer(
        nk=args.nk, kmax=args.kmax, U_meV=args.u_mev,
        temperature=args.temperature, epsilon_r=args.epsilon_r,
        d_gate=args.d_gate, init_scale=args.init_scale,
    )
    n_e, h_run = common.n_electrons_for_density(setup, n_cm12, args.temperature)
    P0 = common.initial_density_from_seed(
        h_run, setup.seeds[branch], args.temperature,
    )
    result = _solve_single(
        setup, branch, n_e, np.asarray(P0),
        temperature=args.temperature,
        max_iter=args.max_iter, tol_e=args.tol_e, max_step=args.max_step,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _per_carrier_energy(E_target: float, E_neutral: float, n_carriers: float) -> float:
    if abs(n_carriers) < 1e-12:
        return 0.0
    # Per-carrier energy in the solver's native units (matches the reference
    # CSV which is produced at the same unit system).
    return (E_target - E_neutral) / n_carriers


def _cm12_to_dimless(n_cm12: float) -> float:
    return n_cm12 * 1e12 * (common.PER_CM ** 2)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    branches = tuple(args.branches)
    density_points = tuple(float(n) for n in args.density_points)
    reference_rows = load_reference_targets(
        Path(args.reference_csv), branches=branches, density_points=density_points,
    )

    def _run_one(branch: str, n_cm12: float) -> tuple[cpp_hf.SolveResult, float]:
        if args.continuation:
            return _solve_with_continuation(args, branch, n_cm12, args.nk)
        return _solve_cold(args, branch, n_cm12)

    rows: list[dict[str, Any]] = []
    for branch in branches:
        neutral_result, neutral_elapsed = _run_one(branch, 0.0)
        neutral_energy = float(neutral_result.energy)
        print(
            f"neutral {branch:>3} conv={bool(neutral_result.converged)} "
            f"it={int(neutral_result.n_iter):3d} "
            f"E={neutral_energy:.6f} time={neutral_elapsed:.2f}s"
        )

        for n_cm12 in density_points:
            result, elapsed = _run_one(branch, n_cm12)

            n_carriers = _cm12_to_dimless(n_cm12)
            energy_per_carrier = _per_carrier_energy(
                float(result.energy), neutral_energy, n_carriers,
            )
            ref_row = reference_rows[(branch, _normalize_density_key(n_cm12))]
            # Reference CSV stores absolute total energies.  Compute its
            # per-carrier value relative to its own neutral baseline.
            ref_neutral_key = (branch, _normalize_density_key(0.0))
            ref_neutral_row = reference_rows.get(ref_neutral_key)
            if ref_neutral_row is None:
                # Fall back to absolute energy comparison.
                ref_epp = float(ref_row["energy"])
                epp = float(result.energy)
            else:
                ref_epp = _per_carrier_energy(
                    float(ref_row["energy"]),
                    float(ref_neutral_row["energy"]),
                    n_carriers,
                )
                epp = energy_per_carrier
            delta = epp - ref_epp
            ok = abs(delta) <= float(args.energy_tol_mev)

            row = {
                "branch": branch,
                "density_cm12": n_cm12,
                "converged": bool(result.converged),
                "iterations": int(result.n_iter),
                "energy": float(result.energy),
                "energy_per_carrier_meV": epp,
                "reference_energy_per_carrier_meV": ref_epp,
                "delta_energy_per_carrier_meV": delta,
                "within_tolerance": ok,
                "continuation": bool(args.continuation),
                "time_s": float(elapsed),
                "nk_target": int(args.nk),
                "continuation_schedule": (
                    tuple(args.continuation_schedule) if args.continuation else ()
                ),
            }
            rows.append(row)
            print(
                f"{branch:>3} n={n_cm12:+.2f} conv={row['converged']} "
                f"it={row['iterations']:3d} Epp={epp:.6f} ref={ref_epp:.6f} "
                f"delta={delta:+.6f} ok={ok} time={elapsed:.2f}s"
            )
    return rows


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = run(args)
    write_rows(rows, Path(args.output))
    failures = [r for r in rows if not r["within_tolerance"]]
    if failures:
        max_delta = max(abs(float(r["delta_energy_per_carrier_meV"])) for r in failures)
        print(
            f"baseline mismatches: {len(failures)}/{len(rows)} "
            f"(max |delta|={max_delta:.6f} meV)"
        )
        if args.fail_on_mismatch:
            raise SystemExit(1)
    else:
        print(f"all {len(rows)} baseline points are within tolerance.")


if __name__ == "__main__":
    main()
