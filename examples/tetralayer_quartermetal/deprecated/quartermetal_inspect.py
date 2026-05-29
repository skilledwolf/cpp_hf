#!/usr/bin/env python
"""Quick inspection of a quartermetal grid NPZ — energies, Δ_tb, DOS,
convergence rates per config.  Use when debugging a run before plotting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_NPZ = (Path(__file__).resolve().parents[3]
                / "examples" / "outputs" / "quartermetal_grid.npz")
CONFIGS = ("HF_PM_C3", "HF_SVP_C3", "HF_SVP", "H_PM", "HF_PM")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", nargs="?", default=DEFAULT_NPZ, type=Path)
    args = p.parse_args(argv)

    if not args.npz.exists():
        print(f"NPZ not found: {args.npz}", file=sys.stderr)
        return 1

    data = np.load(args.npz)
    n_grid = data["n_grid"]
    D_grid = data["D_grid"]
    n_n = len(n_grid)
    n_D = len(D_grid)

    print(f"Grid: {n_n} (n) × {n_D} (D)")
    dn = n_grid[1] - n_grid[0] if len(n_grid) > 1 else 0.0
    dD = D_grid[1] - D_grid[0] if len(D_grid) > 1 else 0.0
    print(f"  n: {n_grid[0]:.3f} ... {n_grid[-1]:.3f}  ({dn:.4f} step)")
    print(f"  D: {D_grid[0]:.3f} ... {D_grid[-1]:.3f}  ({dD:.4f} step)")
    print()

    for label in CONFIGS:
        if f"{label}_converged" not in data:
            continue
        conv = data[f"{label}_converged"]
        iters = data[f"{label}_iters"]
        E = data[f"{label}_energy"]
        delta = data[f"{label}_delta_tb"]
        dos = data[f"{label}_dos_at_EF"]

        n_conv = int(conv.sum())
        n_total = conv.size
        avg_iter = float(iters[conv].mean()) if n_conv > 0 else float("nan")
        max_iter = int(iters[conv].max()) if n_conv > 0 else 0

        e_finite = E[conv & np.isfinite(E)]
        d_finite = delta[conv & np.isfinite(delta)]
        dos_finite = dos[conv & np.isfinite(dos)]

        print(f"=== {label} ===")
        print(f"  converged: {n_conv}/{n_total}  ({100.0*n_conv/n_total:.1f}%)")
        print(f"  iters: mean={avg_iter:.1f}  max={max_iter}")
        if e_finite.size > 0:
            print(f"  E (meV):   min={e_finite.min():+.3f}  "
                  f"max={e_finite.max():+.3f}  span={np.ptp(e_finite):.3f}")
            print(f"  Δ_tb:     min={d_finite.min():+.3f}  "
                  f"max={d_finite.max():+.3f}")
            print(f"  log10 DOS: min={np.log10(np.clip(dos_finite,1e-12,None)).min():.2f}  "
                  f"max={np.log10(dos_finite.max()):.2f}")
        if n_conv < n_total:
            unconv_n, unconv_D = np.where(~conv)
            print(f"  unconverged cells: {len(unconv_n)}")
            for ni, di in zip(unconv_n[:6], unconv_D[:6]):
                print(f"    (n={n_grid[ni]:.3f}, D={D_grid[di]:.3f})")
            if len(unconv_n) > 6:
                print(f"    ... and {len(unconv_n) - 6} more")
        print()

    # SVP vs PM_C3 energy comparison
    pair = next(((s, p) for s, p in (
        ("HF_SVP_C3", "HF_PM_C3"), ("HF_SVP", "HF_PM_C3"),
        ("HF_SVP", "HF_PM"),
    ) if f"{s}_energy" in data and f"{p}_energy" in data), None)
    if pair is not None:
        svp_label, pm_label = pair
        e_field = "energy_rel_ref"
        if f"{svp_label}_{e_field}" not in data or f"{pm_label}_{e_field}" not in data:
            e_field = "energy"
        e_svp = data[f"{svp_label}_{e_field}"]
        e_pm = data[f"{pm_label}_{e_field}"]
        c_svp = data[f"{svp_label}_converged"]
        c_pm = data[f"{pm_label}_converged"]
        both = c_svp & c_pm
        if both.any():
            dE = e_svp[both] - e_pm[both]
            broken = (dE < -1e-4).sum()
            equal = (np.abs(dE) <= 1e-4).sum()
            higher = (dE > 1e-4).sum()
            print(f"=== {svp_label} vs {pm_label} energy (both converged) ===")
            print(f"  total cells: {int(both.sum())}")
            print(f"  E_SVP < E_PM (broken-symmetry): {broken}")
            print(f"  E_SVP ≈ E_PM (no breaking):     {equal}")
            print(f"  E_SVP > E_PM (SCF stuck):       {higher}")
            print(f"  ΔE distribution (meV): "
                  f"min={dE.min():+.4f}  median={np.median(dE):+.4f}  "
                  f"max={dE.max():+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
