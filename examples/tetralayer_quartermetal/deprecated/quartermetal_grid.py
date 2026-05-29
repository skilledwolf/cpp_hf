#!/usr/bin/env python
"""30×30 (n, D) grid driver for tetralayer ABC quartermetal DOS map.

For each (n, D) cell, runs three SCF configurations and saves
(Δ_tb, μ, energy, energy relative to PM_C3 at CN, DOS at E_F, iters,
converged) to a single NPZ file.  Uses multiprocessing to scale across CPU workers; each
worker pins its BLAS to a single thread to avoid oversubscription.

Scan convention:

  * all runs use the rhombic k cell
  * the interaction reference density is the self-consistent PM_C3 state at CN
  * saved relative energies are measured from that PM_C3 CN reference
  * C3 and time-reversal maps are non-periodic in-patch maps, not torus wraps

The 3-panel plot shows log10(DOS at E_F) vs (n, D).  Cells that fail
to converge are masked with a hatch overlay.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _worker_init():
    """Pin BLAS to 1 thread per worker — multiprocessing oversubscription
    is a real problem with NumPy + LAPACK + Apple Accelerate."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"


CONFIGS = ("HF_PM_C3", "HF_SVP_C3", "HF_SVP")
CONFIGS_C3 = CONFIGS
CONFIGS_C3_PANELS = CONFIGS


def _run_d_row(args: tuple) -> list[dict]:
    """Worker entry point.  For one displacement field D, walks along the
    full n-grid with WARM-START chaining: each config keeps its converged
    density from the previous n as the seed for the next n.  Setup and
    refP are built once per D-row, amortizing model-construction cost.

    Returns a list of per-(n) dicts in the same order as ``n_grid``.
    """
    (D_Vnm, n_grid, nk, init_scale, hartree_kind, T,
     max_iter_h, max_iter_hf, tol_E, vq_scale, enable_c3,
     panel_mode) = args
    _ = (max_iter_h, enable_c3, panel_mode)

    import _common as qm
    import cpp_hf
    from cpp_hf import SolverConfig, solve_direct_minimization

    # Build the full (32-orbital) setup on the rhombic BZ.  All scan configs
    # use this same cell and the same PM_C3 CN reference.
    setup_full = qm.build_setup(
        D_Vnm=D_Vnm, nk=nk, init_scale=init_scale, hartree_kind=hartree_kind,
        small_orbital=False, bz_kind="rhombic",
    )
    if vq_scale != 1.0:
        setup_full  = setup_full._replace(Vq=setup_full.Vq * float(vq_scale))

    layer_full  = np.asarray(setup_full.model.layer)
    refP_full = qm.pm_c3_cn_reference_density(
        setup_full, T=T, max_iter=max_iter_hf, tol_E=tol_E,
    )

    # Per-config "current seed" — updated each n-step (warm start).
    seeds_warm: dict[str, np.ndarray | None] = {
        "HF_PM_C3":   refP_full,
        "HF_SVP_C3":  None,
        "HF_SVP":     None,
    }

    rows: list[dict] = []

    for n_cm12 in n_grid:
        # Total electron count on the full spin-valley basis.
        n_e_full,  h_run_full  = qm.n_electrons_for_density(
            setup_full,  n_cm12, T)

        kernel_hf_full = cpp_hf.HartreeFockKernel(
            weights=setup_full.weights, hamiltonian=np.asarray(h_run_full.hs),
            coulomb_q=setup_full.Vq, T=float(T),
            include_hartree=True, include_exchange=True,
            reference_density=refP_full,
            hartree_matrix=setup_full.hartree_matrix,
        )
        out = dict(n_cm12=float(n_cm12), D_Vnm=float(D_Vnm),
                    n_e=float(n_e_full))
        out["reference_energy"] = qm.hf_energy_for_density(
            kernel_hf_full, refP_full,
        )

        # Helper: run one HF DM config with fallback
        def _run_hf(label: str, projector_key: str, seed_branch: str,
                    require_imbalance: float = 0.0):
            cfg = SolverConfig(
                max_iter=int(max_iter_hf), tol_E=float(tol_E), max_step=0.6,
                block_sizes=(8, 8, 8, 8),
                project_fn=setup_full.project_fns[projector_key],
            )
            res = _solve_with_fallback_dm(
                kernel_hf_full, seeds_warm[label], n_e_full, cfg,
                cold_seed_fn=lambda: qm.initial_density_from_seed(
                    h_run_full, setup_full.seeds[seed_branch], T),
                solve_dm_fn=solve_direct_minimization,
                require_imbalance=require_imbalance,
                imbalance_refP=refP_full if require_imbalance > 0 else None,
                imbalance_w=setup_full.weights if require_imbalance > 0 else None,
            )
            _populate_dm(out, label, res, layer_full, setup_full.weights, T,
                          refP=refP_full, n_orb_per_sv=8,
                          ref_energy=out["reference_energy"])
            if res.converged:
                seeds_warm[label] = np.asarray(res.density)

        _run_hf("HF_PM_C3",  "PM_C3",  "PM")
        _run_hf("HF_SVP_C3", "SVP_C3", "SVP", require_imbalance=10.0)
        _run_hf("HF_SVP",    "SVP",    "SVP", require_imbalance=10.0)

        rows.append(out)

    return rows


# Backwards-compat alias for one-cell path (used by tests / interactive runs)
def _run_one_cell(args: tuple) -> dict:
    n_cm12, D_Vnm, *rest = args
    rows = _run_d_row((D_Vnm, [n_cm12], *rest))
    return rows[0]


def _sv_imbalance(P: np.ndarray, refP: np.ndarray, w2d: np.ndarray,
                   n_orb_per_sv: int = 8) -> float:
    diag_P = np.einsum("ij,ijaa->a", w2d, np.asarray(P)).real
    diag_R = np.einsum("ij,ijaa->a", w2d, np.asarray(refP)).real
    n_blocks = diag_P.shape[0] // n_orb_per_sv
    delta = np.array([
        (diag_P[i*n_orb_per_sv:(i+1)*n_orb_per_sv].sum()
         - diag_R[i*n_orb_per_sv:(i+1)*n_orb_per_sv].sum())
        for i in range(n_blocks)
    ])
    idx = int(np.argmax(np.abs(delta)))
    others = np.delete(delta, idx)
    return float(abs(delta[idx]) / max(np.mean(np.abs(others)), 1e-12))


def _solve_with_fallback(kernel, warm_seed, n_e, cfg, *,
                          cold_seed_fn, solve_scf_fn,
                          always_compare_cold: bool = False,
                          require_imbalance: float = 0.0,
                          imbalance_refP: np.ndarray | None = None,
                          imbalance_w: np.ndarray | None = None):
    """Try warm seed first; on failure or when ``always_compare_cold`` is
    set, also run the cold seed and return the lower-energy converged
    result.

    For SVP runs, the warm seed often lands in the PM basin even when a
    QM basin exists.  Setting ``require_imbalance`` (e.g. 10.0) forces a
    cold restart whenever the warm-converged density has SV-imbalance
    below the threshold — typical of PM-trapped states.
    """
    if warm_seed is None:
        return solve_scf_fn(kernel, cold_seed_fn(), n_e, config=cfg)
    res_warm = solve_scf_fn(kernel, warm_seed, n_e, config=cfg)
    needs_cold = (not res_warm.converged) or always_compare_cold
    if (require_imbalance > 0.0 and res_warm.converged
            and imbalance_refP is not None and imbalance_w is not None):
        imb_warm = _sv_imbalance(res_warm.density_matrix,
                                  imbalance_refP, imbalance_w)
        if imb_warm < require_imbalance:
            needs_cold = True
    if not needs_cold:
        return res_warm
    res_cold = solve_scf_fn(kernel, cold_seed_fn(), n_e, config=cfg)
    candidates = [r for r in (res_warm, res_cold) if r.converged]
    if not candidates:
        return res_warm
    return min(candidates, key=lambda r: float(r.energy))


def _populate_dm(out: dict, label: str, res, layer: np.ndarray,
                 w2d: np.ndarray, T: float, refP: np.ndarray | None = None,
                 n_orb_per_sv: int = 8,
                 ref_energy: float | None = None) -> None:
    """Same as ``_populate`` but for ``DMResult`` whose fields are named
    differently (``fock`` not ``fock_matrix``, ``mu`` not
    ``chemical_potential``, ``n_iter`` not ``iterations``)."""
    import _common as qm
    fock = np.asarray(res.fock)
    P = np.asarray(res.density)
    out[f"{label}_energy"] = float(res.energy)
    if ref_energy is not None:
        out[f"{label}_energy_rel_ref"] = float(res.energy) - float(ref_energy)
    out[f"{label}_mu"]     = float(res.mu)
    out[f"{label}_delta_tb"] = qm.delta_tb_from_fock(fock, layer)
    out[f"{label}_iters"]  = int(res.n_iter)
    out[f"{label}_converged"] = bool(res.converged)
    eps = np.empty((fock.shape[0], fock.shape[1], fock.shape[-1]),
                   dtype=np.float64)
    for k1 in range(fock.shape[0]):
        for k2 in range(fock.shape[1]):
            fk = 0.5 * (fock[k1, k2] + fock[k1, k2].conj().T)
            eps[k1, k2] = np.linalg.eigvalsh(fk)
    out[f"{label}_dos_at_EF"] = qm.dos_at_mu(eps, w2d, out[f"{label}_mu"], T)
    if refP is not None and fock.shape[-1] >= n_orb_per_sv * 2:
        diag_P   = np.einsum("ij,ijaa->a", w2d, P).real
        diag_ref = np.einsum("ij,ijaa->a", w2d, refP).real
        n_blocks = diag_P.shape[0] // n_orb_per_sv
        delta_n = np.array([
            (diag_P[i*n_orb_per_sv:(i+1)*n_orb_per_sv].sum()
             - diag_ref[i*n_orb_per_sv:(i+1)*n_orb_per_sv].sum())
            for i in range(n_blocks)
        ])
        out[f"{label}_sv_dn_max"] = float(np.max(delta_n))
        out[f"{label}_sv_dn_min"] = float(np.min(delta_n))
        idx_max = int(np.argmax(np.abs(delta_n)))
        others = np.delete(delta_n, idx_max)
        denom = max(np.mean(np.abs(others)), 1e-12)
        out[f"{label}_sv_imbalance"] = float(abs(delta_n[idx_max]) / denom)
    else:
        out[f"{label}_sv_dn_max"] = float("nan")
        out[f"{label}_sv_dn_min"] = float("nan")
        out[f"{label}_sv_imbalance"] = 1.0


def _solve_with_fallback_dm(kernel, warm_seed, n_e, cfg, *,
                             cold_seed_fn, solve_dm_fn,
                             require_imbalance: float = 0.0,
                             imbalance_refP=None, imbalance_w=None):
    """DM variant of solve_with_fallback (uses .density not .density_matrix)."""
    if warm_seed is None:
        return solve_dm_fn(kernel, cold_seed_fn(), n_e, config=cfg)
    res_warm = solve_dm_fn(kernel, warm_seed, n_e, config=cfg)
    needs_cold = (not res_warm.converged)
    if (require_imbalance > 0.0 and res_warm.converged
            and imbalance_refP is not None and imbalance_w is not None):
        imb_warm = _sv_imbalance(res_warm.density, imbalance_refP, imbalance_w)
        if imb_warm < require_imbalance:
            needs_cold = True
    if not needs_cold:
        return res_warm
    res_cold = solve_dm_fn(kernel, cold_seed_fn(), n_e, config=cfg)
    candidates = [r for r in (res_warm, res_cold) if r.converged]
    if not candidates:
        return res_warm
    return min(candidates, key=lambda r: float(r.energy))


def _populate(out: dict, label: str, res, layer: np.ndarray,
              w2d: np.ndarray, T: float, refP: np.ndarray | None = None,
              n_orb_per_sv: int = 8,
              scale_e: float = 1.0, scale_dos: float = 1.0,
              ref_energy: float | None = None) -> None:
    """Extract derived quantities (Δ_tb, μ, E, DOS at E_F, conv,
    sv_imbalance) from a SCFResult.

    ``scale_e`` and ``scale_dos``: multipliers for energy and DOS, used
    to convert per-flavor quantities (8-orbital model) into total
    quantities (×4 = degeneracy).  Δ_tb and μ are per-orbital scalars
    and don't need scaling.

    ``sv_imbalance``: max(Δn_sector) / mean(|Δn_others|).  For 8-orbital
    PM-only models there's just one sector so imbalance is trivially 1.
    """
    import _common as qm
    fock = np.asarray(res.fock_matrix)
    P = np.asarray(res.density_matrix)
    out[f"{label}_energy"] = float(res.energy) * scale_e
    if ref_energy is not None:
        out[f"{label}_energy_rel_ref"] = (
            float(res.energy) * scale_e - float(ref_energy)
        )
    out[f"{label}_mu"]     = float(res.chemical_potential)
    out[f"{label}_delta_tb"] = qm.delta_tb_from_fock(fock, layer)
    out[f"{label}_iters"]  = int(res.iterations)
    out[f"{label}_converged"] = bool(res.converged)
    eps = np.empty((fock.shape[0], fock.shape[1], fock.shape[-1]),
                   dtype=np.float64)
    for k1 in range(fock.shape[0]):
        for k2 in range(fock.shape[1]):
            fk = 0.5 * (fock[k1, k2] + fock[k1, k2].conj().T)
            eps[k1, k2] = np.linalg.eigvalsh(fk)
    out[f"{label}_dos_at_EF"] = qm.dos_at_mu(eps, w2d, out[f"{label}_mu"], T) * scale_dos
    if refP is not None and fock.shape[-1] >= n_orb_per_sv * 2:
        diag_P   = np.einsum("ij,ijaa->a", w2d, P).real
        diag_ref = np.einsum("ij,ijaa->a", w2d, refP).real
        n_blocks = diag_P.shape[0] // n_orb_per_sv
        delta_n = np.array([
            (diag_P[i*n_orb_per_sv:(i+1)*n_orb_per_sv].sum()
             - diag_ref[i*n_orb_per_sv:(i+1)*n_orb_per_sv].sum())
            for i in range(n_blocks)
        ])
        out[f"{label}_sv_dn_max"] = float(np.max(delta_n))
        out[f"{label}_sv_dn_min"] = float(np.min(delta_n))
        idx_max = int(np.argmax(np.abs(delta_n)))
        others = np.delete(delta_n, idx_max)
        denom = max(np.mean(np.abs(others)), 1e-12)
        out[f"{label}_sv_imbalance"] = float(abs(delta_n[idx_max]) / denom)
    else:
        # Small-orbital (8-orb) model: no spin-valley structure to break.
        out[f"{label}_sv_dn_max"] = float("nan")
        out[f"{label}_sv_dn_min"] = float("nan")
        out[f"{label}_sv_imbalance"] = 1.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-grid", type=int, default=30)
    p.add_argument("--D-grid", type=int, default=30)
    p.add_argument("--n-min", type=float, default=0.05)
    p.add_argument("--n-max", type=float, default=1.5)
    p.add_argument("--D-min", type=float, default=0.05)
    p.add_argument("--D-max", type=float, default=1.0)
    p.add_argument("--nk", type=int, default=24,
                   help="k-mesh density (spec: 48; 24 for fast iteration).")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--init-scale", type=float, default=50.0)
    p.add_argument("--hartree-kind",
                   choices=("ungated", "ungated_eps3", "dualgate",
                            "dualgate_geomean", "dualgate_eps2",
                            "dualgate_eps3", "dualgate_eps4",
                            "dualgate_eps5", "dualgate_eps10",
                            "dualgate_eps20", "poisson", "poisson_proj",
                            "poisson_proj_sym"),
                   default="dualgate_eps5",
                   help="Interaction screening preset. dualgate_eps* applies "
                        "the same isotropic epsilon to finite-q Fock and q=0 "
                        "Hartree; default dualgate_eps5.")
    p.add_argument("--max-iter-h", type=int, default=25)
    p.add_argument("--max-iter-hf", type=int, default=25)
    p.add_argument("--tol-E", type=float, default=1e-4)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--enable-c3", action="store_true",
                   help="Deprecated; scans always include PM_C3 and SVP_C3.")
    p.add_argument("--panel-mode", choices=("default", "spec3"),
                   default="spec3",
                   help="Deprecated; scans always run HF_PM_C3, HF_SVP_C3, HF_SVP.")
    p.add_argument("--vq-scale", type=float, default=1.0,
                   help="Diagnostic: scale the exchange kernel by this factor. "
                        "Default 1.0 = physical.  Try 2-5 to deepen SVP basins.")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "examples" / "outputs" /
                            "quartermetal_grid.npz")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip the SCF and just plot from existing NPZ.")
    args = p.parse_args(argv)

    n_grid = np.linspace(args.n_min, args.n_max, args.n_grid)
    D_grid = np.linspace(args.D_min, args.D_max, args.D_grid)

    if not args.plot_only:
        # Build job list: one job per D-row.  Each row walks the full
        # n-grid with warm-start chaining (cuts ~5× iters from cold seed).
        n_walk = list(map(float, n_grid))   # walk in given order (small → large)
        jobs = []
        for D in D_grid:
            jobs.append((float(D), n_walk, int(args.nk),
                          float(args.init_scale), args.hartree_kind,
                          float(args.T), int(args.max_iter_h),
                          int(args.max_iter_hf), float(args.tol_E),
                          float(args.vq_scale), bool(args.enable_c3),
                          str(args.panel_mode)))

        n_rows = len(jobs)
        n_cells = n_rows * len(n_walk)
        print(f"Submitting {n_rows} D-rows × {len(n_walk)} n-cells × 3 configs "
              f"= {3 * n_cells} solves on {args.workers} workers, "
              f"nk={args.nk}, T={args.T} meV.")

        results: list[dict] = []
        ctx = mp.get_context("spawn")
        t0 = time.perf_counter()
        with ctx.Pool(processes=int(args.workers),
                       initializer=_worker_init) as pool:
            for ri, row in enumerate(pool.imap_unordered(_run_d_row, jobs,
                                                          chunksize=1)):
                results.extend(row)
                elapsed = time.perf_counter() - t0
                rows_done = ri + 1
                rate_rows = rows_done / max(elapsed, 1e-9)
                eta = (n_rows - rows_done) / max(rate_rows, 1e-9)
                print(f"  D-row {rows_done:3d}/{n_rows}  "
                      f"{elapsed:6.1f}s  {rate_rows*len(n_walk):.2f} cells/s  "
                      f"eta={eta:6.1f}s")

        print(f"All D-rows done in {time.perf_counter() - t0:.1f}s.")
        _save_results(args.output, results, n_grid, D_grid, configs=CONFIGS)

    _plot_results(args.output, args.output.with_suffix(".png"))
    return 0


def _save_results(path: Path, results: list[dict],
                  n_grid: np.ndarray, D_grid: np.ndarray,
                  configs: tuple[str, ...] = CONFIGS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_n = len(n_grid)
    n_D = len(D_grid)
    arrays: dict = dict(n_grid=n_grid, D_grid=D_grid)
    arrays["reference_energy"] = np.full((n_n, n_D), np.nan, dtype=np.float64)
    fields = (
        "energy", "energy_rel_ref", "mu", "delta_tb", "iters",
        "converged", "dos_at_EF", "sv_dn_max", "sv_dn_min", "sv_imbalance",
    )
    for label in configs:
        for f in fields:
            arrays[f"{label}_{f}"] = np.full(
                (n_n, n_D),
                np.nan if f != "iters" and f != "converged" else 0,
                dtype=(np.int32 if f in ("iters",)
                       else (np.bool_ if f == "converged" else np.float64)),
            )
    n_to_idx = {float(n): i for i, n in enumerate(n_grid)}
    D_to_idx = {float(D): i for i, D in enumerate(D_grid)}
    for r in results:
        i = n_to_idx[float(r["n_cm12"])]
        j = D_to_idx[float(r["D_Vnm"])]
        if "reference_energy" in r:
            arrays["reference_energy"][i, j] = float(r["reference_energy"])
        for label in configs:
            for f in fields:
                key = f"{label}_{f}"
                if key in r:
                    arrays[key][i, j] = r[key]
    if "HF_PM_C3_energy" in arrays:
        for label in configs:
            if f"{label}_energy" in arrays:
                arrays[f"{label}_minus_HF_PM_C3_energy"] = (
                    arrays[f"{label}_energy"] - arrays["HF_PM_C3_energy"]
                )
            if f"{label}_energy_rel_ref" in arrays:
                arrays[f"{label}_minus_HF_PM_C3_energy_rel_ref"] = (
                    arrays[f"{label}_energy_rel_ref"]
                    - arrays["HF_PM_C3_energy_rel_ref"]
                )
    np.savez(path, **arrays)
    print(f"Saved {path}")


def _plot_results(npz_path: Path, png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = np.load(npz_path)
    n_grid = data["n_grid"]
    D_grid = data["D_grid"]

    # Auto-detect which configs are saved in the NPZ
    configs_present = []
    for c in (CONFIGS_C3_PANELS + CONFIGS_C3 + CONFIGS):
        if f"{c}_dos_at_EF" in data and c not in configs_present:
            configs_present.append(c)
    top_cfgs = configs_present[:3]   # show at most 3 in top row

    # Top row: DOS at E_F per config; bottom row: SVP–PM energy diagnostic.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # ---- Top: DOS at E_F for each config ----
    for ax, label in zip(axes[0], top_cfgs):
        dos = data[f"{label}_dos_at_EF"].T   # (D, n)
        conv = data[f"{label}_converged"].T
        log_dos = np.log10(np.clip(dos, 1e-12, None))
        im = ax.imshow(
            log_dos, origin="lower",
            extent=(n_grid[0], n_grid[-1], D_grid[0], D_grid[-1]),
            aspect="auto", cmap="viridis",
        )
        unconv = ~conv
        if unconv.any() and unconv.shape[0] >= 2 and unconv.shape[1] >= 2:
            ax.contourf(n_grid, D_grid, unconv.astype(float),
                         levels=[0.5, 1.5], hatches=["//"],
                         colors="none", extend="neither")
        ax.set_title(label)
        ax.set_xlabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
        ax.set_ylabel(r"$D$ (V/nm)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(r"$\log_{10}\,\mathrm{DOS}(E_F)$")

    # ---- Bottom: physics diagnostics ----
    # (1) SVP - PM_C3 energy map per carrier (negative = QM phase wins).
    pair_choices = [("HF_SVP_C3", "HF_PM_C3"), ("HF_SVP", "HF_PM_C3")]
    pair = next(((s, p) for s, p in pair_choices
                  if f"{s}_energy" in data and f"{p}_energy" in data), None)
    if pair is not None:
        svp_label, pm_label = pair
        ax = axes[1, 0]
        e_field = "energy_rel_ref"
        if (f"{svp_label}_{e_field}" not in data
                or f"{pm_label}_{e_field}" not in data):
            e_field = "energy"
        e_svp = data[f"{svp_label}_{e_field}"].T
        e_pm = data[f"{pm_label}_{e_field}"].T
        c_both = (data[f"{svp_label}_converged"] & data[f"{pm_label}_converged"]).T
        # Carrier excess per cell = n_e - ne_cn = n_cm12 × 1e12 × aG² (per
        # unit cell in BZ-summed weights).  For n=0 cells use a tiny
        # floor to avoid divide-by-zero — they're meaningless for ΔE/N.
        PER_CM = 0.246e-7
        n_2d = n_grid[None, :] * np.ones_like(D_grid[:, None])
        carrier_excess = n_2d * 1e12 * (PER_CM ** 2)
        carrier_excess = np.where(carrier_excess > 1e-9, carrier_excess, np.nan)
        dE_per_carrier = np.where(c_both, (e_svp - e_pm) / carrier_excess,
                                   np.nan)
        finite = dE_per_carrier[np.isfinite(dE_per_carrier)]
        vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 50.0
        vmax = max(vmax, 1.0)
        im = ax.imshow(
            dE_per_carrier, origin="lower",
            extent=(n_grid[0], n_grid[-1], D_grid[0], D_grid[-1]),
            aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
        )
        ax.set_title(f"({svp_label} - {pm_label}) / N_carrier (PM_C3 CN ref)")
        ax.set_xlabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
        ax.set_ylabel(r"$D$ (V/nm)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("meV / carrier")

    # (1b) sv_imbalance for the strongest broken-symmetry config available
    imb_label = next((c for c in ("HF_SVP_C3", "HF_SVP")
                       if f"{c}_sv_imbalance" in data), None)
    if imb_label is not None:
        ax = axes[1, 1]
        imbal = data[f"{imb_label}_sv_imbalance"].T
        conv = data[f"{imb_label}_converged"].T
        imbal = np.where(conv, imbal, np.nan)
        im = ax.imshow(
            np.log10(np.clip(imbal, 1.0, None)), origin="lower",
            extent=(n_grid[0], n_grid[-1], D_grid[0], D_grid[-1]),
            aspect="auto", cmap="magma",
        )
        ax.set_title(rf"$\log_{{10}}$ SV imbalance ({imb_label}) — "
                      r"high = clean QM")
        ax.set_xlabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
        ax.set_ylabel(r"$D$ (V/nm)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(r"$\log_{10}\,(\Delta n_{\rm max}/\Delta n_{\rm avg-others})$")

    # (2) Δ_tb for the first available HF config
    ax = axes[1, 2]
    delta_label = next((c for c in ("HF_PM", "HF_PM_C3", "HF_SVP")
                         if f"{c}_delta_tb" in data), None)
    if delta_label is None:
        ax.set_visible(False)
        return
    delta = data[f"{delta_label}_delta_tb"].T
    im = ax.imshow(
        delta, origin="lower",
        extent=(n_grid[0], n_grid[-1], D_grid[0], D_grid[-1]),
        aspect="auto", cmap="plasma",
    )
    ax.set_title(rf"$\Delta_{{\rm tb}}$ ({delta_label}, meV)")
    ax.set_xlabel(r"$n$ ($10^{12}$ cm$^{-2}$)")
    ax.set_ylabel(r"$D$ (V/nm)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("Tetralayer ABC graphene — rhombic PM_C3-referenced scan "
                  "(top: DOS; bottom: phase diagnostics)")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    raise SystemExit(main())
