#!/usr/bin/env python
"""30×30 (n, D) grid for SVP_C3 at ε=2 with PM_C3 CN reference subtraction.

Mirrors quartermetal_grid_pm_c3.py but with the SVP_C3 projector (and SVP
seed) — i.e., enforce spin-valley polarized + C3 symmetric.  Adds a 4th
panel to the plot showing an SVP order parameter so we can tell which
cells actually converged to a polarized solution vs. fell back to PM-like.

Per-cell SCF settings: max_iter=25, tol_E=1e-4, nk=48, kmax=0.30, T=0.3,
ε=2 (dualgate_eps2), PM_C3 self-consistent CN reference.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parent
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"


def _run_d_row(args: tuple) -> list[dict]:
    (D_Vnm, n_grid, nk, kmax, T, hartree_kind, init_scale,
     max_iter_ref, max_iter_hf, tol_E) = args

    import gc
    import _quartermetal_common as qm
    import cpp_hf
    from cpp_hf import SolverConfig, solve_direct_minimization
    from cpp_hf.fock import build_fock
    from cpp_hf.utils import selfenergy_fft

    setup = qm.build_setup(
        D_Vnm=D_Vnm, nk=nk, kmax=kmax, T=T,
        bz_kind="rhombic", init_scale=init_scale,
        hartree_kind=hartree_kind,
    )
    layer = np.asarray(setup.model.layer)

    # Reference density: same as PM_C3 grid — the self-consistent PM_C3
    # state at CN.  Both PM_C3 and SVP_C3 production runs subtract this
    # same reference so energies are comparable across phases.
    try:
        refP = qm.pm_c3_cn_reference_density(
            setup, T=T, max_iter=max_iter_ref, tol_E=tol_E,
        )
    except RuntimeError as exc:
        msg = str(exc)
        return [
            {
                "n": float(n), "D": float(D_Vnm),
                "energy": float("nan"), "mu": float("nan"),
                "delta_tb": float("nan"), "dos_at_EF": float("nan"),
                "iters": -1, "converged": False, "imbalance": float("nan"),
                "svp_op": float("nan"),
                "ref_failed": True, "error": msg,
            }
            for n in n_grid
        ]

    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(setup.h_template.hs),
        coulomb_q=setup.Vq, T=T,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    project_fn = setup.project_fns["SVP_C3"]

    # Pre-compute Σ_x[refP] and HH @ diag(refP) so per-cell physical Fock
    # reconstruction is cheap (linear in P).
    w2d = np.asarray(setup.weights)
    Sigma_ref = selfenergy_fft(
        np.asarray(kernel._VR_shifted), np.asarray(refP),
        _apply_ifftshift=False,
        hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
    )
    diag_R_ref = np.einsum("ij,ijaa->a", w2d, np.asarray(refP)).real
    V_H_ref = setup.hartree_matrix @ diag_R_ref  # shape (n_orb,)

    # Initial seed: SVP at the first n value
    n0 = float(n_grid[0])
    n_e0, h_run0 = qm.n_electrons_for_density(setup, n0, T)
    P_seed = qm.initial_density_from_seed(h_run0, setup.seeds["SVP"], T)
    last_P = P_seed

    out_rows: list[dict] = []
    for n_cm12 in n_grid:
        n_e, h_run = qm.n_electrons_for_density(setup, float(n_cm12), T)
        kernel = cpp_hf.HartreeFockKernel(
            weights=setup.weights,
            hamiltonian=np.asarray(h_run.hs),
            coulomb_q=setup.Vq, T=T,
            include_hartree=True, include_exchange=True,
            reference_density=refP,
            hartree_matrix=setup.hartree_matrix,
        )
        cfg = SolverConfig(
            max_iter=max_iter_hf, tol_E=tol_E, max_step=0.6,
            block_sizes=(8, 8, 8, 8), project_fn=project_fn,
        )
        t0 = time.perf_counter()
        res = solve_direct_minimization(kernel, last_P, n_e, config=cfg)
        dt = time.perf_counter() - t0

        # Physical Fock = relative Fock + Σ_x[refP] + diag(HH @ diag(refP))
        P_arr = np.ascontiguousarray(np.asarray(res.density), dtype=np.complex128)
        Sigma, H, _ = build_fock(
            P_arr, h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
            HH=kernel.HH, w2d=kernel.w2d,
            include_exchange=kernel.include_exchange,
            include_hartree=kernel.include_hartree,
            exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
            contact_g=kernel.contact_g,
            contact_Oi=kernel.contact_Oi,
            contact_Oj=kernel.contact_Oj,
        )
        fock_rel = np.asarray(kernel.h) + np.asarray(Sigma) + np.asarray(H)
        w = w2d
        n_orb_total = fock_rel.shape[-1]
        fock = fock_rel + np.asarray(Sigma_ref)
        for a in range(n_orb_total):
            fock[:, :, a, a] = fock[:, :, a, a] + V_H_ref[a]
        delta_tb = qm.delta_tb_from_fock(fock, layer)

        # DOS at E_F via bztetra.twod (physical Fock, mu re-found)
        eigs = np.linalg.eigvalsh(fock.reshape(-1, fock.shape[-2],
                                                 fock.shape[-1]))
        eigs = eigs.reshape(*fock.shape[:2], -1)
        from cpp_hf.utils import find_chemical_potential
        E_F = float(find_chemical_potential(eigs, w, n_e, T,
                                              method="bisection"))
        import bztetra.twod as twod
        s = 2.0 * kmax
        h_tri = s * np.sqrt(3.0) / 2.0
        Bmat = np.array([[s / 2.0, s / 2.0], [-h_tri, h_tri]])
        W_dos = twod.density_of_states_weights(
            Bmat, eigs, np.array([E_F])
        )
        area_prefactor = float(w.sum())
        dos_at_EF = float(W_dos.sum()) * area_prefactor

        # Spin-valley polarization order parameter.
        # The 32-orbital basis is laid out as 4 (spin × valley) blocks
        # of 8 orbitals each.  Compute total occupation per (s,v) block,
        # both for the running density (P) and reference (refP).  The
        # excess electron count per block is Δ_block.  For an ideal SVP
        # state, all doping sits in one block (Δ_max = doped_total,
        # others ≈ 0) and `svp_op` is close to 1; for PM-like, doping
        # spreads uniformly across all 4 blocks and svp_op → 0.
        diag_P = np.einsum("ij,ijaa->a", w, P_arr).real
        diag_R = np.einsum("ij,ijaa->a", w, np.asarray(refP)).real
        n_orb_block = 8  # (4 layer × 2 sublattice) per (s, v) block
        n_blocks = diag_P.shape[0] // n_orb_block
        delta = np.array([
            (diag_P[i*n_orb_block:(i+1)*n_orb_block].sum()
             - diag_R[i*n_orb_block:(i+1)*n_orb_block].sum())
            for i in range(n_blocks)
        ])
        total_doping = float(np.sum(delta))
        max_block = float(np.max(delta))
        # Normalized SVP order parameter:
        # 0 → uniform across spin-valley blocks (PM-like)
        # 1 → all doping in one block (perfect SVP)
        denom = total_doping if abs(total_doping) > 1e-12 else 1e-12
        svp_op = float((max_block - total_doping / n_blocks)
                       / (denom * (n_blocks - 1) / n_blocks))
        # Existing imbalance ratio (for back-compat / sanity)
        idx = int(np.argmax(np.abs(delta)))
        others = np.delete(delta, idx)
        imb = float(abs(delta[idx]) / max(np.mean(np.abs(others)), 1e-12))

        out_rows.append({
            "n": float(n_cm12), "D": float(D_Vnm),
            "energy": float(res.energy),
            "mu": float(res.mu),
            "delta_tb": float(delta_tb),
            "dos_at_EF": float(dos_at_EF),
            "iters": int(res.n_iter),
            "converged": bool(res.converged),
            "imbalance": float(imb),
            "svp_op": float(svp_op),
            "ref_failed": False, "error": "",
            "wall_s": float(dt),
        })
        last_P = P_arr  # warm-start chain
        del kernel, res, Sigma, H, fock_rel, fock, eigs, W_dos
        gc.collect()

    return out_rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-grid", type=int, default=30)
    p.add_argument("--D-grid", type=int, default=30)
    p.add_argument("--n-min", type=float, default=0.10)
    p.add_argument("--n-max", type=float, default=1.50)
    p.add_argument("--D-min", type=float, default=0.05)
    p.add_argument("--D-max", type=float, default=1.0)
    p.add_argument("--nk", type=int, default=48)
    p.add_argument("--kmax", type=float, default=0.30)
    p.add_argument("--T", type=float, default=0.3)
    p.add_argument("--init-scale", type=float, default=50.0)
    p.add_argument("--hartree-kind", type=str, default="dualgate_eps2")
    p.add_argument("--max-iter-ref", type=int, default=200,
                    help="iter cap for PM_C3 CN reference SCF")
    p.add_argument("--max-iter-hf", type=int, default=25,
                    help="iter cap for per-cell SVP_C3 SCF")
    p.add_argument("--tol-E", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--output", type=Path,
                    default=REPO_ROOT / "examples" / "outputs" /
                    "quartermetal_grid_svp_c3.npz")
    args = p.parse_args()

    n_grid = np.linspace(args.n_min, args.n_max, args.n_grid)
    D_grid = np.linspace(args.D_min, args.D_max, args.D_grid)

    print(f"Running {args.D_grid}×{args.n_grid} SVP_C3 grid:", flush=True)
    print(f"  n ∈ [{args.n_min:.3f}, {args.n_max:.3f}]")
    print(f"  D ∈ [{args.D_min:.3f}, {args.D_max:.3f}]")
    print(f"  nk={args.nk}, kmax={args.kmax}, T={args.T}")
    print(f"  hartree_kind={args.hartree_kind}, max_iter_hf={args.max_iter_hf}, "
          f"max_iter_ref={args.max_iter_ref}, workers={args.workers}")

    work = [
        (float(D), n_grid, int(args.nk), float(args.kmax),
         float(args.T), args.hartree_kind, float(args.init_scale),
         int(args.max_iter_ref), int(args.max_iter_hf), float(args.tol_E))
        for D in D_grid
    ]

    results: dict[float, list[dict]] = {}
    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init) as pool:
        for i, rows in enumerate(pool.imap_unordered(_run_d_row, work)):
            D = rows[0]["D"]
            results[D] = rows
            n_done = sum(1 for _, rs in results.items() for _ in rs)
            n_conv = sum(1 for _, rs in results.items() for r in rs if r["converged"])
            n_total = len(D_grid) * len(n_grid)
            t_elapsed = time.perf_counter() - t_start
            t_eta = t_elapsed / max(n_done, 1) * (n_total - n_done)
            print(f"  done D={D:.3f}: {n_done}/{n_total} cells, "
                  f"{n_conv} conv, t={t_elapsed/60:.1f}m, "
                  f"ETA={t_eta/60:.1f}m", flush=True)

    n_grid_arr = n_grid
    D_grid_arr = np.asarray(sorted(results.keys()))
    keys = ("energy", "mu", "delta_tb", "dos_at_EF", "iters",
            "converged", "imbalance", "svp_op", "ref_failed", "wall_s")
    arrays = {}
    for k in keys:
        a = np.full((len(D_grid_arr), len(n_grid_arr)),
                    np.nan if k not in ("converged", "ref_failed", "iters")
                          else (False if k != "iters" else -1))
        for i, D in enumerate(D_grid_arr):
            rows = results[D]
            for j, r in enumerate(rows):
                a[i, j] = r[k]
        arrays[k] = a

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, n_grid=n_grid_arr, D_grid=D_grid_arr,
             hartree_kind=args.hartree_kind, max_iter_hf=args.max_iter_hf,
             nk=args.nk, kmax=args.kmax, T=args.T, **arrays)
    print(f"Saved {out_path}")

    _plot(out_path, args)
    return 0


def _plot(npz_path: Path, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path, allow_pickle=True)
    n_grid = data["n_grid"]
    D_grid = data["D_grid"]
    dos = data["dos_at_EF"]
    delta_tb = data["delta_tb"]
    converged = data["converged"]
    iters = data["iters"]
    svp_op = data["svp_op"]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    extent = (n_grid[0], n_grid[-1], D_grid[0], D_grid[-1])

    # Panel 1: log10(DOS at E_F)
    log_dos = np.log10(np.where(dos > 0, dos, np.nan))
    im0 = axes[0].imshow(log_dos, origin="lower", aspect="auto",
                          extent=extent, cmap="viridis")
    fig.colorbar(im0, ax=axes[0], label=r"$\log_{10}(\rho(E_F))$")
    axes[0].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    axes[0].set_ylabel(r"$D$ (V/nm)")
    axes[0].set_title("SVP_C3: DOS at $E_F$")

    # Panel 2: Δ_tb (physical)
    im1 = axes[1].imshow(delta_tb, origin="lower", aspect="auto",
                          extent=extent, cmap="RdBu_r",
                          vmin=-np.nanmax(np.abs(delta_tb)),
                          vmax=np.nanmax(np.abs(delta_tb)))
    fig.colorbar(im1, ax=axes[1], label=r"$\Delta_{\rm tb}$ (meV)")
    axes[1].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    axes[1].set_ylabel(r"$D$ (V/nm)")
    axes[1].set_title("SVP_C3: top - bottom layer potential")

    # Panel 3: convergence
    iters_clipped = np.where(converged, iters, np.nan)
    im2 = axes[2].imshow(iters_clipped, origin="lower", aspect="auto",
                          extent=extent, cmap="cividis",
                          vmin=1, vmax=int(args.max_iter_hf))
    fig.colorbar(im2, ax=axes[2], label="iters to converge")
    axes[2].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    axes[2].set_ylabel(r"$D$ (V/nm)")
    axes[2].set_title(f"SVP_C3 convergence (cap={args.max_iter_hf})")
    not_conv = ~converged
    axes[2].contour(np.linspace(extent[0], extent[1], converged.shape[1]),
                     np.linspace(extent[2], extent[3], converged.shape[0]),
                     not_conv.astype(float), levels=[0.5],
                     colors="red", linewidths=0.5)

    # Panel 4: SVP order parameter
    # 0 = uniform spin-valley distribution (PM-like)
    # 1 = all doping in one (s, v) block (perfect SVP)
    im3 = axes[3].imshow(svp_op, origin="lower", aspect="auto",
                          extent=extent, cmap="magma",
                          vmin=0, vmax=1)
    fig.colorbar(im3, ax=axes[3], label="SVP order parameter")
    axes[3].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    axes[3].set_ylabel(r"$D$ (V/nm)")
    axes[3].set_title(r"SVP order parameter $\equiv$ "
                       r"(max$_b$ Δσ$_b$ - mean) / fully-polarized")

    fig.suptitle(f"SVP_C3 grid: nk={args.nk}, kmax={args.kmax}, T={args.T}, "
                 f"ε-kind={args.hartree_kind}, "
                 f"max_iter_hf={args.max_iter_hf}")
    out_png = npz_path.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    raise SystemExit(main())
