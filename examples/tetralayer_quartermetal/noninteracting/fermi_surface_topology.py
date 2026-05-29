"""Fermi-surface topology phase diagram in the (n_e, Δ_tb) plane.

Reproduces the structure of Figure 1(c) in Huang & Wolf (2026): each
cell of the (n_e, Δ_tb) plane is classified by counting connected
components of the active conduction band's filled region — the iso-
contour ε_c(k) = μ on a dense k-patch.  Each cell is labelled (C, H)
with

  C = # connected components of the filled (electron) region,
  H = # connected components of the empty region inside the active patch
      (i.e. holes within the filled region),

giving the five classes that appear in the working window:

  C = 1, H = 0    simply connected                     (single pocket)
  C = 1, H = 1    annular                              (Fermi sea + hole)
  C = 3, H = 0    three pockets                        (one per C₃ rotation)
  C = 1, H = 3    1-component-with-3-holes  (C₁H₃)
  C = 4, H = 0    four pockets              (C₄H₀)

Conventions:

  - x-axis: per-flavor density n_e (10¹² cm⁻²).  Total density passed
    to the chemical-potential solver is n_total = 4 · n_e (spin × valley).
  - y-axis: Δ_tb (meV) is our top-bottom layer-bias potential.  The
    paper's V_z parameterises a per-step layer increment, so paper
    V_z = Δ_tb / 3.  Defaults span Δ_tb ∈ [90, 150] meV ≡ V_z ∈ [30, 50].

Self-contained: only imports _common (sibling) for the SWMcC model
setup and density helper.  Marks an optional working-point star.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import _common as qm  # noqa: E402


# Five classes recognised in the working window.
CLASS_LABELS = {
    "simply": "simply conn.",
    "annular": "annular",
    "three": "three pockets",
    "C1H3": r"$C_1 H_3$",
    "C4H0": r"$C_4 H_0$",
    "other": "other",
    "empty": "empty (gap)",
}
CLASS_COLORS = {
    "simply": "#5fb95f",     # green
    "annular": "#fbb13c",    # orange
    "three": "#3f7fbf",      # blue
    "C1H3": "#a06fc0",       # purple
    "C4H0": "#ec4040",       # red
    "other": "#888888",      # gray
    "empty": "#dddddd",      # light gray
}


def classify(C: int, H: int) -> str:
    if C == 1 and H == 0:
        return "simply"
    if C == 1 and H == 1:
        return "annular"
    if C == 3 and H == 0:
        return "three"
    if C == 1 and H == 3:
        return "C1H3"
    if C == 4 and H == 0:
        return "C4H0"
    return "other"


def fs_topology_for_cell(model, mu: float, *, nk: int, kmax: float
                          ) -> tuple[int, int]:
    """Classify the active-band Fermi surface at (model, μ) by (C, H).

    Diagonalises h(k) on a Cartesian (nk, nk) patch, identifies the
    active conduction band as the band whose minimum is closest to μ
    from below, and counts connected components of {ε(k) < μ} (= C) and
    of {ε(k) > μ} bounded inside the patch (= H).
    """
    from scipy.ndimage import label

    h = model.discretize(nk=int(nk), kmax=float(kmax))
    h_arr = np.asarray(h.hs)
    eigs = np.linalg.eigvalsh(
        h_arr.reshape(-1, h_arr.shape[-2], h_arr.shape[-1])
    ).reshape(*h_arr.shape[:2], -1)

    # Active conduction band: lowest band whose min is below μ AND whose
    # max is above μ (the band that the chemical potential cuts).
    band_min = eigs.min(axis=(0, 1))
    band_max = eigs.max(axis=(0, 1))
    active = np.where((band_min < mu) & (band_max > mu))[0]
    if active.size == 0:
        return (0, 0)
    cb = int(active.min())
    band_E = eigs[..., cb]

    # 8-connectivity for label.
    structure = np.ones((3, 3), dtype=int)
    filled = band_E < mu
    empty = ~filled

    # C: count of filled-region components (open boundary OK — we just
    # count components inside the patch).
    _, C = label(filled, structure=structure)

    # H: count of empty-region components that are *enclosed* by filled
    # region — i.e. don't touch the patch boundary.  Components that
    # touch the boundary are part of "outside the FS" and don't count.
    lbl, n_components = label(empty, structure=structure)
    boundary_labels = set(np.unique(np.concatenate([
        lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1],
    ])))
    boundary_labels.discard(0)
    H = n_components - len(boundary_labels)
    return (int(C), int(H))


SPIN_VALLEY = 4
DELTA_TB_PER_D = 510.0  # meV per (V/nm) for tetralayer at our default ε_zz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=30,
                        help="# samples along n_e (per-flavor) axis.")
    parser.add_argument("--Delta-grid", type=int, default=30,
                        help="# samples along Δ_tb axis.")
    parser.add_argument("--n-min", type=float, default=0.25,
                        help="Min n_e per flavor (10¹² cm⁻²).")
    parser.add_argument("--n-max", type=float, default=1.2,
                        help="Max n_e per flavor (10¹² cm⁻²).")
    parser.add_argument("--Delta-min", type=float, default=90.0,
                        help="Min Δ_tb in meV.")
    parser.add_argument("--Delta-max", type=float, default=150.0,
                        help="Max Δ_tb in meV.")
    parser.add_argument("--T", type=float, default=0.3)
    parser.add_argument("--nk-classify", type=int, default=128,
                        help="Cartesian k-mesh size used for FS topology.")
    parser.add_argument("--kmax-classify", type=float, default=0.30,
                        help="Half-width of the classification patch (1/lat).")
    parser.add_argument("--working-point", type=float, nargs=2, default=None,
                        metavar=("N_E_PERFLAVOR", "DELTA_TB_MEV"),
                        help="Mark an (n_e, Δ_tb) star (per flavor, meV).")
    parser.add_argument("--output", type=Path,
                        default=OUT_DIR / "fs_topology.png")
    args = parser.parse_args(argv)

    import contimod as cm

    n_e_grid = np.linspace(args.n_min, args.n_max, args.n_grid)
    Delta_grid = np.linspace(args.Delta_min, args.Delta_max, args.Delta_grid)

    print(f"Classifying {args.Delta_grid}×{args.n_grid} cells "
          f"(nk={args.nk_classify}, kmax={args.kmax_classify})...", flush=True)
    print(f"  n_e (per flavor) ∈ [{args.n_min:.3f}, {args.n_max:.3f}]")
    print(f"  Δ_tb (meV)        ∈ [{args.Delta_min:.1f}, {args.Delta_max:.1f}]")
    t0 = time.perf_counter()
    classes = np.empty((len(Delta_grid), len(n_e_grid)), dtype=object)
    for i, Dtb in enumerate(Delta_grid):
        D_Vnm = float(Dtb) / DELTA_TB_PER_D
        # Build small_orbital model + setup once per Δ_tb row.
        model = cm.graphene.NlayerABC(
            N=qm.N_LAYERS, valleyful=False, spinful=False, U=-float(Dtb),
        )
        setup = qm.build_setup(
            D_Vnm=D_Vnm, nk=64, kmax=0.30, T=args.T,
            bz_kind="rhombic", small_orbital=True,
            hartree_kind="dualgate_eps2",
        )
        for j, n_e in enumerate(n_e_grid):
            n_cm12_total = float(n_e) * SPIN_VALLEY
            _, h_run = qm.n_electrons_for_density(setup, n_cm12_total, args.T)
            mu = float(h_run.fermi.mu)
            C, H = fs_topology_for_cell(
                model, mu,
                nk=int(args.nk_classify), kmax=float(args.kmax_classify),
            )
            classes[i, j] = classify(C, H) if C > 0 else "empty"
        n_done = (i + 1) * len(n_e_grid)
        n_total = len(Delta_grid) * len(n_e_grid)
        t = time.perf_counter() - t0
        print(f"  done Δ_tb={Dtb:6.2f} meV: {n_done}/{n_total}  t={t:.1f}s "
              f"ETA={t / max(n_done, 1) * (n_total - n_done):.1f}s",
              flush=True)

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    cls_to_int = {k: i for i, k in enumerate(CLASS_LABELS.keys())}
    arr = np.vectorize(cls_to_int.get)(classes)

    cmap = ListedColormap([CLASS_COLORS[k] for k in CLASS_LABELS.keys()])
    boundaries = np.arange(len(CLASS_LABELS) + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    extent = (n_e_grid[0], n_e_grid[-1], Delta_grid[0], Delta_grid[-1])
    ax.imshow(arr, origin="lower", aspect="auto", extent=extent,
              cmap=cmap, norm=norm)
    ax.set_xlabel(r"$n_e$ per flavor (× $10^{12}$ cm$^{-2}$)")
    ax.set_ylabel(r"$\Delta_{tb}$ (meV)")
    ax.set_title("Bare conduction-band Fermi-surface topology")

    handles = [Patch(facecolor=CLASS_COLORS[k], edgecolor="0.3",
                      label=CLASS_LABELS[k])
               for k in CLASS_LABELS.keys() if (arr == cls_to_int[k]).any()]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)

    if args.working_point is not None:
        n_wp, D_wp = args.working_point
        ax.plot(n_wp, D_wp, marker="*", color="gold", markersize=18,
                 markeredgecolor="black")

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
