"""Reproduces ``examples/outputs/quartermetal_phase_portrait.png`` (and the
sibling Fermi-surface and band-structure figures) by driving the SCF
through ``contimod`` end-to-end.

Self-contained: no dependency on ``examples/tetralayer_quartermetal/_common.py``.
The model, the rhombic Brillouin-zone patch, the layer-resolved Coulomb
interaction, the dual-gate Hartree matrix, the symmetry projectors
(PM_C3 / C3 via ``contimod.meanfield.symmetry.build_graphene_projections``),
and the SCF itself (``MeshgridHamiltonian.solve_meanfield(..., backend="cpp_hf")``)
are all built directly from the contimod public API.

This script uses the new ``embed_reference_potential=True`` kernel
option (added across cpp_hf / jax_hf / contimod): the kernel pre-bakes
``V_HF[refP] = J[refP] + Σ[refP]`` into ``h`` once at construction so
the SCF natively solves *standard* HF (gradient ``F = h + V_HF[P]``)
instead of the reference-subtracted modified-HF default.  The Fock
returned in ``result.hmf.hs`` is therefore the standard physical Fock
directly — no manual reconstruction (no Σ_x[refP] precompute, no
HH·diag(refP) diagonal addition).  ``center_embedded_hartree=True``
subtracts the orbital-uniform mean of J[refP] so absolute eigenvalues
sit near 0, then μ is found self-consistently and bands sit on a
sensible energy scale.

The only direct ``cpp_hf.utils`` import remaining is
``find_chemical_potential`` for re-evaluating μ against the physical
eigenvalues — convenient because contimod's solver result already
carries μ but we re-find it for consistency with the original
quartermetal-study convention.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import contimod as cm
from contimod.meanfield.api import DensitySeed, SelfEnergySeed
from contimod.meanfield.coulomb import (
    dualgate_hartree_q0_matrix,
    layer_coulomb_kernel,
)
from contimod.meanfield.symmetry import build_graphene_projections
from contimod.utils.spectrum_fermi import FermiParams

from cpp_hf.utils import find_chemical_potential


# ---- Physical constants (tetralayer ABC) --------------------------------
N_LAYERS = 4
LAT_NM = 0.246
LAYER_SPACING_NM = 0.34
PER_CM = 0.246e-7
EPSILON_ZZ_DEFAULT = 2.0
GATE_DISTANCE_NM = 30.0


# ---- SCF settings -------------------------------------------------------
T = 0.3
NK = 96
KMAX = 0.35
MAX_ITER = 25
MAX_ITER_REF = 200
TOL_E = 1e-4
INIT_SCALE = 50.0  # SVP-seed scale, in meV


CELLS = [
    (0.20, 0.30, "low-n / low-D"),
    (0.30, 1.00, "low-n / high-D"),
    (0.40, 0.50, "boundary / mid"),
    (0.60, 0.70, "SVP transition (D~0.7)"),
    (0.80, 0.50, "metallic / mid-D"),
    (0.80, 1.00, "metallic / high-D"),
    (1.20, 0.80, "high-n / high-D"),
]


# ---- Setup helpers (replace _common.build_setup) -----------------------

def _U_meV_from_D(D_Vnm: float) -> float:
    """Spec layer-bias drop (top minus bottom) in meV at displacement field D."""
    return D_Vnm * (N_LAYERS - 1) * LAYER_SPACING_NM / EPSILON_ZZ_DEFAULT * 1000.0


def _z_aG(model) -> np.ndarray:
    """Per-orbital z coordinates in units of graphene a_G (= LAT_NM nm)."""
    z_norm = np.asarray(model.layer)
    z_phys_nm = z_norm * (N_LAYERS - 1) * LAYER_SPACING_NM / 2.0
    return z_phys_nm / LAT_NM


def _eps_from_kind(hartree_kind: str) -> tuple[float, float]:
    """Map ``"dualgate_eps<N>"`` to (eps_perp, eps_zz) — both equal to N."""
    if hartree_kind.startswith("dualgate_eps"):
        eps_val = float(hartree_kind.removeprefix("dualgate_eps"))
        return eps_val, eps_val
    raise ValueError(f"Unsupported hartree_kind {hartree_kind!r}")


def _rhombic_bz(kmax: float) -> "cm.kdiscrete.BrillouinZone":
    """60° rhombic graphene-K patch with reciprocal basis (s/2, ∓s√3/2)."""
    s = 2.0 * float(kmax)
    h_tri = s * np.sqrt(3.0) / 2.0
    Bmat = np.array([[s / 2.0, s / 2.0], [-h_tri, h_tri]])
    return cm.kdiscrete.BrillouinZone(B=Bmat)


def _build_setup(D_Vnm: float, *, hartree_kind: str = "dualgate_eps2") -> dict:
    """Build the full tetralayer / rhombic-patch setup.

    Returns a dict carrying the model, discretised Hamiltonian, Coulomb,
    dual-gate Hartree matrix, the PM/SVP seeds, the C3-aware symmetry
    projectors, and the per-flavor charge-neutrality density.
    """
    eps_perp, eps_zz = _eps_from_kind(hartree_kind)
    U_spec = _U_meV_from_D(D_Vnm)

    model = cm.graphene.NlayerABC(
        N=N_LAYERS, valleyful=True, spinful=True, U=-U_spec,
    )
    bz = _rhombic_bz(KMAX)
    h = model.discretize(nk=NK, bz=bz)
    h.fermi = FermiParams(T=T, mu=0.0)

    # Layer-resolved Coulomb V(q).
    Vq = layer_coulomb_kernel(
        np.asarray(h.kmesh.distance_grid), _z_aG(model),
        epsilon_zz=eps_zz, epsilon_perp=eps_perp, a_nm=LAT_NM,
    )
    Vq = np.ascontiguousarray(np.asarray(Vq), dtype=np.float64)

    # Dual-gate q=0 Hartree matrix.
    HH = dualgate_hartree_q0_matrix(
        _z_aG(model), d_gate=GATE_DISTANCE_NM / LAT_NM,
        epsilon_r=eps_perp, epsilon_r_perp=eps_zz, a=LAT_NM,
    )
    HH = np.ascontiguousarray(np.asarray(HH), dtype=np.float64)

    # Seeds: PM = zero (cold-fill); SVP = trace-preserving spin-valley
    # polariser scaled by INIT_SCALE meV.
    identity = np.asarray(model.identity)
    s3 = np.asarray(model.spin_op(3))
    v3 = np.asarray(model.valley_op(3))
    proj_pol = 0.25 * (identity + s3) @ (identity + v3)
    sv_seed_shape = (-1.0) * proj_pol + (1.0 / 3.0) * (identity - proj_pol)
    pm_seed_op = h.get_operator("zero")
    svp_seed_op = float(INIT_SCALE) * h.get_operator(sv_seed_shape)

    # C3-aware symmetry projectors (PM_C3, C3, SVP_C3, ...) via contimod.
    # The QM convention treats the "SVP_C3" *phase* as C3-projected only —
    # SVP comes from the SVP seed, not a projector — so we route the
    # SVP_C3-phase project_fn to ``projections["C3"]``.
    projections = build_graphene_projections(model, ks=np.asarray(h.kmesh.ks))
    project_fns = {
        "PM_C3": projections["PM_C3"],
        "SVP_C3": projections["C3"],
    }

    # Per-flavor charge-neutrality density.
    ne_cn = float(h.state.compute_density_from_filling(0.5 - 1e-6))

    return dict(
        model=model, h=h, Vq=Vq, HH=HH,
        pm_seed_op=pm_seed_op, svp_seed_op=svp_seed_op,
        project_fns=project_fns,
        ne_cn=ne_cn, hartree_kind=hartree_kind,
    )


def _noninteracting_cn_density(setup: dict) -> np.ndarray:
    """U=0 non-interacting CN density on the same mesh as ``setup['h']``."""
    bz = setup["h"].kmesh.brillouin_zone
    m_u0 = cm.graphene.NlayerABC(
        N=N_LAYERS, valleyful=True, spinful=True, U=0.0,
    )
    h_u0 = m_u0.discretize(nk=NK, bz=bz)
    h_u0.fermi = FermiParams(T=T, mu=0.0)
    P, _ = h_u0.state.compute_densitymatrix_for_density(setup["ne_cn"], T=T)
    return np.ascontiguousarray(np.asarray(P), dtype=np.complex128)


def _pm_c3_cn_reference(setup: dict) -> np.ndarray:
    """Self-consistent PM_C3 density at charge neutrality (used as refP).

    Uses the same ``embed_reference_potential=True`` convention as the
    production cells, so this density is the *standard* HF minimum at
    CN under the PM_C3 projector (not the reference-subtracted
    modified-HF minimum).  The bootstrap-only refP for this SCF is the
    non-interacting CN density at U=0.
    """
    refP_init = _noninteracting_cn_density(setup)
    h_cn = setup["h"].copy()
    h_cn.compute_chemicalpotential(density=setup["ne_cn"])
    res = h_cn.solve_meanfield(
        setup["Vq"],
        backend="cpp_hf", solver="dm",
        seed=DensitySeed(refP_init),
        include_hartree=True, include_exchange=True,
        reference_density=refP_init,
        hartree_matrix_override=setup["HH"],
        embed_reference_potential=True,
        center_embedded_hartree=True,
        config={
            "max_iter": MAX_ITER_REF, "tol_E": TOL_E, "max_step": 0.6,
            "block_sizes": (8, 8, 8, 8),
            "project_fn": setup["project_fns"]["PM_C3"],
        },
    )
    if not res.converged:
        raise RuntimeError(
            f"PM_C3 CN reference SCF did not converge ({res.n_iter} iters)"
        )
    return np.ascontiguousarray(
        np.asarray(res.density_matrix), dtype=np.complex128,
    )


def _h_run_at_density(setup: dict, n_cm12: float):
    """Copy ``setup['h']`` and set μ for the requested doping above CN."""
    dd = float(n_cm12) * 1e12 * (PER_CM ** 2)
    h_run = setup["h"].copy()
    h_run.compute_chemicalpotential(density=setup["ne_cn"] + dd)
    n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))
    return n_e, h_run


def _solve_meanfield_phase(setup: dict, h_run, *, seed_op, project_fn,
                            refP, max_iter: int = MAX_ITER):
    """One contimod SCF call for a single (cell, phase).

    Uses ``embed_reference_potential=True`` so the converged Fock in
    ``result.hmf.hs`` is the *standard* HF Fock — no post-processing
    required to recover bands / DOS / Δ_tb.
    """
    return h_run.solve_meanfield(
        setup["Vq"],
        backend="cpp_hf", solver="dm",
        seed=SelfEnergySeed(seed_op),
        include_hartree=True, include_exchange=True,
        reference_density=refP,
        hartree_matrix_override=setup["HH"],
        embed_reference_potential=True,
        center_embedded_hartree=True,
        config={
            "max_iter": int(max_iter), "tol_E": TOL_E, "max_step": 0.6,
            "block_sizes": (8, 8, 8, 8),
            "project_fn": project_fn,
        },
    )


def _physical_eigs(res, weights, n_e):
    """Diagonalize the standard-HF Fock from the embedded SCF result.

    With ``embed_reference_potential=True`` the SCF result's
    ``hmf.hs`` already equals ``h_bare + V_HF[P]`` (i.e. F_std), so
    this is just an eigvalsh + μ search — no Σ_x[refP] / HH·diag(refP)
    reconstruction.
    """
    fock = np.asarray(res.hmf.hs)
    n_orb = fock.shape[-1]
    eigs = np.linalg.eigvalsh(fock.reshape(-1, n_orb, n_orb))
    eigs = eigs.reshape(*fock.shape[:2], -1)
    E_F = float(find_chemical_potential(
        eigs, weights, n_e, T, method="bisection",
    ))
    return eigs, E_F


def _k_y0_slice(eigs: np.ndarray, ks: np.ndarray):
    """Bands along the k_y = 0 mirror diagonal (i = j) of the rhombic patch."""
    nk = eigs.shape[0]
    diag = np.arange(nk)
    return ks[diag, diag, 0], eigs[diag, diag, :]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=int, default=2,
                        choices=(2, 3, 4, 5, 10, 20),
                        help="Dual-gate dielectric constant.")
    parser.add_argument("--svp-grid", type=Path, default=None,
                        help="Optional path to a precomputed SVP_C3 DOS-at-EF "
                             "grid npz used as the phase-portrait background.")
    args = parser.parse_args(argv)

    hartree_kind = f"dualgate_eps{args.eps}"
    suffix = "" if args.eps == 2 else f"_eps{args.eps}"
    out_dir = REPO_ROOT / "examples" / "outputs"
    grid_npz = args.svp_grid if args.svp_grid is not None else (
        out_dir / f"quartermetal_grid_svp_c3{suffix}.npz"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from contimod.plotting.meshgrid import plot_fermisurface

    setup_cache: dict[float, tuple] = {}
    cell_data: list[tuple] = []

    print(f"Running SCFs at representative cells "
          f"(hartree_kind={hartree_kind})...", flush=True)
    for n_cm12, D_Vnm, label in CELLS:
        print(f"  ({n_cm12}, {D_Vnm}) — {label}", flush=True)
        if D_Vnm not in setup_cache:
            setup = _build_setup(D_Vnm, hartree_kind=hartree_kind)
            refP = _pm_c3_cn_reference(setup)
            setup_cache[D_Vnm] = (setup, refP)
        setup, refP = setup_cache[D_Vnm]

        n_e, h_run = _h_run_at_density(setup, n_cm12)
        ks = np.asarray(h_run.kmesh.ks)
        bounds = np.asarray(h_run.kmesh.bounds)
        weights = np.asarray(h_run.kmesh.weights)

        results_per_phase = {}
        for proj_key, seed_attr in [("PM_C3", "pm_seed_op"),
                                      ("SVP_C3", "svp_seed_op")]:
            t0 = time.perf_counter()
            res = _solve_meanfield_phase(
                setup, h_run,
                seed_op=setup[seed_attr],
                project_fn=setup["project_fns"][proj_key],
                refP=refP,
            )
            eigs, E_F = _physical_eigs(res, weights, n_e)
            kxs, bands = _k_y0_slice(eigs, ks)
            results_per_phase[proj_key] = dict(
                ks=ks, bounds=bounds, eigs=eigs, E_F=E_F,
                kxs=kxs, bands=bands,
                converged=bool(res.converged), n_iter=int(res.n_iter),
            )
            print(f"    {proj_key}: it={res.n_iter} conv={int(res.converged)} "
                  f"E_F={E_F:.1f}  t={time.perf_counter()-t0:.1f}s",
                  flush=True)
        cell_data.append((n_cm12, D_Vnm, label, results_per_phase))

    # ---- Plot 1: Fermi surfaces ----------------------------------------
    fig_fs, axes_fs = plt.subplots(
        len(CELLS), 2, figsize=(11, 4.5 * len(CELLS)),
        constrained_layout=True,
    )
    for row, (n_cm12, D_Vnm, label, phases) in enumerate(cell_data):
        for col, proj_key in enumerate(["PM_C3", "SVP_C3"]):
            d = phases[proj_key]
            plot_fermisurface(
                d["ks"][..., 0], d["ks"][..., 1],
                d["eigs"], energies=[d["E_F"]],
                ax=axes_fs[row, col], bounds=d["bounds"], linewidths=1.4,
            )
            axes_fs[row, col].set_title(
                f"{proj_key}: (n={n_cm12}, D={D_Vnm}) — {label}\n"
                f"iters={d['n_iter']}, conv={int(d['converged'])}, "
                f"E_F={d['E_F']:.1f} meV", fontsize=9,
            )
    fig_fs.suptitle(
        f"Fermi surfaces at representative cells (ε={hartree_kind}, "
        f"nk={NK}, kmax={KMAX}, T={T} meV) — contimod end-to-end",
        fontsize=11,
    )
    out_fs = out_dir / f"quartermetal_fermi_surfaces{suffix}.png"
    fig_fs.savefig(out_fs, dpi=140)
    plt.close(fig_fs)
    print(f"\nSaved {out_fs}")

    # ---- Plot 2: Band structures (k_y = 0 slice, centered on E_F) ------
    fig_bs, axes_bs = plt.subplots(
        len(CELLS), 2, figsize=(11, 3.0 * len(CELLS)),
        constrained_layout=True,
    )
    yzoom = 50.0
    for row, (n_cm12, D_Vnm, label, phases) in enumerate(cell_data):
        for col, proj_key in enumerate(["PM_C3", "SVP_C3"]):
            d = phases[proj_key]
            ax = axes_bs[row, col]
            de = d["bands"] - d["E_F"]
            in_window = np.abs(de) <= yzoom * 1.5
            for b in range(d["bands"].shape[1]):
                mask = in_window[:, b]
                ax.scatter(d["kxs"][mask], de[mask, b],
                            s=6, color="k", alpha=0.5, edgecolors="none")
            ax.axhline(0.0, color="tab:red", lw=0.8, linestyle="dashed")
            ax.set_ylim(-yzoom, yzoom)
            ax.set_xlim(d["kxs"].min(), d["kxs"].max())
            ax.set_xlabel(r"$k_x$ (1/lat)")
            ax.set_ylabel(r"$E - E_F$ (meV)")
            ax.set_title(f"{proj_key}: (n={n_cm12}, D={D_Vnm}) — {label}",
                          fontsize=9)
            ax.grid(True, alpha=0.3)
    fig_bs.suptitle(
        f"Band structure scatter at $k_y=0$, centered on $E_F$ "
        f"(ε={hartree_kind}, nk={NK}, ±{yzoom:.0f} meV)", fontsize=11,
    )
    out_bs = out_dir / f"quartermetal_band_structures{suffix}.png"
    fig_bs.savefig(out_bs, dpi=140)
    plt.close(fig_bs)
    print(f"Saved {out_bs}")

    # ---- Plot 3: Phase portrait — SVP_C3 DOS map with FS insets --------
    grid = np.load(grid_npz, allow_pickle=True)
    n_grid = grid["n_grid"]
    D_grid = grid["D_grid"]
    extent = (n_grid[0], n_grid[-1], D_grid[0], D_grid[-1])
    dos_arr = np.where(grid["dos_at_EF"] > 0, grid["dos_at_EF"], np.nan)
    p97 = float(np.nanpercentile(dos_arr, 97))

    fig_pp, ax_pp = plt.subplots(figsize=(13, 11), constrained_layout=False)
    im = ax_pp.imshow(dos_arr, origin="lower", aspect="auto",
                       extent=extent, cmap="viridis", vmin=0, vmax=p97)
    fig_pp.colorbar(
        im, ax=ax_pp,
        label=r"SVP_C3 $\rho(E_F)$ (linear, clipped at p97)",
        shrink=0.5, location="left",
    )
    ax_pp.set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    ax_pp.set_ylabel(r"$D$ (V/nm)")
    ax_pp.set_title("SVP_C3 phase diagram with Fermi-surface insets")

    n_span = float(n_grid[-1] - n_grid[0])
    D_span = float(D_grid[-1] - D_grid[0])
    for n_cm12, D_Vnm, label, phases in cell_data:
        ax_pp.plot(n_cm12, D_Vnm, "o", color="white",
                    markersize=8, markeredgecolor="black", zorder=5)
        nx = (n_cm12 - n_grid[0]) / n_span
        ny = (D_Vnm - D_grid[0]) / D_span
        dx = 0.06 if nx < 0.7 else -0.20
        dy = 0.06 if ny < 0.7 else -0.20
        ax_in = ax_pp.inset_axes(
            [nx + dx, ny + dy, 0.13, 0.13],
            transform=ax_pp.transAxes,
        )
        d = phases["SVP_C3"]
        plot_fermisurface(
            d["ks"][..., 0], d["ks"][..., 1],
            d["eigs"], energies=[d["E_F"]],
            ax=ax_in, bounds=d["bounds"], linewidths=0.8,
        )
        ax_in.set_xlabel(""); ax_in.set_ylabel("")
        ax_in.set_xticks([]); ax_in.set_yticks([])
        ax_in.set_title("", fontsize=0)
        ax_pp.annotate(
            "", xy=(n_cm12, D_Vnm), xycoords="data",
            xytext=(nx + dx + 0.065, ny + dy + 0.065),
            textcoords=ax_pp.transAxes,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
        )

    out_pp = out_dir / f"quartermetal_phase_portrait{suffix}.png"
    fig_pp.savefig(out_pp, dpi=140, bbox_inches="tight")
    plt.close(fig_pp)
    print(f"Saved {out_pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
