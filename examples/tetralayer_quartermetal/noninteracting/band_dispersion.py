"""Bare-band dispersion at the K valley with DOS sidebar.

Reproduces the structure of Figure 1(b) in Huang & Wolf (2026): the
SWMcC tetralayer-graphene band structure along a k_x cut through the
K valley at fixed layer-bias potential Δ_tb, with the energy-resolved
DOS computed on a dense (k_x, k_y) Cartesian patch shown as a sidebar.
The two bands nearest the chemical potential — the active conduction
band and its valence partner — are highlighted; remote bands are drawn
in grey.

Conventions:

  - Δ_tb (meV) is the top-bottom layer-bias potential.  The paper's
    V_z parameterises a per-step layer increment, so paper V_z = Δ_tb / 3.
    Default Δ_tb = 123 meV ≡ V_z = 41 meV.
  - n_e (10¹² cm⁻²) is the per-flavor electron density.  Total density
    = 4 × n_e is what's passed to ``compute_chemicalpotential``.

Self-contained: only imports _common (sibling) for the SWMcC model
setup.  No SCF, no Hartree, no Fock — pure single-particle h(k)
diagonalisation.
"""
from __future__ import annotations

import argparse
import os
import sys
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


# Lattice constant (nm).  k in 1/lat → 1/nm by dividing by LAT_NM.
LAT_NM = qm.LAT_NM


def diag_along_kx_cut(model, kx_array: np.ndarray) -> np.ndarray:
    """Eigenvalues along the (k_x, 0) line at the K valley.

    Builds a 2D Cartesian patch of the model centered at K, picks the
    k_y = 0 row, and returns the sorted bands as (n_kx, n_orb).
    """
    # Use a 2D grid that includes k_y = 0 exactly on a centered linspace.
    nk_y = 4  # we only need one row, but discretize requires a 2D mesh
    kmax_x = float(np.max(np.abs(kx_array)))
    kmax_y = max(kmax_x * 0.05, 0.01)
    # Build a 1D k-array directly via NlayerABC.h(k); fall back to mesh
    # discretization if the model lacks that.
    try:
        H_at_k = np.stack([np.asarray(model.h(np.array([kx, 0.0])))
                            for kx in kx_array], axis=0)
    except Exception:  # pragma: no cover — older contimod
        h = model.discretize(nk=max(len(kx_array), 64), kmax=kmax_x)
        ks = np.asarray(h.kmesh.ks)
        iy0 = int(np.argmin(np.abs(ks[0, :, 1])))
        H_at_k = np.asarray(h.hs)[:, iy0, :, :]
    eigs = np.linalg.eigvalsh(H_at_k)
    return eigs


def dos_curve(model, *, nk: int, kmax: float,
              energies: np.ndarray) -> np.ndarray:
    """Energy-resolved DOS on a dense Cartesian (k_x, k_y) patch.

    Returns the DOS evaluated at each energy in ``energies`` using the
    triangulation-based 2D DOS from ``bztetra.twod``.
    """
    h = model.discretize(nk=int(nk), kmax=float(kmax))
    h_arr = np.asarray(h.hs)
    eigs = np.linalg.eigvalsh(
        h_arr.reshape(-1, h_arr.shape[-2], h_arr.shape[-1])
    ).reshape(*h_arr.shape[:2], -1)
    weights = np.asarray(h.kmesh.weights)
    s = 2.0 * float(kmax)
    Bmat = np.array([[s, 0.0], [0.0, s]])  # Cartesian patch
    import bztetra.twod as twod
    # Returns shape (n_energies, nx, ny, n_bands).  Integrate over k and bands.
    W = twod.density_of_states_weights(Bmat, eigs, energies)
    return W.sum(axis=(1, 2, 3)) * float(weights.sum())


SPIN_VALLEY = 4
DELTA_TB_PER_D = 510.0  # meV per (V/nm) for tetralayer at default ε_zz


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Delta-tb", type=float, default=123.0,
                        help="Top-bottom layer-bias potential in meV "
                             "(default 123, ≡ paper V_z = 41 meV).")
    parser.add_argument("--n-e", type=float, default=1.0,
                        help="Per-flavor carrier density in 10¹² cm⁻² "
                             "(default 1.0; total density = 4 × n_e).")
    parser.add_argument("--T", type=float, default=0.3, help="Temperature [meV].")
    parser.add_argument("--kmax-cut-nm", type=float, default=0.6,
                        help="Half-width of the k_x cut in 1/nm.")
    parser.add_argument("--n-kcut", type=int, default=401,
                        help="Number of points along the k_x cut.")
    parser.add_argument("--nk-dos", type=int, default=200,
                        help="2D grid size for the DOS sidebar.")
    parser.add_argument("--kmax-dos", type=float, default=0.30,
                        help="Half-width of the DOS-integration patch (1/lat).")
    parser.add_argument("--e-min", type=float, default=-150.0,
                        help="Lower energy bound on the band plot [meV].")
    parser.add_argument("--e-max", type=float, default=200.0,
                        help="Upper energy bound on the band plot [meV].")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path.")
    args = parser.parse_args(argv)

    # Build the SWMcC model with the requested layer bias.
    U_spec = float(args.Delta_tb)
    D_Vnm = U_spec / DELTA_TB_PER_D
    import contimod as cm
    model = cm.graphene.NlayerABC(
        N=qm.N_LAYERS, valleyful=False, spinful=False, U=-U_spec,
    )

    # 1D band cut along k_x at K (in 1/lat).
    kmax_lat = args.kmax_cut_nm * LAT_NM
    kx_lat = np.linspace(-kmax_lat, kmax_lat, args.n_kcut)
    eigs_cut = diag_along_kx_cut(model, kx_lat)  # (n_kx, n_orb)
    kx_nm = kx_lat / LAT_NM

    # Find chemical potential at the requested per-flavor density.
    setup = qm.build_setup(
        D_Vnm=D_Vnm, nk=64, kmax=0.30, T=args.T,
        bz_kind="rhombic", small_orbital=True, hartree_kind="dualgate_eps2",
    )
    n_cm12_total = float(args.n_e) * SPIN_VALLEY
    _, h_run = qm.n_electrons_for_density(setup, n_cm12_total, args.T)
    mu = float(h_run.fermi.mu)

    # DOS sidebar.
    energies = np.linspace(args.e_min, args.e_max, 800)
    dos = dos_curve(model, nk=args.nk_dos, kmax=args.kmax_dos, energies=energies)

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_b, ax_d) = plt.subplots(
        1, 2, figsize=(8.5, 5.0),
        gridspec_kw=dict(width_ratios=(3.5, 1.0), wspace=0.04),
        sharey=True, constrained_layout=True,
    )

    # Identify the two bands nearest μ at k = 0 (band-edge proxy).
    i0 = int(np.argmin(np.abs(kx_nm)))
    band_E_at_K = eigs_cut[i0]
    order = np.argsort(np.abs(band_E_at_K - mu))
    active_bands = sorted(order[:2].tolist())
    cb_idx = active_bands[1]  # higher of the two = conduction band
    vb_idx = active_bands[0]  # lower = valence band

    n_orb = eigs_cut.shape[1]
    for b in range(n_orb):
        if b == cb_idx:
            color, lw, z = "tab:red", 1.6, 4
        elif b == vb_idx:
            color, lw, z = "tab:blue", 1.6, 4
        else:
            color, lw, z = "0.7", 0.9, 2
        ax_b.plot(kx_nm, eigs_cut[:, b], color=color, lw=lw, zorder=z)

    ax_b.axhline(mu, color="k", ls="--", lw=0.8, zorder=5,
                  label=fr"$\mu = {mu:.1f}$ meV")
    ax_b.set_xlim(-args.kmax_cut_nm, args.kmax_cut_nm)
    ax_b.set_ylim(args.e_min, args.e_max)
    ax_b.set_xlabel(r"$k_x$ (nm$^{-1}$)")
    ax_b.set_ylabel(r"$E$ (meV)")
    ax_b.set_title(
        f"Bare 4LG bands at K  ·  $\\Delta_{{tb}}$ = {U_spec:.0f} meV  "
        f"·  $n_e$ = {args.n_e:g} × 10$^{{12}}$ cm$^{{-2}}$ per flavor"
    )
    ax_b.legend(loc="upper right", frameon=False, fontsize=9)
    ax_b.grid(True, alpha=0.3)

    ax_d.plot(dos, energies, color="0.2", lw=1.0)
    ax_d.fill_betweenx(energies, 0.0, dos, color="0.2", alpha=0.15)
    ax_d.axhline(mu, color="k", ls="--", lw=0.8)
    ax_d.set_xlim(left=0)
    ax_d.set_xlabel("DOS")
    ax_d.set_title("DOS")
    ax_d.grid(True, alpha=0.3)

    out = args.output
    if out is None:
        out = OUT_DIR / f"band_dispersion_Dtb{int(round(U_spec))}meV.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
