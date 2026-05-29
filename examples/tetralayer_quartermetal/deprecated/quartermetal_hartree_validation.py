#!/usr/bin/env python
"""Hartree-only validation for tetralayer rhombohedral graphene.

Five reference cells from the quartermetal task spec, each with a known
top-bottom layer-potential drop ``Delta_tb`` after self-consistent Hartree
screening.  The bare q=0 Hartree matrix has spectral radius ~1e6 meV; we
RPA-dress it with the non-interacting compressibility Pi at the target
chemical potential before handing off to ``solve_hartree_newton``.

This is the smallest unit that proves the divergent-Hartree solver works
on the tetralayer ABC physics — no exchange, no PM/SVP projector.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
for opt in (REPO_ROOT.parent / "contimod" / "src",
            REPO_ROOT.parent / "contimod_graphene" / "src"):
    if opt.exists() and str(opt) not in sys.path:
        sys.path.insert(0, str(opt))

import cpp_hf
from cpp_hf import (
    HartreeNewtonConfig, solve_hartree_newton,
    SCFConfig, solve_scf,
    SolverConfig, solve_direct_minimization,
)

import contimod as cm
from contimod.meanfield.coulomb import dualgate_hartree_q0_matrix
from contimod.utils.spectrum_fermi import FermiParams


# ---- Physical parameters from the task spec ----
N_LAYERS = 4
NK_DEFAULT = 48
KMAX = 0.30
TEMPERATURE = 1.0          # meV
EPSILON_ZZ = 2.0           # out-of-plane dielectric
EPSILON_PERP = 13.8        # in-plane dielectric
GATE_DISTANCE_NM = 30.0    # symmetric dual-gate, top = bottom
LAT_NM = 0.246             # graphene a (nm)
LAYER_SPACING_NM = 0.34    # interlayer separation (nm)
PER_CM = 0.246e-7          # graphene a in cm (1 / cm conversion)

# (n in 10^12 cm^-2, D in V/nm, reference Delta_tb in meV, tolerance)
VALIDATION_CELLS = [
    (0.0, 0.10,  +2.3, 0.5),
    (0.0, 0.30,  +8.7, 0.5),
    (1.5, 0.30, +18.0, 0.5),
    (0.0, 1.00, +85.0, 1.0),
    (1.5, 1.00, +135.0, 1.0),
]


# ---- Model + Coulomb setup ----

def U_meV_from_D(D_Vnm: float, *, N: int = N_LAYERS,
                 layer_spacing_nm: float = LAYER_SPACING_NM,
                 epsilon_zz: float = EPSILON_ZZ) -> float:
    """Spec layer-bias U in meV: V_top - V_bot = +U for D > 0.

    Spec formula: ``U = D·(N-1)·d / ε_zz`` (D in V/nm → V → meV).  Spec
    convention places top at ``+U/2`` and bottom at ``-U/2``, i.e.
    ``Δ_tb_bare = +U``.  Contimod's ``NlayerABC(U=...)`` uses the OPPOSITE
    sign (top at ``-U/2``); we flip when passing to ``NlayerABC``.
    """
    return D_Vnm * (N - 1) * layer_spacing_nm / epsilon_zz * 1000.0


def build_model(D_Vnm: float, *, nk: int, kmax: float = KMAX,
                T: float = TEMPERATURE):
    U_meV_spec = U_meV_from_D(D_Vnm)
    # Flip sign for contimod's opposite convention; bare Δ_tb (top − bot) is +U_spec.
    m = cm.graphene.NlayerABC(N=N_LAYERS, valleyful=True, spinful=True,
                               U=-U_meV_spec)
    h = m.discretize(nk=nk, kmax=kmax)
    h.fermi = FermiParams(T=float(T), mu=0.0)
    return m, h


def build_bare_HH(model, *,
                  gate_nm: float = GATE_DISTANCE_NM,
                  eps_perp: float = EPSILON_PERP,
                  eps_zz: float = EPSILON_ZZ,
                  aG_nm: float = LAT_NM) -> np.ndarray:
    """Bare q=0 dual-gate Hartree matrix on the (nb, nb) orbital grid."""
    z_norm = np.asarray(model.layer)
    z_phys_nm = z_norm * (N_LAYERS - 1) * LAYER_SPACING_NM / 2.0
    z_aG = z_phys_nm / aG_nm
    HH = dualgate_hartree_q0_matrix(
        z_aG, d_gate=gate_nm / aG_nm,
        epsilon_r=eps_perp, epsilon_r_perp=eps_zz, a=aG_nm,
    )
    return np.ascontiguousarray(HH, dtype=np.float64)


# ---- Single-particle helpers ----

def _eigh_all(h_array: np.ndarray):
    """Eigendecompose every k.  Returns (eps, V) with eps shape (nk1, nk2, nb)."""
    nk1, nk2, nb, _ = h_array.shape
    eps = np.empty((nk1, nk2, nb), dtype=np.float64)
    V = np.empty_like(h_array)
    for ik1 in range(nk1):
        for ik2 in range(nk2):
            hk = h_array[ik1, ik2]
            eps[ik1, ik2], V[ik1, ik2] = np.linalg.eigh(0.5 * (hk + hk.conj().T))
    return eps, V


def find_mu_for_density(eps: np.ndarray, w2d: np.ndarray,
                         n_target: float, T: float,
                         mu_min: float = -300.0, mu_max: float = 300.0,
                         max_iter: int = 80) -> float:
    Tsafe = max(T, 1e-12)
    w_b = w2d[..., None]

    def total(mu):
        x = np.clip((eps - mu) / Tsafe, -50.0, 50.0)
        f = 1.0 / (1.0 + np.exp(x))
        return float(np.sum(w_b * f))

    lo, hi = mu_min, mu_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if total(mid) < n_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return 0.5 * (lo + hi)


def cn_density(h_array: np.ndarray, w2d: np.ndarray,
                n_cn: float, T: float) -> np.ndarray:
    eps, V = _eigh_all(h_array)
    mu = find_mu_for_density(eps, w2d, n_cn, T)
    Tsafe = max(T, 1e-12)
    P = np.empty_like(h_array)
    for ik1 in range(eps.shape[0]):
        for ik2 in range(eps.shape[1]):
            f = 1.0 / (1.0 + np.exp(np.clip((eps[ik1, ik2] - mu) / Tsafe, -50, 50)))
            Vk = V[ik1, ik2]
            P[ik1, ik2] = (Vk * f) @ Vk.conj().T
    return 0.5 * (P + P.conj().swapaxes(-1, -2))


def compressibility_intraband(eps: np.ndarray, V: np.ndarray, w2d: np.ndarray,
                               mu: float, T: float) -> np.ndarray:
    """Intra-band Pi: sum_k w_k sum_n (-df/de)(e_n - mu) |V_n(a)|^2 |V_n(b)|^2.

    Spec formula.  Vanishes in gapped insulators with mu in the gap — for
    those cells use compressibility_full_lindhard instead.
    """
    Tsafe = max(T, 1e-12)
    nk1, nk2, nb = eps.shape
    Pi = np.zeros((nb, nb), dtype=np.float64)
    for ik1 in range(nk1):
        for ik2 in range(nk2):
            f = 1.0 / (1.0 + np.exp(np.clip((eps[ik1, ik2] - mu) / Tsafe, -50, 50)))
            neg_dfde = f * (1.0 - f) / Tsafe
            psi_sq = np.abs(V[ik1, ik2]) ** 2  # (nb_orb, nb_band)
            wn = w2d[ik1, ik2] * neg_dfde       # per-band weight
            Pi += psi_sq @ (wn[:, None] * psi_sq.T)
    return 0.5 * (Pi + Pi.T)


def compressibility_full_lindhard(eps: np.ndarray, V: np.ndarray,
                                   w2d: np.ndarray, mu: float, T: float,
                                   eta: float = 1e-3) -> np.ndarray:
    """Full static Lindhard Pi (intra + interband):

        Pi[a, b] = sum_k w_k sum_{n,m} M_{nm} V[a,n]* V[a,m] V[b,m]* V[b,n]

    where M_{nm} = (f_n - f_m) / (e_m - e_n)        for n != m
                  -df/de(e_n)                       for n = m

    For real-symmetric h this is real symmetric.  ``eta`` regularizes
    accidental near-degeneracies; sub-meV is fine since we evaluate at q=0.
    """
    Tsafe = max(T, 1e-12)
    nk1, nk2, nb = eps.shape
    Pi = np.zeros((nb, nb), dtype=np.float64)
    for ik1 in range(nk1):
        for ik2 in range(nk2):
            ek = eps[ik1, ik2]
            Vk = V[ik1, ik2]                # (nb_orb, nb_band)
            f = 1.0 / (1.0 + np.exp(np.clip((ek - mu) / Tsafe, -50, 50)))
            # Build M[n, m]
            de = ek[None, :] - ek[:, None]   # e_m - e_n
            df = f[:, None] - f[None, :]     # f_n - f_m
            with np.errstate(divide="ignore", invalid="ignore"):
                M = df / (de + np.where(np.abs(de) < eta, eta, 0.0))
            # Diagonal: -df/de = f(1-f)/T
            np.fill_diagonal(M, f * (1.0 - f) / Tsafe)
            # X[a, b, n] = V[a, n] * V[b, n].conj()  (build per pair)
            #
            # Pi[a,b] = sum_{n,m} M[n,m] * X[a,b,n].conj() * X[a,b,m]
            #        = X[a,b,:].conj() @ M @ X[a,b,:]
            #
            # Vectorized over (a, b) using einsum.
            # Vc[n, a] = V[a, n].conj()  (band first, orbital second)
            Vc = Vk.conj().T            # (nb_band, nb_orb)
            Vt = Vk.T                    # (nb_band, nb_orb)
            # X[n, a, b] = Vt[n, a] * Vc[n, b]   (band, orb_a, orb_b)
            X = Vt[:, :, None] * Vc[:, None, :]   # (nb_band, nb_orb, nb_orb)
            # MX[m, a, b] = sum_n M[n, m] * X[n, a, b].conj()  -> apply M to band axis
            # Actually we want sum_{nm} M[n,m] X*[n,a,b] X[m,a,b].
            # = sum_m X[m,a,b] * (sum_n M[n,m] X*[n,a,b])
            Xc = X.conj()
            # T1[m, a, b] = sum_n M[n, m] * Xc[n, a, b]
            T1 = np.einsum("nm,nab->mab", M, Xc)
            # Pi_k[a, b] = sum_m X[m, a, b] * T1[m, a, b]
            Pi_k = np.einsum("mab,mab->ab", X, T1).real
            Pi += w2d[ik1, ik2] * Pi_k
    return 0.5 * (Pi + Pi.T)


# Default: full Lindhard.  Set use_full_lindhard=False to recover spec formula.
def compressibility_at_mu(eps, V, w2d, mu, T, *, use_full_lindhard: bool = True):
    if use_full_lindhard:
        return compressibility_full_lindhard(eps, V, w2d, mu, T)
    return compressibility_intraband(eps, V, w2d, mu, T)


def rpa_dress(HH_bare: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Static RPA: HH_dressed = (I + HH_bare @ Pi)^{-1} @ HH_bare."""
    nb = HH_bare.shape[0]
    M = np.eye(nb) + HH_bare @ Pi
    return np.linalg.solve(M, HH_bare)


# ---- Output extraction ----

def delta_tb_from_fock(fock: np.ndarray, layer_idx: np.ndarray) -> float:
    """Top minus bottom layer-averaged diagonal of F (averaged over k)."""
    nk1, nk2 = fock.shape[0], fock.shape[1]
    diag = np.einsum("ijaa->a", fock).real / (nk1 * nk2)
    layers_sorted = np.sort(np.unique(layer_idx))
    bot = float(diag[layer_idx == layers_sorted[0]].mean())
    top = float(diag[layer_idx == layers_sorted[-1]].mean())
    return top - bot


# ---- Per-cell driver ----

def run_one_cell(n_cm12: float, D_Vnm: float, *, nk: int, verbose: bool = True,
                  solver: str = "newton"):
    t0 = time.perf_counter()
    model, h_run = build_model(D_Vnm, nk=nk)
    h_arr = np.ascontiguousarray(np.asarray(h_run.hs), dtype=np.complex128)
    w2d = np.ascontiguousarray(np.asarray(h_run.kmesh.weights), dtype=np.float64)
    layer_idx = np.asarray(model.layer)

    # CN reference (U=0, half-filling)
    model_cn, h_cn = build_model(0.0, nk=nk)
    h_cn_arr = np.ascontiguousarray(np.asarray(h_cn.hs), dtype=np.complex128)
    w_cn = np.ascontiguousarray(np.asarray(h_cn.kmesh.weights), dtype=np.float64)
    ne_cn = float(h_cn.state.compute_density_from_filling(0.5 - 1e-6))
    refP = cn_density(h_cn_arr, w_cn, ne_cn, TEMPERATURE)

    dd = n_cm12 * 1e12 * (PER_CM ** 2)
    n_target = ne_cn + dd

    HH_bare = build_bare_HH(model)

    # RPA dressing at the non-interacting mu of the BIASED h (h_run)
    eps, V = _eigh_all(h_arr)
    mu_bare = find_mu_for_density(eps, w2d, n_target, TEMPERATURE)
    Pi = compressibility_at_mu(eps, V, w2d, mu_bare, TEMPERATURE)
    HH_dressed = rpa_dress(HH_bare, Pi)

    spec_HH_dressed = float(np.linalg.norm(HH_dressed, 2))
    spec_HH_bare = float(np.linalg.norm(HH_bare, 2))
    spec_Pi = float(np.linalg.norm(Pi, 2))

    # Hartree-only kernel.  Exchange channel-packed (1,1) form keeps Vq small.
    coulomb_q_dummy = np.zeros((h_arr.shape[0], h_arr.shape[1], 1, 1),
                               dtype=np.float64)
    kernel = cpp_hf.HartreeFockKernel(
        weights=w2d,
        hamiltonian=h_arr,
        coulomb_q=coulomb_q_dummy,
        T=TEMPERATURE,
        include_hartree=True,
        include_exchange=False,
        reference_density=refP,
        hartree_matrix=HH_dressed,
    )
    if solver == "newton":
        cfg = HartreeNewtonConfig(
            max_iter=40, tol_E=1e-7, tol_sigma=1e-7,
            backtrack_max=6, backtrack_shrink=0.5,
        )
        res = solve_hartree_newton(kernel, refP, n_target, config=cfg)
        n_iter, converged, fock_out = (
            res.iterations, res.converged, np.asarray(res.fock_matrix),
        )
    elif solver == "scf":
        cfg = SCFConfig(
            max_iter=120, density_tol=1e-7, comm_tol=1e-6,
            mixing=0.3, level_shift=0.0,
            acceleration="diis", diis_size=8, diis_start=2,
            diis_damping=0.7, trust_radius=0.05,
        )
        res = solve_scf(kernel, refP, n_target, config=cfg)
        n_iter, converged, fock_out = (
            res.iterations, res.converged, np.asarray(res.fock_matrix),
        )
    elif solver == "dm":
        cfg = SolverConfig(
            max_iter=200, tol_E=1e-7,
            hartree_precondition=True, hartree_pc_scale=1.0,
            occupation_precondition=True,
        )
        res = solve_direct_minimization(kernel, refP, n_target, config=cfg)
        n_iter, converged, fock_out = (
            int(res.n_iter), bool(res.converged), np.asarray(res.fock),
        )
    else:
        raise ValueError(f"unknown solver: {solver}")

    delta = delta_tb_from_fock(fock_out, layer_idx)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"  [{solver:6s}] n={n_cm12:5.2f} D={D_Vnm:5.2f}  it={n_iter:3d} "
              f"conv={int(converged)}  Δ_tb={delta:+8.3f} meV  "
              f"‖HH_bare‖={spec_HH_bare:.2e}  ‖Π‖={spec_Pi:.2e}  "
              f"‖HH_dressed‖={spec_HH_dressed:.2e}  t={elapsed:5.1f}s")
    return delta, res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nk", type=int, default=NK_DEFAULT)
    p.add_argument("--cell", type=int, default=None,
                   help="Run only the i-th validation cell (0..4).")
    p.add_argument("--solver", choices=("newton", "scf", "dm"),
                   default="newton",
                   help="Which solver to use for the Hartree subproblem.")
    args = p.parse_args(argv)

    print(f"Tetralayer ABC graphene Hartree-only validation, "
          f"nk={args.nk}, T={TEMPERATURE} meV, "
          f"ε_⊥={EPSILON_PERP}, ε_zz={EPSILON_ZZ}, gate={GATE_DISTANCE_NM} nm")
    print()

    cells = VALIDATION_CELLS if args.cell is None else [VALIDATION_CELLS[args.cell]]
    fails = []
    for n, D, ref, tol in cells:
        delta, _ = run_one_cell(n, D, nk=args.nk, solver=args.solver)
        ok = abs(delta - ref) < tol
        status = "PASS" if ok else "FAIL"
        print(f"      ref={ref:+7.2f} ± {tol:.1f}  |Δ-ref|={abs(delta-ref):6.3f}  {status}")
        if not ok:
            fails.append((n, D, ref, delta, tol))

    print()
    if not fails:
        print("All validation cells passed.")
        return 0
    print(f"{len(fails)} cell(s) failed:")
    for n, D, ref, got, tol in fails:
        print(f"  (n={n}, D={D})  ref={ref:+.2f} ± {tol}  got={got:+.2f}  err={got-ref:+.3f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
