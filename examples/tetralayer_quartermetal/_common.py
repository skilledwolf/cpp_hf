"""Shared setup for tetralayer rhombohedral graphene quartermetal study.

Mirrors the structure of ``_bilayer_common.py`` but for the four-layer ABC
problem in the quartermetal task spec:

  - ``cm.graphene.NlayerABC(N=4, valleyful=True, spinful=True, U=U_mev)``
  - layer-resolved Coulomb for exchange plus q=0 Hartree overrides
  - PM, SVP, PM_C3, and SVP_C3 projectors on the rhombic k cell
  - SVP-biased seed for symmetry-broken initialization

Sign convention: spec puts top at +U/2, bottom at -U/2.  contimod's
``NlayerABC(U=...)`` uses the OPPOSITE convention; we flip the sign when
constructing the model so that bare Δ_tb = +U_spec for D > 0.

To keep the validation harness reusable, this module exposes both gated
(``hartree_matrix_dualgate``) and ungated (``hartree_matrix_ungated``)
forms of the Hartree q=0 kernel.  ``dualgate_eps*`` presets use the same
isotropic dielectric for finite-q exchange and q=0 Hartree so Fock-only and
Hartree+Fock scans use a consistent interaction scale.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
for opt in (REPO_ROOT.parent / "contimod" / "src",
            REPO_ROOT.parent / "contimod_graphene" / "src"):
    if opt.exists() and str(opt) not in sys.path:
        sys.path.insert(0, str(opt))

_CACHE_ROOT = Path(tempfile.gettempdir()) / "cpp_hf_optional_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))


# ---- Physical constants from the spec ----
N_LAYERS = 4
NK_DEFAULT = 48
KMAX = 0.30
TEMPERATURE = 1.0          # meV
EPSILON_ZZ = 2.0
EPSILON_PERP = 13.8
GATE_DISTANCE_NM = 30.0
LAT_NM = 0.246
LAYER_SPACING_NM = 0.34
PER_CM = 0.246e-7

# Spin-valley sector layout: NlayerABC(valleyful, spinful) lays out
# (spin × valley) blocks of 8 layer-sublattice orbitals, with spin slowest
# and valley next.  N_ORB_PER_SV = 2 sublattice × 4 layer = 8.
N_ORB_PER_SV = 8


def U_meV_from_D(D_Vnm: float, *, N: int = N_LAYERS,
                 layer_spacing_nm: float = LAYER_SPACING_NM,
                 epsilon_zz: float = EPSILON_ZZ) -> float:
    """Spec layer-bias drop in meV: V_top - V_bot = +U for D > 0."""
    return D_Vnm * (N - 1) * layer_spacing_nm / epsilon_zz * 1000.0


# ---- Layer-resolved Coulomb (ungated, matches bilayer pattern) ----

def _z_aG(model) -> np.ndarray:
    """Per-orbital z coordinates in units of graphene a (= LAT_NM nm)."""
    z_norm = np.asarray(model.layer)
    z_phys_nm = z_norm * (N_LAYERS - 1) * LAYER_SPACING_NM / 2.0
    return z_phys_nm / LAT_NM


def build_layer_Vq(model, h_template, *,
                   epsilon_perp: float = EPSILON_PERP,
                   epsilon_zz: float = EPSILON_ZZ,
                   lat_nm: float = LAT_NM) -> np.ndarray:
    """Layer-resolved 2D Coulomb on the (nk1, nk2, nb, nb) grid (ungated)."""
    from contimod.meanfield.coulomb import layer_coulomb_kernel
    distances = np.asarray(h_template.kmesh.distance_grid)
    Vq = layer_coulomb_kernel(
        distances, _z_aG(model),
        epsilon_zz=float(epsilon_zz), epsilon_perp=float(epsilon_perp),
        a_nm=float(lat_nm),
    )
    return np.ascontiguousarray(np.asarray(Vq), dtype=np.float64)


def hartree_matrix_ungated(model, *,
                           epsilon_perp: float = EPSILON_PERP,
                           epsilon_zz: float = EPSILON_ZZ,
                           lat_nm: float = LAT_NM) -> np.ndarray:
    """q=0 layer-Coulomb HH from ``layer_coulomb_kernel`` (ungated)."""
    from contimod.meanfield.coulomb import layer_coulomb_kernel
    HH = layer_coulomb_kernel(
        np.array([[0.0]]), _z_aG(model),
        epsilon_zz=float(epsilon_zz), epsilon_perp=float(epsilon_perp),
        a_nm=float(lat_nm),
    )[0, 0]
    return np.ascontiguousarray(np.asarray(HH), dtype=np.float64)


def hartree_matrix_dualgate(model, *,
                            gate_nm: float = GATE_DISTANCE_NM,
                            epsilon_perp: float = EPSILON_PERP,
                            epsilon_zz: float = EPSILON_ZZ,
                            lat_nm: float = LAT_NM) -> np.ndarray:
    """q=0 dual-gate HH from ``dualgate_hartree_q0_matrix`` (literal spec)."""
    from contimod.meanfield.coulomb import dualgate_hartree_q0_matrix
    HH = dualgate_hartree_q0_matrix(
        _z_aG(model), d_gate=gate_nm / lat_nm,
        epsilon_r=epsilon_perp, epsilon_r_perp=epsilon_zz, a=lat_nm,
    )
    return np.ascontiguousarray(np.asarray(HH), dtype=np.float64)


def hartree_matrix_poisson(model, *,
                           t_top_nm: float = GATE_DISTANCE_NM,
                           t_bot_nm: float = GATE_DISTANCE_NM,
                           eps_top: float = 3.4,
                           eps_bot: float = 3.4,
                           eps_interlayer: float = 2.0,   # spec ε_il=2.0
                           d_interlayer_nm: float = LAYER_SPACING_NM,
                           lat_nm: float = LAT_NM,
                           project_out_uniform: bool = False) -> np.ndarray:
    """Hartree matrix from a 1D Poisson tridiagonal with metallic gate boundaries.

    Builds the layer-resolved capacitance network (gate caps + interlayer
    geometric capacitance) and inverts the Poisson tridiagonal:

        ℓ=1:    (C_int + C_b) φ_1 - C_int φ_2                 = σ_1
        1<ℓ<N:  -C_int φ_{ℓ-1} + 2 C_int φ_ℓ - C_int φ_{ℓ+1}    = σ_ℓ
        ℓ=N:    -C_int φ_{N-1} + (C_int + C_t) φ_N               = σ_N

    Returns HH in the cpp_hf unit convention (meV per (1/a_G²)) so that
    V_H[orbital] = HH @ σ where σ has the per-orbital weighted-density
    units used by contimod.

    ``project_out_uniform`` removes the all-ones component of HH.  Use
    only if you have separately verified that the uniform-mode response
    is irrelevant for your physics — it can be substantial.
    """
    EPS_0 = 8.8541878128e-12   # F/m
    E_CHARGE = 1.602176634e-19  # C
    N = int(N_LAYERS)
    layer_array = np.asarray(model.layer, dtype=float)
    # Map per-orbital layer ∈ {-1, -1/3, 1/3, 1} → integer index ∈ {0..N-1}
    layer_idx = np.round((layer_array + 1.0) * 0.5 * (N - 1)).astype(int)

    t_top_m = float(t_top_nm) * 1e-9
    t_bot_m = float(t_bot_nm) * 1e-9
    d_int_m = float(d_interlayer_nm) * 1e-9

    C_t = EPS_0 * float(eps_top) / t_top_m         # F/m²
    C_b = EPS_0 * float(eps_bot) / t_bot_m
    C_int = EPS_0 * float(eps_interlayer) / d_int_m

    diag = np.full(N, 2.0 * C_int)
    diag[0] = C_int + C_b
    diag[-1] = C_int + C_t
    off = np.full(N - 1, -C_int)
    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    A_inv = np.linalg.inv(A)                      # V·m²/C

    a_G_m_sq = (float(lat_nm) * 1e-9) ** 2
    HH_layer = (1000.0 * E_CHARGE / a_G_m_sq) * A_inv   # meV / (1/a_G²)

    HH_orb = HH_layer[layer_idx[:, None], layer_idx[None, :]]
    HH_orb = np.ascontiguousarray(HH_orb, dtype=np.float64)

    if project_out_uniform == "symmetric":
        nb = HH_orb.shape[0]
        u = np.ones(nb) / np.sqrt(nb)
        HH_orb = (HH_orb
                   - np.outer(u, u @ HH_orb)
                   - np.outer(HH_orb @ u, u)
                   + (u @ HH_orb @ u) * np.outer(u, u))
    elif project_out_uniform in ("right", True):
        # One-sided right-projection: kills HH·u (uniform-σ → 0 layer pot).
        # Preserves the structure of HH otherwise (no fictitious sign flips
        # in off-diagonals).  Enough to suppress numerical-drift amplification
        # of the dominant uniform-mode eigenvalue when Σσ ≈ 0.
        nb = HH_orb.shape[0]
        u = np.ones(nb) / np.sqrt(nb)
        HH_orb = HH_orb - np.outer(HH_orb @ u, u)
    return HH_orb


# ---- Tetralayer model setup ----

class QMSetup(NamedTuple):
    h_template: Any
    model: Any
    weights: np.ndarray              # (nk, nk)
    Vq: np.ndarray                   # (nk, nk, nb, nb) layer Coulomb
    hartree_matrix: np.ndarray       # (nb, nb)
    seeds: dict[str, np.ndarray]     # one-body operators for cold seeding
    project_fns: dict[str, Callable] # PM, SVP projectors
    ne_cn: float                     # CN electron count


def build_setup(*, D_Vnm: float, nk: int = NK_DEFAULT, kmax: float = KMAX,
                T: float = TEMPERATURE, init_scale: float = 30.0,
                hartree_kind: str = "ungated",
                small_orbital: bool = False,
                bz_kind: str = "rhombic") -> QMSetup:
    """Build the tetralayer ABC mean-field setup at displacement field D.

    ``small_orbital=True`` builds with ``valleyful=spinful=False``, giving
    an 8-orbital model (sublattice × layer) with explicit degeneracy 4.
    cpp_hf then runs on 8×8 matrices instead of 32×32 — ~16× faster per
    iter.  Valid for PM-projected configs (no spin-valley structure
    needed at the SCF level).  For SVP, keep ``small_orbital=False``.

    ``bz_kind="rhombic"`` uses a 60-degree rhombic Brillouin-zone patch with
    reciprocal-basis columns ``b1=(s/2,-s√3/2)``, ``b2=(s/2,+s√3/2)``,
    ``s = 2·kmax``.  Continuum symmetries are handled as non-periodic
    in-patch maps on the actual Cartesian k coordinates rather than as
    reciprocal-cell wraps.  ``"cartesian"`` is retained only for diagnostics.
    """
    import contimod as cm
    from contimod.utils.spectrum_fermi import FermiParams

    U_spec = U_meV_from_D(D_Vnm)
    # Flip sign for contimod's opposite convention.
    if small_orbital:
        model = cm.graphene.NlayerABC(N=N_LAYERS, valleyful=False,
                                       spinful=False, U=-U_spec)
    else:
        model = cm.graphene.NlayerABC(N=N_LAYERS, valleyful=True,
                                       spinful=True, U=-U_spec)

    if bz_kind == "rhombic":
        s = 2.0 * float(kmax)
        h_tri = s * np.sqrt(3.0) / 2.0
        Bmat = np.array([[s / 2.0, s / 2.0], [-h_tri, h_tri]])
        bz = cm.kdiscrete.BrillouinZone(B=Bmat)
        h = model.discretize(nk=int(nk), bz=bz)
    elif bz_kind == "cartesian":
        h = model.discretize(nk=int(nk), kmax=float(kmax))
    else:
        raise ValueError(f"unknown bz_kind: {bz_kind}")
    h.fermi = FermiParams(T=float(T), mu=0.0)

    ne_cn = float(h.state.compute_density_from_filling(0.5 - 1e-6))
    weights = np.ascontiguousarray(np.asarray(h.kmesh.weights),
                                    dtype=np.float64)
    vq_epsilon_perp = EPSILON_PERP
    vq_epsilon_zz = EPSILON_ZZ
    if hartree_kind == "dualgate_geomean":
        eps_eff = float(np.sqrt(EPSILON_PERP * EPSILON_ZZ))
        vq_epsilon_perp = eps_eff
        vq_epsilon_zz = eps_eff
    elif hartree_kind in {
        "dualgate_eps2", "dualgate_eps3", "dualgate_eps4",
        "dualgate_eps5", "dualgate_eps10", "dualgate_eps20",
    }:
        vq_epsilon_perp = float(hartree_kind.removeprefix("dualgate_eps"))
        vq_epsilon_zz = vq_epsilon_perp
    elif hartree_kind == "ungated_eps3":
        vq_epsilon_perp = 3.0
        vq_epsilon_zz = 3.0
    Vq = build_layer_Vq(
        model, h, epsilon_perp=vq_epsilon_perp, epsilon_zz=vq_epsilon_zz,
    )
    if hartree_kind == "ungated":
        HH = hartree_matrix_ungated(model)
    elif hartree_kind == "dualgate":
        HH = hartree_matrix_dualgate(model)
    elif hartree_kind == "dualgate_geomean":
        # Use geometric-mean ε in the prefactor to match the anisotropic
        # Coulomb's true scaling (contimod's prefactor uses only ε_in,
        # underweighting interlayer Coulomb by ~ε_in/√(ε_in·ε_out)).
        eps_eff = float(np.sqrt(EPSILON_PERP * EPSILON_ZZ))
        HH = hartree_matrix_dualgate(model, epsilon_perp=eps_eff,
                                       epsilon_zz=eps_eff)
    elif hartree_kind == "dualgate_eps2":
        # Use ε_zz alone (=2.0) for both prefactor and q-anisotropy: maximally
        # strong Hartree response, no in-plane screening.  Diagnostic only.
        HH = hartree_matrix_dualgate(model, epsilon_perp=EPSILON_ZZ,
                                       epsilon_zz=EPSILON_ZZ)
    elif hartree_kind == "dualgate_eps4":
        # Intermediate between geomean (5.25) and eps2: 4.0
        HH = hartree_matrix_dualgate(model, epsilon_perp=4.0, epsilon_zz=4.0)
    elif hartree_kind == "dualgate_eps3":
        HH = hartree_matrix_dualgate(model, epsilon_perp=3.0, epsilon_zz=3.0)
    elif hartree_kind == "dualgate_eps5":
        HH = hartree_matrix_dualgate(model, epsilon_perp=5.0, epsilon_zz=5.0)
    elif hartree_kind == "dualgate_eps10":
        HH = hartree_matrix_dualgate(model, epsilon_perp=10.0, epsilon_zz=10.0)
    elif hartree_kind == "dualgate_eps20":
        HH = hartree_matrix_dualgate(model, epsilon_perp=20.0, epsilon_zz=20.0)
    elif hartree_kind == "ungated_eps3":
        HH = hartree_matrix_ungated(model, epsilon_perp=3.0, epsilon_zz=3.0)
    elif hartree_kind == "poisson":
        HH = hartree_matrix_poisson(model)
    elif hartree_kind == "poisson_proj":
        HH = hartree_matrix_poisson(model, project_out_uniform="right")
    elif hartree_kind == "poisson_proj_sym":
        HH = hartree_matrix_poisson(model, project_out_uniform="symmetric")
    else:
        raise ValueError(f"unknown hartree_kind: {hartree_kind}")

    # Seeds and projectors (PM, SVP) — same as bilayer pattern but with
    # n_orb=8 per spin-valley block.
    if small_orbital:
        # 8-orbital model: no spin/valley structure.  PM is trivial
        # (everything is in the symmetric subspace by construction).
        # SVP and SVP_flipped are not meaningful; return identity projectors.
        seeds = {"PM": h.get_operator("zero")}
        project_fns = {"PM": (lambda A: A)}
        return QMSetup(
            h_template=h, model=model, weights=weights, Vq=Vq,
            hartree_matrix=HH, seeds=seeds, project_fns=project_fns,
            ne_cn=ne_cn,
        )

    identity_op = np.asarray(model.identity)
    s1, s2, s3 = [np.asarray(model.spin_op(i)) for i in (1, 2, 3)]
    v1, v3 = np.asarray(model.valley_op(1)), np.asarray(model.valley_op(3))
    U_tr = np.asarray(v1) @ (1j * np.asarray(s2))

    # Spec-recommended SVP seed shape: lower the outlier sector (s=+1, v=+1)
    # by ``init_scale`` meV and raise the other 3 sectors by ``init_scale/3``
    # meV, so the perturbation is trace-preserving in the orbital sector
    # weights.  Concretely:
    #     V_seed = -init_scale * proj_pol + (init_scale/3) * (I - proj_pol)
    # Each block has 8 orbitals, so the bias matrix above shifts (s=+1,v=+1)
    # diagonal entries by -init_scale and the rest by +init_scale/3.
    proj_pol = 0.25 * (identity_op + s3) @ (identity_op + v3)
    sv_seed_shape = (-1.0) * proj_pol + (1.0 / 3.0) * (identity_op - proj_pol)
    # Hole-doping picks the opposite SV sector as outlier — flipped seed.
    sv_seed_shape_flipped = (1.0) * proj_pol - (1.0 / 3.0) * (identity_op - proj_pol)

    seeds = {
        "PM": h.get_operator("zero"),
        "SVP": float(init_scale) * h.get_operator(sv_seed_shape),
        "SVP_flipped": float(init_scale) * h.get_operator(sv_seed_shape_flipped),
    }

    # PM group: spin {I, σ_x, σ_y, σ_z} × valley {I, τ_z}, plus time reversal.
    spin_elems = [identity_op, s1, s2, s3]
    valley_elems = [identity_op, v3]
    G_pm = np.stack([S @ V for S in spin_elems for V in valley_elems], axis=0)

    # SVP uses the valley-folding equivalence between K and K' blocks:
    # contimod represents K' as H_K(-kx, ky), so the opposite-valley inactive
    # sectors are compared after a kx mirror.  Use a non-periodic map on the
    # actual k coordinates: points with a represented mirror partner are
    # paired, while boundary points without a target are left untouched.
    svp_k_map = _make_nonperiodic_k_map(
        np.asarray(h.kmesh.ks),
        lambda k: np.array([-k[0], k[1]], dtype=float),
        return_valid=True,
    )
    svp_map_i, svp_map_j, svp_map_valid = svp_k_map
    svp_proj = _make_svp_project_fn(
        s3=s3, v3=v3, n_orb=N_ORB_PER_SV,
        outlier_sv=(+1, +1), k_index_map=(svp_map_i, svp_map_j),
        k_valid_mask=svp_map_valid,
        use_tr_flip=True,
    )
    pm_proj = _make_pm_project_fn_nonperiodic(
        unitary_group=G_pm,
        time_reversal_U=U_tr,
        ks=np.asarray(h.kmesh.ks),
    )
    project_fns = {
        "PM": pm_proj,
        # Same SVP projector for both seed conventions; the difference is
        # the basin reached from the cold seed.
        "SVP": svp_proj,
        "SVP_flipped": svp_proj,
    }
    # C3 projector is part of the production rhombic scan convention.
    if bz_kind == "rhombic":
        c3_proj = make_c3_project_fn_numpy(
            N_LAYERS, int(nk), ks=np.asarray(h.kmesh.ks),
        )
        project_fns["PM_C3"] = compose_project_fns(pm_proj, c3_proj)
        # The SVP_C3 branch is C3-constrained and selected by the SVP seed;
        # spin-valley imbalance is checked by the scan driver.
        project_fns["SVP_C3"] = c3_proj

    return QMSetup(
        h_template=h, model=model, weights=weights, Vq=Vq,
        hartree_matrix=HH, seeds=seeds, project_fns=project_fns,
        ne_cn=ne_cn,
    )


# ---- SVP projector (n_orb=8 for tetralayer) ----

def _make_nonperiodic_k_map(ks: np.ndarray, transform,
                            *,
                            decimals: int = 12,
                            atol: float = 1e-10,
                            return_valid: bool = False):
    """Index map for a Cartesian k-space transform on a finite patch.

    If the transformed point exists in the sampled patch, map to it.  If not,
    map the point to itself.  This enforces continuum symmetries without
    adding an artificial reciprocal-cell wrap at the patch boundary.
    """
    ks_arr = np.asarray(ks, dtype=float)
    if ks_arr.ndim != 3 or ks_arr.shape[-1] != 2:
        raise ValueError("ks must have shape (nk1, nk2, 2)")
    nk1, nk2 = ks_arr.shape[:2]
    flat = ks_arr.reshape(-1, 2)
    lookup: dict[tuple[float, float], int] = {}
    for idx, k in enumerate(flat):
        lookup[tuple(np.round(k, decimals=decimals))] = idx

    idx_map = np.arange(nk1 * nk2, dtype=np.int32).reshape(nk1, nk2)
    valid = np.zeros((nk1, nk2), dtype=bool)

    def _lookup(target: np.ndarray) -> int | None:
        idx = lookup.get(tuple(np.round(np.asarray(target, dtype=float),
                                        decimals=decimals)))
        if idx is None:
            return None
        if np.linalg.norm(flat[idx] - target) > atol:
            return None
        return int(idx)

    for i in range(nk1):
        for j in range(nk2):
            out = _lookup(transform(ks_arr[i, j]))
            if out is not None:
                idx_map[i, j] = out
                valid[i, j] = True

    map_i = idx_map // nk2
    map_j = idx_map % nk2
    if return_valid:
        return map_i, map_j, valid
    return map_i, map_j


def _avg_unitary_conj_numpy(A: np.ndarray, G: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(A)
    for g in G:
        acc = acc + (g @ A) @ np.conj(g.T)
    return acc / np.asarray(float(len(G)), dtype=A.dtype)


def _make_pm_project_fn_nonperiodic(*, unitary_group: np.ndarray,
                                    time_reversal_U: np.ndarray,
                                    ks: np.ndarray):
    """PM projector with non-periodic time reversal on the finite k patch."""
    G = np.asarray(unitary_group)
    U = np.asarray(time_reversal_U)
    UH = np.conj(U.T)
    map_i, map_j, valid = _make_nonperiodic_k_map(
        ks, lambda k: np.array([-k[0], -k[1]], dtype=float),
        return_valid=True,
    )

    def project(A: np.ndarray) -> np.ndarray:
        A_arr = np.asarray(A)
        out = _avg_unitary_conj_numpy(A_arr, G)
        A_neg = out[map_i, map_j, ...]
        A_tr = U @ np.conj(A_neg) @ UH
        projected = np.array(out, copy=True)
        projected[valid] = 0.5 * (out[valid] + A_tr[valid])
        return projected

    return project


def _make_svp_project_fn(*, s3, v3, n_orb: int,
                          outlier_sv: tuple[int, int] = (+1, +1),
                          k_convention: str = "flip",
                          k_flip_axes: tuple[int, ...] = (0,),
                          k_index_map: tuple[np.ndarray, np.ndarray] | None = None,
                          k_valid_mask: np.ndarray | None = None,
                          use_tr_flip: bool = True):
    """SVP: pick one spin-valley flavor as 'outlier', average remaining 3.

    ``k_index_map`` optionally supplies the k-space map used to compare
    opposite-valley sectors.  It should already encode any finite-patch
    boundary policy, e.g. mapping missing mirror partners to themselves.
    """
    s3_np = np.asarray(s3)
    v3_np = np.asarray(v3)
    nb = s3_np.shape[0]
    n_blocks = nb // n_orb
    so, vo = float(outlier_sv[0]), float(outlier_sv[1])

    idx_outlier = None
    idx_same_v: int | None = None
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
            f"Could not identify 4 spin-valley blocks (n_orb={n_orb}, nb={nb})"
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
    if k_index_map is not None:
        map_i = np.asarray(k_index_map[0], dtype=np.intp)
        map_j = np.asarray(k_index_map[1], dtype=np.intp)
    else:
        map_i = map_j = None
    valid_mask = (None if k_valid_mask is None
                  else np.asarray(k_valid_mask, dtype=bool))

    def _flip(A):
        if map_i is not None and map_j is not None:
            return A[map_i, map_j, ...]
        if k_conv == "flip":
            return np.flip(A, axis=k_axes)
        if k_conv == "swap_neg":
            if A.shape[0] != A.shape[1]:
                raise ValueError("swap_neg k-map requires a square k-grid")
            nk = A.shape[0]
            idx = (-np.arange(nk, dtype=np.int32)) % nk
            return A[idx[None, :], idx[:, None], ...]
        nk1, nk2 = A.shape[0], A.shape[1]
        i = ((-np.arange(nk1, dtype=np.int32)) % nk1
             if 0 in k_axes else np.arange(nk1, dtype=np.int32))
        j = ((-np.arange(nk2, dtype=np.int32)) % nk2
             if 1 in k_axes else np.arange(nk2, dtype=np.int32))
        return A[i[:, None], j[None, :], ...]

    def project(P: np.ndarray) -> np.ndarray:
        P_arr = np.asarray(P)
        if valid_mask is None:
            out = np.array(P_arr * mask, copy=True)
            active = None
        else:
            out = np.array(P_arr, copy=True)
            out_active = out[valid_mask]
            out_active *= mask
            out[valid_mask] = out_active
            active = valid_mask
        P_same = P_arr[..., sl_same, sl_same]
        if use_tr_flip:
            P_ov0_in = _flip(P_arr[..., sl_ov0, sl_ov0])
            P_ov1_in = _flip(P_arr[..., sl_ov1, sl_ov1])
        else:
            P_ov0_in = P_arr[..., sl_ov0, sl_ov0]
            P_ov1_in = P_arr[..., sl_ov1, sl_ov1]
        Q = (P_same + P_ov0_in + P_ov1_in) / 3.0
        Q_out = _flip(Q) if use_tr_flip else Q
        if active is None:
            out[..., sl_same, sl_same] = Q
            out[..., sl_ov0, sl_ov0] = Q_out
            out[..., sl_ov1, sl_ov1] = Q_out
        else:
            same_view = out[..., sl_same, sl_same]
            ov0_view = out[..., sl_ov0, sl_ov0]
            ov1_view = out[..., sl_ov1, sl_ov1]
            same_view[active] = Q[active]
            ov0_view[active] = Q_out[active]
            ov1_view[active] = Q_out[active]
        return out

    return project


# ---- Density helpers ----

def n_electrons_for_density(setup: QMSetup, n_cm12: float, T: float):
    """Compute target electron count for doping n (×10¹² cm⁻²); return
    (n_e, h_run) where h_run is the kernel sized for that filling."""
    from contimod.utils.spectrum_fermi import FermiParams
    dd = n_cm12 * 1e12 * (PER_CM ** 2)
    h_run = setup.h_template.copy()
    h_run.fermi = FermiParams(T=float(T), mu=0.0)
    h_run.compute_chemicalpotential(density=float(setup.ne_cn + dd))
    n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))
    return n_e, h_run


def initial_density_from_seed(h_run, seed_op, T: float,
                              density: float | None = None) -> np.ndarray:
    """Cold-seed density matrix: diagonalize h_run + seed_op and Fermi-fill."""
    from contimod.meanfield.init_guess import init_to_density_matrix
    if density is None:
        density = float(h_run.state.compute_density() / float(h_run.degeneracy))
    seed_arr = np.asarray(seed_op)
    if np.max(np.abs(seed_arr)) == 0.0:
        P0, _ = h_run.state.compute_densitymatrix_for_density(float(density),
                                                              T=float(T))
        return np.ascontiguousarray(np.asarray(P0), dtype=np.complex128)
    P0 = init_to_density_matrix(h_run, seed_op, density=float(density),
                                 T=float(T), init_kind="auto")
    return np.ascontiguousarray(np.asarray(P0), dtype=np.complex128)


def noninteracting_cn_reference(setup: QMSetup, T: float | None = None,
                                  ref_kind: str = "u0",
                                  cn_projector: str = "PM_C3",
                                  cn_max_iter: int = 25,
                                  cn_tol_E: float = 1e-4) -> np.ndarray:
    """Reference density for the Hartree subtraction `σ = diag(P) - diag(refP)`.

    ``ref_kind="u0"`` (spec literal): non-interacting CN at U=0, no field.
    ``ref_kind="sameU"``: non-interacting CN at the SAME U as the SCF run.
    ``ref_kind="cn_sc"``: self-consistent HF density at CN filling, SAME U.
        Bootstrap — runs PM-projected HF SCF at n=ne_cn with refP=non-int U=0,
        returns the converged density.  σ in subsequent doped runs measures
        only the carrier excess above the self-screened CN.
    """
    import contimod as cm
    from contimod.utils.spectrum_fermi import FermiParams
    h_arr = np.asarray(setup.h_template.hs)
    nb = h_arr.shape[-1]
    small = (nb == 8)
    nk = int(setup.weights.shape[0])
    if T is None:
        T = float(setup.h_template.fermi.T)

    if ref_kind == "u0":
        U_ref = 0.0
    elif ref_kind == "sameU":
        # Reuse the SCF model's U value (already with the contimod sign flip
        # baked in via build_setup, so just read .U from params).
        U_ref = float(setup.model.params.U)
    elif ref_kind == "cn_sc":
        # Bootstrap: solve HF SCF at CN with non-int U=0 reference, return
        # converged density.  cn_projector lets the caller match the
        # symmetry to the production run (e.g. "PM_C3" for C3-enforced).
        return _bootstrap_cn_sc_reference(
            setup, T=T, projector_key=cn_projector,
            max_iter=cn_max_iter, tol_E=cn_tol_E,
        )
    else:
        raise ValueError(f"unknown ref_kind: {ref_kind}")

    m = cm.graphene.NlayerABC(N=N_LAYERS,
                               valleyful=not small, spinful=not small,
                               U=U_ref)
    bz = setup.h_template.kmesh.brillouin_zone
    if bz is not None:
        h = m.discretize(nk=nk, bz=bz)
    else:
        h = m.discretize(nk=nk, kmax=KMAX)
    h.fermi = FermiParams(T=float(T), mu=0.0)
    P, _ = h.state.compute_densitymatrix_for_density(setup.ne_cn)
    return np.ascontiguousarray(np.asarray(P), dtype=np.complex128)


def bootstrap_cn_reference(setup: QMSetup, **kwargs) -> np.ndarray:
    """Deprecated: prefer ``noninteracting_cn_reference`` per the spec."""
    return noninteracting_cn_reference(setup)


def pm_c3_cn_reference_density(setup: QMSetup, *,
                               T: float | None = None,
                               max_iter: int = 25,
                               tol_E: float = 1e-4) -> np.ndarray:
    """Self-consistent PM_C3 density at charge neutrality.

    This is the reference density for quarter-metal scans: all doped HF
    kernels contract interactions with ``P - P_PM_C3(CN)``.
    """
    return noninteracting_cn_reference(
        setup, T=T, ref_kind="cn_sc", cn_projector="PM_C3",
        cn_max_iter=max_iter, cn_tol_E=tol_E,
    )


def hf_energy_for_density(kernel, P: np.ndarray) -> float:
    """Evaluate the current referenced HF functional for ``P``."""
    from cpp_hf.fock import build_fock, hf_energy
    Sigma, H, _ = build_fock(
        P, h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
        HH=kernel.HH, w2d=kernel.w2d,
        include_exchange=kernel.include_exchange,
        include_hartree=kernel.include_hartree,
        exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
        contact_g=kernel.contact_g,
        contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
    )
    return float(hf_energy(
        P, h=kernel.h, Sigma=Sigma, H=H, weights_b=kernel.weights_b,
        refP=kernel.refP,
    ))


def _bootstrap_cn_sc_reference(setup: QMSetup, *,
                                T: float | None = None,
                                projector_key: str = "PM",
                                max_iter: int = 25,
                                tol_E: float = 1e-4) -> np.ndarray:
    """Solve symmetry-projected HF SCF at CN filling with full H+exchange.

    ``projector_key``: which symmetry to enforce in the CN bootstrap.
        Should match (or be a strict subset of) the symmetry of the run
        we'll use the reference for.  Options:
        - ``"PM"`` (default): just PM (spin-valley average + TR)
        - ``"PM_C3"``: PM + C3 — required when the production run enforces C3
        - ``"SVP"``, ``"SVP_C3"``: usually NOT what you want for refP since
          CN should be symmetric, but available for diagnostic.

    Uses the U=0 non-interacting CN density as the initial reference for
    the bootstrap SCF, runs DM at the same external U as the production
    setup, returns the converged density.
    """
    import cpp_hf
    from cpp_hf import SolverConfig, solve_direct_minimization
    if T is None:
        T = float(setup.h_template.fermi.T)
    if projector_key not in setup.project_fns:
        raise ValueError(f"projector '{projector_key}' not in setup.project_fns "
                          f"(available: {list(setup.project_fns)})")

    refP_init = noninteracting_cn_reference(setup, T=T, ref_kind="u0")

    h_arr = np.asarray(setup.h_template.hs)
    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=h_arr,
        coulomb_q=setup.Vq, T=float(T),
        include_hartree=True, include_exchange=True,
        reference_density=refP_init,
        hartree_matrix=setup.hartree_matrix,
    )
    cfg = SolverConfig(
        max_iter=int(max_iter), tol_E=float(tol_E), max_step=0.6,
        block_sizes=(8, 8, 8, 8),
        project_fn=setup.project_fns[projector_key],
    )
    res = solve_direct_minimization(kernel, refP_init, float(setup.ne_cn),
                                     config=cfg)
    if not bool(res.converged):
        raise RuntimeError(
            f"CN reference projector {projector_key!r} did not converge "
            f"within {int(max_iter)} iterations"
        )
    return np.ascontiguousarray(np.asarray(res.density), dtype=np.complex128)


def delta_tb_from_fock(fock: np.ndarray, layer_idx: np.ndarray) -> float:
    """Top minus bottom layer-averaged diagonal of F (averaged over k)."""
    nk1, nk2 = fock.shape[0], fock.shape[1]
    diag = np.einsum("ijaa->a", fock).real / (nk1 * nk2)
    layers_sorted = np.sort(np.unique(layer_idx))
    bot = float(diag[layer_idx == layers_sorted[0]].mean())
    top = float(diag[layer_idx == layers_sorted[-1]].mean())
    return top - bot


# ---- C3 projector (adapted from RNG_displacementfield_screening/c3.py) ----

C3_OMEGA = np.exp(2.0j * np.pi / 3.0)


def _make_c3_orbital_unitary(n_layers: int) -> np.ndarray:
    """C3z unitary on the (spin) ⊗ (valley) ⊗ (A_1, B_1, ..., A_N, B_N) basis
    used by contimod's NlayerABC valleyful=spinful=True.

    Diagonal: ω^((ℓ-1)+s) at K, conjugate at K'.  Spin = identity.
    """
    N = int(n_layers)
    diag_K = np.array(
        [C3_OMEGA ** ((ell - 1) + s) for ell in range(1, N + 1) for s in (0, 1)],
        dtype=np.complex128,
    )
    U_K = np.diag(diag_K)
    U_Kprime = np.conj(U_K)
    U_valley = np.zeros((2 * 2 * N, 2 * 2 * N), dtype=np.complex128)
    U_valley[: 2 * N, : 2 * N] = U_K
    U_valley[2 * N :, 2 * N :] = U_Kprime
    return np.kron(np.eye(2, dtype=np.complex128), U_valley)


def _make_c3_index_perms(nk: int):
    """Index permutations of a centered fractional grid
    ``np.linspace(-0.5, 0.5, nk, endpoint=False)`` under C3 and C3².

    Assumes the mirror-oriented rhombic basis
    ``b1=(s/2,-s√3/2)``, ``b2=(s/2,+s√3/2)``.  C3 maps fractional
    coordinates as ``(x, y) -> (-y, x-y)``.  Requires even nk so the
    centered-grid half-cell offset remains an integer index shift.
    """
    if nk % 2 != 0:
        raise ValueError(f"C3 index permutation requires even nk, got {nk}.")
    nk_half = nk // 2
    ii, jj = np.meshgrid(np.arange(nk), np.arange(nk), indexing="ij")
    perm_i  = (-jj) % nk
    perm_j  = (ii - jj + nk_half) % nk
    perm_i2 = (jj - ii + nk_half) % nk
    perm_j2 = (-ii) % nk
    return perm_i, perm_j, perm_i2, perm_j2


def _make_c3_nonperiodic_maps(ks: np.ndarray,
                              *,
                              decimals: int = 12,
                              atol: float = 1e-10):
    """Index maps for Cartesian C3 rotations that stay inside the k patch.

    Returns maps for R k and R² k plus a mask selecting complete three-point
    orbits.  Points whose full C3 orbit is not represented on the finite
    continuum patch are intentionally left untouched by the C3 projector.
    """
    ks_arr = np.asarray(ks, dtype=float)
    if ks_arr.ndim != 3 or ks_arr.shape[-1] != 2:
        raise ValueError("ks must have shape (nk1, nk2, 2)")
    nk1, nk2 = ks_arr.shape[:2]
    flat = ks_arr.reshape(-1, 2)
    lookup: dict[tuple[float, float], int] = {}
    for idx, k in enumerate(flat):
        lookup[tuple(np.round(k, decimals=decimals))] = idx

    theta = 2.0 * np.pi / 3.0
    R = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta),  np.cos(theta)]],
        dtype=float,
    )
    R2 = R @ R

    self_idx = np.arange(nk1 * nk2, dtype=np.int32).reshape(nk1, nk2)
    idx_R = np.array(self_idx, copy=True)
    idx_R2 = np.array(self_idx, copy=True)
    valid_R = np.zeros((nk1, nk2), dtype=bool)
    valid_R2 = np.zeros((nk1, nk2), dtype=bool)

    def _lookup(rotated: np.ndarray) -> int | None:
        idx = lookup.get(tuple(np.round(rotated, decimals=decimals)))
        if idx is None:
            return None
        if np.linalg.norm(flat[idx] - rotated) > atol:
            return None
        return int(idx)

    for i in range(nk1):
        for j in range(nk2):
            k = ks_arr[i, j]
            out = _lookup(R @ k)
            if out is not None:
                idx_R[i, j] = out
                valid_R[i, j] = True
            out = _lookup(R2 @ k)
            if out is not None:
                idx_R2[i, j] = out
                valid_R2[i, j] = True

    complete = valid_R & valid_R2
    return (
        idx_R // nk2, idx_R % nk2,
        idx_R2 // nk2, idx_R2 % nk2,
        complete,
    )


def make_c3_project_fn_numpy(n_layers: int, nk: int, ks: np.ndarray):
    """Pure-NumPy C3 projector for cpp_hf.

    F_sym[k] = (1/3) [ F[k] + U F[C3⁻¹k] U† + U² F[C3⁻²k] U²† ]

    Diagonal U lets us replace ``U A U†`` with elementwise multiply by
    ``M[i, j] = u_i * conj(u_j)`` — saves an O(D³)→O(D²) factor.

    C3 is treated as a non-periodic continuum-patch symmetry: complete
    in-patch C3 orbits are averaged, while points whose rotated partners leave
    the finite patch are mapped to themselves with the identity.  This avoids
    folding by artificial reciprocal vectors.
    """
    if ks is None:
        raise ValueError("make_c3_project_fn_numpy requires actual k coordinates")
    U_np = _make_c3_orbital_unitary(n_layers)
    diag_U  = np.diag(U_np)
    diag_U2 = diag_U * diag_U
    M_C3  = diag_U[:,  None] * np.conj(diag_U[None, :])
    M_C32 = diag_U2[:, None] * np.conj(diag_U2[None, :])
    pi, pj, pi2, pj2, c3_complete = _make_c3_nonperiodic_maps(ks)

    def project(A):
        A = np.asarray(A)
        # F[C3⁻¹k] in our convention is the C3² permutation index.
        A_term1 = np.array(A, copy=True)
        A_term2 = np.array(A, copy=True)
        rot1 = A[pi2, pj2] * M_C3
        rot2 = A[pi,  pj]  * M_C32
        A_term1[c3_complete] = rot1[c3_complete]
        A_term2[c3_complete] = rot2[c3_complete]
        return (A + A_term1 + A_term2) / 3.0
    return project


def compose_project_fns(*fns):
    """Compose project_fns left-to-right: ``compose(f, g)(A) = g(f(A))``."""
    def project(A):
        for f in fns:
            A = f(A)
        return A
    return project


def dos_at_mu(eps: np.ndarray, weights: np.ndarray, mu: float, T: float) -> float:
    """Fermi-smeared DOS at μ: -df/dε integrated over BZ.

    Returns the total DOS (per unit cell) — caller divides by spin-valley
    degeneracy if desired.  Units: 1/meV per unit cell (BZ-integrated).
    """
    Tsafe = max(T, 1e-12)
    f = 1.0 / (1.0 + np.exp(np.clip((eps - mu) / Tsafe, -50, 50)))
    neg_dfde = f * (1.0 - f) / Tsafe
    return float(np.einsum("ij,ijb->", weights, neg_dfde))
