"""Shared setup for bilayer graphene density scan examples.

Provides:
  * canonical physical parameters (nk=49, T=0.5, etc.)
  * bilayer problem + seeds + symmetry projections
  * density-point iterator
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_CACHE_ROOT = Path(tempfile.gettempdir()) / "cpp_hf_optional_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for optional_src in (
    REPO_ROOT.parent / "contimod" / "src",
    REPO_ROOT.parent / "contimod_graphene" / "src",
):
    if optional_src.exists() and str(optional_src) not in sys.path:
        sys.path.insert(0, str(optional_src))

# -------- Physical parameters --------
NK = 49
KMAX = 0.14
U_MEV = 40.0
TEMPERATURE = 0.5
EPSILON_R = 10.0
D_GATE = 40.0
INIT_SCALE = 45.0
PER_CM = 0.246e-7

# Parameters for layer-resolved Coulomb (with Hartree).  hBN-like dielectric.
EPSILON_ZZ = 3.4 * 4      # out-of-plane dielectric
EPSILON_PERP = 6.9 * 4    # in-plane dielectric
LAT_NM = 0.246            # graphene lattice constant (nm)
LAYER_SPACING_NM = 0.335  # bilayer interlayer distance (nm)

# Default scan points in units of 1e12 cm^-2.
DEFAULT_DENSITY_POINTS = tuple(round(-0.60 + 0.02 * i, 2) for i in range(31))
BRANCHES = ("PM", "SVP")


class BilayerSetup(NamedTuple):
    h_template: Any
    ne_cn: float
    weights: np.ndarray
    Vq: np.ndarray
    seeds: dict[str, np.ndarray]
    project_fns: dict[str, Any]
    hartree_matrix: np.ndarray | None = None  # (nb, nb), only for layer-resolved
    reference_density: np.ndarray | None = None  # (nk, nk, nb, nb), CN ref
    model: Any = None


def _flip_k(
    A: np.ndarray,
    k_convention: str,
    flip_axes: tuple[int, ...],
) -> np.ndarray:
    if k_convention == "flip":
        return np.flip(A, axis=flip_axes)
    if k_convention == "mod":
        nk1, nk2 = A.shape[0], A.shape[1]
        i = (
            (-np.arange(nk1, dtype=np.int32)) % nk1
            if 0 in flip_axes
            else np.arange(nk1, dtype=np.int32)
        )
        j = (
            (-np.arange(nk2, dtype=np.int32)) % nk2
            if 1 in flip_axes
            else np.arange(nk2, dtype=np.int32)
        )
        return A[i[:, None], j[None, :], ...]
    raise ValueError(f"k_convention must be 'mod' or 'flip', got {k_convention!r}")


def _make_time_reversal_U(s2: np.ndarray, v1: np.ndarray) -> np.ndarray:
    return np.asarray(v1) @ (1j * np.asarray(s2))


def _make_pm_group(
    identity: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
    spin_elems = [identity, s1, s2, s3]
    valley_elems = [identity, v3]
    return np.stack([S @ V for S in spin_elems for V in valley_elems], axis=0)


def _make_svp_project_fn(
    *,
    s3: np.ndarray,
    v3: np.ndarray,
    n_orb: int,
    outlier_sv: tuple[int, int] = (+1, +1),
    k_convention: str = "flip",
    k_flip_axes: tuple[int, ...] = (0,),
):
    s3_np = np.asarray(s3)
    v3_np = np.asarray(v3)
    nb = s3_np.shape[0]
    n_blocks = nb // n_orb
    so, vo = float(outlier_sv[0]), float(outlier_sv[1])

    idx_outlier = None
    idx_same_v = None
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
            "Could not identify 4 spin-valley blocks from s3/v3 "
            f"(n_orb={n_orb}, nb={nb}, outlier_sv={outlier_sv})"
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

    def project(P: np.ndarray) -> np.ndarray:
        P_arr = np.asarray(P)
        out = np.array(P_arr * mask, copy=True)

        P_same = P_arr[..., sl_same, sl_same]
        P_ov0_flip = _flip_k(P_arr[..., sl_ov0, sl_ov0], k_conv, k_axes)
        P_ov1_flip = _flip_k(P_arr[..., sl_ov1, sl_ov1], k_conv, k_axes)

        Q = (P_same + P_ov0_flip + P_ov1_flip) / 3.0
        Q_flip = _flip_k(Q, k_conv, k_axes)

        out[..., sl_same, sl_same] = Q
        out[..., sl_ov0, sl_ov0] = Q_flip
        out[..., sl_ov1, sl_ov1] = Q_flip
        return out

    return project


def build_bilayer(
    *,
    nk: int = NK,
    kmax: float = KMAX,
    U_meV: float = U_MEV,
    temperature: float = TEMPERATURE,
    epsilon_r: float = EPSILON_R,
    d_gate: float = D_GATE,
    init_scale: float = INIT_SCALE,
) -> BilayerSetup:
    """Build the bilayer graphene problem via contimod."""
    import contimod as cm
    from contimod.utils.spectrum_fermi import FermiParams

    from cpp_hf.symmetry import make_project_fn

    model = cm.graphene.MultilayerAB(valleyful=True, spinful=True, U=float(U_meV))
    h_template = model.discretize(nk=int(nk), kmax=float(kmax))
    h_template.fermi = FermiParams(T=float(temperature), mu=0.0)

    ne_cn = float(h_template.state.compute_density_from_filling(0.5 - 1e-6))
    weights = np.asarray(h_template.kmesh.weights)

    Vq = cm.coulomb.dualgate_coulomb(
        h_template.kmesh.distance_grid,
        epsilon_r=float(epsilon_r),
        d_gate=float(d_gate),
    )
    Vq = np.asarray(Vq.magnitude)[..., None, None]

    identity_op = np.asarray(model.identity)
    s1, s2, s3 = [np.asarray(model.spin_op(i)) for i in (1, 2, 3)]
    v1, v3 = np.asarray(model.valley_op(1)), np.asarray(model.valley_op(3))
    U_tr = _make_time_reversal_U(s2, v1)

    projector_sv = 0.25 * (identity_op + s3) @ (identity_op + v3)
    sv_contrast = -projector_sv + 3.0 * (identity_op - projector_sv)

    # Spin-only contrast operator (valley-symmetric, breaks spin): -s3 makes
    # spin-up lower energy.  Negating sign flips which spin is favored.
    spin_contrast = -s3

    seeds = {
        "PM": h_template.get_operator("zero"),
        "SVP": -float(init_scale) * h_template.get_operator(sv_contrast),
        # SP: spin-polarized but valley-symmetric
        "SP": -float(init_scale) * h_template.get_operator(spin_contrast),
        # SVP_flipped: same SVP projection, opposite seed sign (tests whether
        # the SVP projected subspace is connected - should converge to the
        # same state as SVP if so, else to a distinct configuration).
        "SVP_flipped": +float(init_scale) * h_template.get_operator(sv_contrast),
    }

    G_pm = _make_pm_group(identity_op, s1, s2, s3, v3)
    # SP projection: enforce valley-z invariance (avg over [I, v3]) but allow
    # arbitrary spin structure.  Don't apply time reversal (TR flips spin,
    # which would un-polarize the state).
    G_sp = np.stack([np.asarray(identity_op), np.asarray(v3)], axis=0)
    project_fns = {
        "PM": make_project_fn(
            unitary_group=G_pm,
            time_reversal_U=np.asarray(U_tr),
            time_reversal_k_convention="flip",
        ),
        "SVP": _make_svp_project_fn(
            s3=np.asarray(s3),
            v3=np.asarray(v3),
            n_orb=4,
            outlier_sv=(+1, +1),
            k_convention="flip",
            k_flip_axes=(0,),
        ),
        "SP": make_project_fn(unitary_group=G_sp),
        "SVP_flipped": _make_svp_project_fn(
            s3=np.asarray(s3),
            v3=np.asarray(v3),
            n_orb=4,
            outlier_sv=(+1, +1),
            k_convention="flip",
            k_flip_axes=(0,),
        ),
    }

    return BilayerSetup(
        h_template=h_template,
        ne_cn=ne_cn,
        weights=weights,
        Vq=Vq,
        seeds=seeds,
        project_fns=project_fns,
        model=model,
    )


def build_bilayer_layer_resolved(
    *,
    nk: int = NK,
    kmax: float = KMAX,
    U_meV: float = U_MEV,
    temperature: float = TEMPERATURE,
    epsilon_zz: float = EPSILON_ZZ,
    epsilon_perp: float = EPSILON_PERP,
    lat_nm: float = LAT_NM,
    layer_spacing_nm: float = LAYER_SPACING_NM,
    init_scale: float = INIT_SCALE,
) -> BilayerSetup:
    """Like ``build_bilayer`` but with a layer-resolved Coulomb kernel.

    Uses ``contimod.meanfield.coulomb.layer_coulomb_kernel`` with per-orbital
    z-coordinates derived from ``model.layer``, giving a (nk, nk, nb, nb)
    kernel that supports both layer-resolved exchange AND Hartree.

    Also extracts ``hartree_matrix`` from ``Vq`` at q=0.  The caller must
    supply (or bootstrap) a ``reference_density`` for the Hartree subtraction;
    this builder leaves ``reference_density=None`` in the returned tuple.
    """
    import contimod as cm
    from contimod.meanfield.coulomb import layer_coulomb_kernel
    from contimod.utils.spectrum_fermi import FermiParams

    from cpp_hf.symmetry import make_project_fn

    model = cm.graphene.MultilayerAB(valleyful=True, spinful=True, U=float(U_meV))
    h_template = model.discretize(nk=int(nk), kmax=float(kmax))
    h_template.fermi = FermiParams(T=float(temperature), mu=0.0)

    ne_cn = float(h_template.state.compute_density_from_filling(0.5 - 1e-6))
    weights = np.asarray(h_template.kmesh.weights)

    # Layer-resolved Coulomb: z_coords length = nb (per-orbital layer coord)
    z_coords = np.asarray(model.layer) * (layer_spacing_nm / lat_nm)
    Vq = layer_coulomb_kernel(
        h_template.kmesh.distance_grid, z_coords,
        epsilon_zz=float(epsilon_zz), epsilon_perp=float(epsilon_perp),
        a_nm=float(lat_nm),
    )
    Vq = np.asarray(Vq, dtype=np.float64)  # shape (nk, nk, nb, nb)

    # Extract hartree_matrix from Vq at q=0
    distances = np.asarray(h_template.kmesh.distance_grid)
    k0 = np.unravel_index(np.argmin(distances), distances.shape)
    hartree_matrix = np.asarray(Vq[k0], dtype=np.float64)

    # Same seeds and projections as build_bilayer
    identity_op = np.asarray(model.identity)
    s1, s2, s3 = [np.asarray(model.spin_op(i)) for i in (1, 2, 3)]
    v1, v3 = np.asarray(model.valley_op(1)), np.asarray(model.valley_op(3))
    U_tr = _make_time_reversal_U(s2, v1)

    projector_sv = 0.25 * (identity_op + s3) @ (identity_op + v3)
    sv_contrast = -projector_sv + 3.0 * (identity_op - projector_sv)
    spin_contrast = -s3

    seeds = {
        "PM": h_template.get_operator("zero"),
        "SVP": -float(init_scale) * h_template.get_operator(sv_contrast),
        "SP": -float(init_scale) * h_template.get_operator(spin_contrast),
        "SVP_flipped": +float(init_scale) * h_template.get_operator(sv_contrast),
    }

    G_pm = _make_pm_group(identity_op, s1, s2, s3, v3)
    G_sp = np.stack([np.asarray(identity_op), np.asarray(v3)], axis=0)
    project_fns = {
        "PM": make_project_fn(
            unitary_group=G_pm,
            time_reversal_U=np.asarray(U_tr),
            time_reversal_k_convention="flip",
        ),
        "SVP": _make_svp_project_fn(
            s3=np.asarray(s3), v3=np.asarray(v3), n_orb=4,
            outlier_sv=(+1, +1), k_convention="flip", k_flip_axes=(0,),
        ),
        "SP": make_project_fn(unitary_group=G_sp),
        "SVP_flipped": _make_svp_project_fn(
            s3=np.asarray(s3), v3=np.asarray(v3), n_orb=4,
            outlier_sv=(+1, +1), k_convention="flip", k_flip_axes=(0,),
        ),
    }

    return BilayerSetup(
        h_template=h_template,
        ne_cn=ne_cn,
        weights=weights,
        Vq=Vq,
        seeds=seeds,
        project_fns=project_fns,
        hartree_matrix=hartree_matrix,
        reference_density=None,  # caller must bootstrap
        model=model,
    )


def bootstrap_cn_reference_density(
    setup: BilayerSetup,
    *,
    temperature: float = TEMPERATURE,
    max_iter: int = 300,
    mixing: float = 0.3,
    density_tol: float = 1e-7,
    comm_tol: float = 1e-6,
) -> np.ndarray:
    """Compute a self-consistent CN density with Hartree enabled, PM projection.

    Bootstraps by using the non-interacting CN density as both the initial
    guess and the ``reference_density``.  Returns the converged CN density
    matrix, which can then be used as the ``reference_density`` for the full
    scan.
    """
    import cpp_hf
    from cpp_hf import SCFConfig, solve_scf

    if setup.hartree_matrix is None:
        raise ValueError("setup must include hartree_matrix (use build_bilayer_layer_resolved).")

    cn_nonint, _ = setup.h_template.state.compute_densitymatrix_for_density(setup.ne_cn)
    cn_nonint = np.asarray(cn_nonint)

    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights,
        hamiltonian=np.asarray(setup.h_template.hs),
        coulomb_q=setup.Vq,
        T=float(temperature),
        include_hartree=True,
        include_exchange=True,
        reference_density=cn_nonint,
        hartree_matrix=setup.hartree_matrix,
    )
    config = SCFConfig(
        max_iter=int(max_iter),
        mixing=float(mixing),
        density_tol=float(density_tol),
        comm_tol=float(comm_tol),
        project_fn=setup.project_fns["PM"],
    )
    result = solve_scf(
        kernel, np.asarray(cn_nonint), float(setup.ne_cn), config=config,
    )
    return np.asarray(result.density_matrix)


def n_electrons_for_density(setup: BilayerSetup, n_cm12: float, temperature: float) -> tuple[float, Any]:
    """Build an h_run at target density; return (n_electrons, h_run)."""
    from contimod.utils.spectrum_fermi import FermiParams

    dd = n_cm12 * 1e12 * (PER_CM ** 2)
    h_run = setup.h_template.copy()
    h_run.fermi = FermiParams(T=float(temperature), mu=0.0)
    h_run.compute_chemicalpotential(density=float(setup.ne_cn + dd))
    n_e = float(h_run.state.compute_density() / float(h_run.degeneracy))
    return n_e, h_run


def initial_density_from_seed(h_run, seed_op, temperature: float) -> np.ndarray:
    """Cold-seed density matrix for a given branch operator."""
    from contimod.meanfield.init_guess import init_to_density_matrix

    P0 = init_to_density_matrix(h_run, seed_op, density=None, T=float(temperature), init_kind="auto")
    return np.asarray(P0)


def write_figure(
    *,
    csv_path: Path,
    rows: list[dict],
    title: str,
    figure_path: Path | None = None,
    branches: list[str] | None = None,
):
    """Plot energy-per-particle vs density for the listed branches.

    If ``branches`` is None, defaults to BRANCHES = ("PM", "SVP").
    """
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not available; skipping figure")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if figure_path is None:
        figure_path = csv_path.with_name(csv_path.stem + "_energy_per_particle.png")

    if branches is None:
        branches = list(BRANCHES)

    colors = {
        "PM": "#1f4e79", "SVP": "#b3472e",
        "SP": "#2e7d32", "SVP_flipped": "#7b1fa2",
    }
    markers = {
        "PM": "o", "SVP": "s",
        "SP": "^", "SVP_flipped": "D",
    }
    styles = {
        "PM": "-", "SVP": "--",
        "SP": "-.", "SVP_flipped": ":",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)

    # Find neutral-point energy per branch (for subtracting)
    neutral_E = {}
    for row in rows:
        if abs(float(row["density_cm12"])) < 1e-8:
            neutral_E[row["branch"]] = float(row["energy"])

    for branch in branches:
        branch_rows = [r for r in rows if r["branch"] == branch]
        if not branch_rows:
            continue
        branch_rows.sort(key=lambda r: float(r["density_cm12"]))
        x = np.array([float(r["density_cm12"]) for r in branch_rows])
        E = np.array([float(r["energy"]) for r in branch_rows])
        # Energy per carrier relative to neutrality
        E_ref = neutral_E.get(branch, float("nan"))
        n_abs = np.abs(x) * 1e12 * (PER_CM ** 2)
        n_abs_safe = np.where(n_abs < 1e-12, np.nan, n_abs)
        y = (E - E_ref) / n_abs_safe
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        ax.plot(
            x[finite], y[finite],
            color=colors.get(branch, "#444"),
            marker=markers.get(branch, "o"),
            linestyle=styles.get(branch, "-"),
            linewidth=2.0, markersize=5.0,
            label=branch,
        )

    ax.axvline(0.0, color="#999", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Carrier density $n$ ($10^{12}\,\mathrm{cm}^{-2}$)")
    ax.set_ylabel("Energy per carrier (meV), relative to CN")
    ax.set_title(title)
    ax.grid(True, color="#e0e0e0", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"wrote figure: {figure_path}")
