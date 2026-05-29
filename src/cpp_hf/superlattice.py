"""Streaming superlattice Fock self-energy.

For Hamiltonians in a plane-wave basis indexed by a moire/superlattice
reciprocal-vector basis ``g_basis`` (TBG, modulated bilayer, twisted MoTe_2,
generic SuperlatticeModel), the Fock self-energy

    Σ_F[α G_a, β G_b](k) = -Σ_{q, G_0} V(|k - q + G_0|)
                           · ρ[α (G_a+G_0), β (G_b+G_0)](q)

is a single 2D convolution in absolute momentum p = k - G, decoupled into
independent (α, β, ΔG = G_a - G_b) channels.  This module precomputes the
extended-grid scatter/gather layout and offers a thin wrapper around the
native ``_native.selfenergy_superlattice_streamed`` primitive, which streams
channels one at a time through a shared scratch buffer so peak memory is
O(N_ext^2 · dim_orb^2), independent of n_delta.

The native loop owns the per-channel scatter, FFT, multiply, iFFT, and
gather entirely in C++, so SCF inner loops pay no Python↔C transition cost
per ΔG channel.

For full SCF runs against a superlattice problem, use
:class:`SuperlatticeHartreeFockKernel` (this module) with the existing
:func:`cpp_hf.solve_scf` / :func:`cpp_hf.solve` Python wrappers — the
underlying ``_native.solve_scf`` / ``_native.solve_dm`` dispatch on the
``superlattice_*`` flags carried in the kernel-args dict and use the
streaming Fock natively inside ``build_fock_compact``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ._compat import _native, native_required


@dataclass(frozen=True)
class ExtendedGridLayout:
    """Indexing data for the extended-momentum-grid superlattice Fock.

    The extended grid is a uniform fractional moire-reciprocal-lattice mesh
    that covers all ``p = k - G`` for ``k`` in the BZ mesh and ``G`` in the
    discrete ``g_basis``, zero-padded to 2× the (k, G) support so the FFT-
    based convolution is aperiodic in lag.

    Attributes
    ----------
    N_ext_x, N_ext_y : int
        Extended-grid dimensions (each ``= 2 · nk · G_dim``).
    n_supp_x, n_supp_y : int
        Support dimensions: ``nk · G_dim``.
    g_a_off : (n_G, 2) int64 array
        Per-G tile offset (in units of ``nkx``, ``nky``) that places the
        absolute-momentum tile ``k - G`` in the extended grid.
    n_delta : int
        Number of unique ΔG = G_a − G_b values across g_basis × g_basis.
    pair_to_delta : (n_G, n_G) int64 array
        For each (G_a, G_b) index pair, the index into the unique-ΔG list.
    V_lag_fft : (N_ext_x, N_ext_y) complex128 array
        FFT of the Cartesian Coulomb kernel tabulated on the FFT-natural lag
        grid of the extended cell, scaled by the BZ-integration weight scalar.
    delta_pair_count : (n_delta,) int64 array
        Number of (G_a, G_b) pairs that map to each ΔG-index.
    delta_pair_i, delta_pair_j : (n_pairs,) int64 arrays
        Concatenated (G_a, G_b) pair indices grouped by ΔG.  Entries
        ``[delta_pair_start[d] : delta_pair_start[d+1]]`` list all
        ``(i, j)`` pairs with ΔG-index ``d``.
    delta_pair_start : (n_delta + 1,) int64 array
        CSR-style row pointer into ``delta_pair_i`` / ``delta_pair_j``.
    """

    N_ext_x: int
    N_ext_y: int
    n_supp_x: int
    n_supp_y: int
    g_a_off: np.ndarray
    n_delta: int
    pair_to_delta: np.ndarray
    V_lag_fft: np.ndarray
    delta_pair_count: np.ndarray
    delta_pair_i: np.ndarray
    delta_pair_j: np.ndarray
    delta_pair_start: np.ndarray


def build_extended_layout(
    g_basis_fractional: np.ndarray,
    g_basis_B: np.ndarray,
    nkx: int,
    nky: int,
    coulomb_V: Callable[[np.ndarray], np.ndarray],
    w_scalar: float,
) -> ExtendedGridLayout:
    """Precompute index arrays + FFT'd Coulomb kernel for the extended grid.

    Parameters
    ----------
    g_basis_fractional : (n_G, 2) int array
        Discrete G-basis as integer multiples of the moire reciprocal-lattice
        vectors (matching the SuperlatticeModel convention).
    g_basis_B : (2, 2) float array
        Cartesian reciprocal-lattice basis: ``G_cart = B @ G_frac``.
    nkx, nky : int
        BZ mesh size.
    coulomb_V : callable
        ``q_cart_2d -> V_meV`` taking 2D arrays of momentum magnitudes and
        returning the Coulomb potential at those magnitudes.
    w_scalar : float
        Uniform BZ-integration weight scalar (kMeshgridBZ convention).
    """
    g_frac = np.asarray(g_basis_fractional, dtype=np.int64)
    n_G = int(g_frac.shape[0])
    # SuperlatticeModel.T(k) attaches absolute plane-wave momentum p = k - G_a
    # to G_a, so the extended-grid tile for G_a sits at offset -G_a.
    p_g_frac = -g_frac
    G_x_min = int(p_g_frac[:, 0].min())
    G_x_max = int(p_g_frac[:, 0].max())
    G_y_min = int(p_g_frac[:, 1].min())
    G_y_max = int(p_g_frac[:, 1].max())
    G_dim_x = G_x_max - G_x_min + 1
    G_dim_y = G_y_max - G_y_min + 1
    n_supp_x = nkx * G_dim_x
    n_supp_y = nky * G_dim_y
    N_ext_x = 2 * n_supp_x
    N_ext_y = 2 * n_supp_y

    g_a_off = np.empty((n_G, 2), dtype=np.int64)
    g_a_off[:, 0] = p_g_frac[:, 0] - G_x_min
    g_a_off[:, 1] = p_g_frac[:, 1] - G_y_min

    # Build the (unique-ΔG list, pair → ΔG-index lookup).
    delta_set: dict[tuple[int, int], int] = {}
    pair_to_delta = np.empty((n_G, n_G), dtype=np.int64)
    for i in range(n_G):
        for j in range(n_G):
            dg = (
                int(g_frac[i, 0] - g_frac[j, 0]),
                int(g_frac[i, 1] - g_frac[j, 1]),
            )
            if dg not in delta_set:
                delta_set[dg] = len(delta_set)
            pair_to_delta[i, j] = delta_set[dg]
    n_delta = len(delta_set)

    delta_pair_count = np.zeros(n_delta, dtype=np.int64)
    for i in range(n_G):
        for j in range(n_G):
            delta_pair_count[int(pair_to_delta[i, j])] += 1

    delta_pair_start = np.zeros(n_delta + 1, dtype=np.int64)
    delta_pair_start[1:] = np.cumsum(delta_pair_count)
    n_pairs_total = int(delta_pair_start[-1])
    delta_pair_i = np.empty(n_pairs_total, dtype=np.int64)
    delta_pair_j = np.empty(n_pairs_total, dtype=np.int64)
    cursor = delta_pair_start.copy()
    for i in range(n_G):
        for j in range(n_G):
            d = int(pair_to_delta[i, j])
            pos = int(cursor[d])
            delta_pair_i[pos] = i
            delta_pair_j[pos] = j
            cursor[d] = pos + 1

    # Cartesian lag grid for the extended cell, scaled by w_scalar so the
    # convolution carries the BZ-integration weight implicitly.
    ix = np.arange(N_ext_x)
    iy = np.arange(N_ext_y)
    frac_lag_x = np.where(ix < N_ext_x // 2, ix, ix - N_ext_x).astype(float) / float(nkx)
    frac_lag_y = np.where(iy < N_ext_y // 2, iy, iy - N_ext_y).astype(float) / float(nky)
    fx, fy = np.meshgrid(frac_lag_x, frac_lag_y, indexing="ij")
    B = np.asarray(g_basis_B, dtype=float)
    lag_cart_x = B[0, 0] * fx + B[0, 1] * fy
    lag_cart_y = B[1, 0] * fx + B[1, 1] * fy
    qmag = np.sqrt(lag_cart_x ** 2 + lag_cart_y ** 2)
    V_lag = np.asarray(coulomb_V(qmag), dtype=float)
    V_lag_fft = np.fft.fftn(float(w_scalar) * V_lag, axes=(0, 1))

    return ExtendedGridLayout(
        N_ext_x=N_ext_x,
        N_ext_y=N_ext_y,
        n_supp_x=n_supp_x,
        n_supp_y=n_supp_y,
        g_a_off=g_a_off,
        n_delta=n_delta,
        pair_to_delta=pair_to_delta,
        V_lag_fft=np.ascontiguousarray(V_lag_fft, dtype=np.complex128),
        delta_pair_count=delta_pair_count,
        delta_pair_i=delta_pair_i,
        delta_pair_j=delta_pair_j,
        delta_pair_start=delta_pair_start,
    )


def superlattice_fock(
    rho: np.ndarray,
    layout: ExtendedGridLayout,
    n_G: int,
    dim_orb: int,
    nkx: int,
    nky: int,
    *,
    V_lag_fft_orbital: np.ndarray | None = None,
) -> np.ndarray:
    """Streaming superlattice Fock via the native ΔG-loop primitive.

    Parameters
    ----------
    rho : array
        Density tensor with shape ``(nkx, nky, n_G * dim_orb, n_G * dim_orb)``.
        Reshaped internally to the 6-D layout consumed by the native routine.
    layout : ExtendedGridLayout
        Precomputed layout from :func:`build_extended_layout`.
    n_G, dim_orb, nkx, nky : int
        Dimensions matching the layout.
    V_lag_fft_orbital : (N_ext_x, N_ext_y, dim_orb, dim_orb) complex, optional
        Orbital-resolved lag-Coulomb FFT.  When supplied, the streaming
        primitive multiplies per orbital pair (α, β) at each q (i.e.,
        ``V_{αβ}(q)`` instead of a scalar ``V(q)``).  Required for
        problems where the Coulomb interaction is orbital-resolved
        (e.g. layer-resolved gate screening in multilayer graphene).

    Returns
    -------
    sigma : ndarray
        Same shape as ``rho``.  ``Σ_F[A, B](k)`` in the kα⊗jβ block layout.
    """
    native_required()
    D = n_G * dim_orb
    rho_arr = np.ascontiguousarray(np.asarray(rho), dtype=np.complex128)
    if rho_arr.shape != (nkx, nky, D, D):
        raise ValueError(
            f"rho must have shape ({nkx}, {nky}, {D}, {D}); got {rho_arr.shape}"
        )
    rho_6d = rho_arr.reshape(nkx, nky, n_G, dim_orb, n_G, dim_orb)
    V_orb_arr = None
    if V_lag_fft_orbital is not None:
        V_orb_arr = np.ascontiguousarray(
            np.asarray(V_lag_fft_orbital), dtype=np.complex128,
        )
        expected = (layout.N_ext_x, layout.N_ext_y, dim_orb, dim_orb)
        if V_orb_arr.shape != expected:
            raise ValueError(
                f"V_lag_fft_orbital must have shape {expected}; got "
                f"{V_orb_arr.shape}"
            )
    sigma_6d = _native.selfenergy_superlattice_streamed(
        rho=rho_6d,
        VR_fft=layout.V_lag_fft,
        g_a_off=layout.g_a_off,
        pair_i=layout.delta_pair_i,
        pair_j=layout.delta_pair_j,
        pair_start=layout.delta_pair_start,
        nkx=int(nkx), nky=int(nky),
        n_G=int(n_G), dim_orb=int(dim_orb),
        N_ext_x=int(layout.N_ext_x), N_ext_y=int(layout.N_ext_y),
        VR_fft_orbital=V_orb_arr,
    )
    return np.asarray(sigma_6d).reshape(nkx, nky, D, D)


class SuperlatticeHartreeFockKernel:
    """Kernel for a moire / superlattice HF problem (k, G plane-wave basis).

    Holds the precomputed extended-grid layout, the Hartree coupling matrix
    on the G-basis, and the bare Hamiltonian.  Configured to be consumed
    by the existing :func:`cpp_hf.solve_scf` / :func:`cpp_hf.solve` Python
    wrappers — the underlying native solver detects the ``superlattice_*``
    flags and dispatches to the streaming Fock + full Hartree paths inside
    ``build_fock_compact``.

    Parameters
    ----------
    weights : (nkx, nky) array
        BZ integration weights (uniform — kMeshgridBZ convention).
    hamiltonian : (nkx, nky, D, D) array
        Bare Hamiltonian on the moire mesh.  ``D == n_G · dim_orb``.
    layout : ExtendedGridLayout
        Precomputed from :func:`build_extended_layout`.
    HH_GG : (n_G, n_G) real array
        Hartree coupling matrix on the G-basis: ``HH_GG[i, j] = V(|G_i - G_j|)``
        with diagonal zeroed out.  ``None`` falls back to building it from
        ``coulomb_V`` evaluated on ``g_cartesian`` differences.
    coulomb_V : callable, optional
        ``q_cart_2d -> V_meV`` taking 2D arrays of momentum magnitudes.
        Only needed if ``HH_GG`` is ``None``.
    g_cart : (n_G, 2) float array, optional
        Cartesian G coordinates.  Only needed if ``HH_GG`` is ``None``.
    T : float
        Electronic temperature (meV).
    include_hartree, include_exchange : bool
    reference_density : optional (nkx, nky, D, D)
    hartree_degeneracy : float
        Spin × valley factor multiplying the Hartree contribution.
    embed_reference_potential : bool
        Bake ``V_HF[refP]`` into h once at construction so the solver runs
        standard HF (gradient is ``h + V_HF[P]``, not ``h + V_HF[P - refP]``).
    center_embedded_hartree : bool
        If embedding a Hartree reference, subtract its scalar diagonal mean
        as a gauge choice (matches contimod convention).
    """

    def __init__(
        self,
        weights: np.ndarray,
        hamiltonian: np.ndarray,
        layout: ExtendedGridLayout,
        *,
        HH_GG: np.ndarray | None = None,
        HH_GG_orbital: np.ndarray | None = None,
        V_lag_fft_orbital: np.ndarray | None = None,
        coulomb_V: Callable[[np.ndarray], np.ndarray] | None = None,
        g_cart: np.ndarray | None = None,
        T: float,
        include_hartree: bool = True,
        include_exchange: bool = True,
        reference_density: np.ndarray | None = None,
        hartree_degeneracy: float = 1.0,
        embed_reference_potential: bool = False,
        center_embedded_hartree: bool = True,
    ) -> None:
        h = np.ascontiguousarray(np.asarray(hamiltonian), dtype=np.complex128)
        if h.ndim != 4 or h.shape[-1] != h.shape[-2]:
            raise ValueError(f"hamiltonian must be (nkx, nky, D, D); got {h.shape}")
        nkx, nky, D, _ = h.shape
        # Derive n_G / dim_orb from layout consistency.
        n_G = int(layout.g_a_off.shape[0])
        if D % n_G != 0:
            raise ValueError(
                f"Hamiltonian dim D={D} not divisible by n_G={n_G} from layout"
            )
        dim_orb = D // n_G

        w2d = np.ascontiguousarray(np.asarray(weights), dtype=np.float64)
        if w2d.shape != (nkx, nky):
            raise ValueError(f"weights must have shape ({nkx}, {nky}); got {w2d.shape}")

        self.T = float(T)
        self.include_hartree = bool(include_hartree)
        self.include_exchange = bool(include_exchange)
        if not (self.include_hartree or self.include_exchange):
            raise ValueError(
                "SuperlatticeHartreeFockKernel must include at least one channel."
            )

        # Hartree coupling on the G-basis: derive from the Coulomb callable
        # if not provided.
        if HH_GG is None:
            if not self.include_hartree:
                HH_GG_arr = np.zeros((n_G, n_G), dtype=np.float64)
            else:
                if coulomb_V is None or g_cart is None:
                    raise ValueError(
                        "include_hartree=True without HH_GG requires "
                        "coulomb_V + g_cart"
                    )
                g_c = np.asarray(g_cart, dtype=float)
                g_diffs = np.linalg.norm(g_c[:, None, :] - g_c[None, :, :], axis=-1)
                HH_GG_arr = np.asarray(coulomb_V(g_diffs), dtype=np.float64)
                np.fill_diagonal(HH_GG_arr, 0.0)
        else:
            HH_GG_arr = np.asarray(HH_GG, dtype=np.float64)
            if HH_GG_arr.shape != (n_G, n_G):
                raise ValueError(
                    f"HH_GG must have shape ({n_G}, {n_G}); got {HH_GG_arr.shape}"
                )
        self._HH_GG = np.ascontiguousarray(HH_GG_arr, dtype=np.float64)

        # Optional orbital-resolved Hartree coupling.  When provided, the
        # native Hartree kernel switches to the per-orbital formula:
        #   σ_H[A, α, B, β] = δ_{αβ} · degeneracy
        #                     · Σ_γ HH_GG_orb[A, B, α, γ] · ρ_γ(ΔG_{AB}).
        # The diagonal-G blocks must be zero on input (q=0 piece dropped).
        if HH_GG_orbital is None:
            self._HH_GG_orbital = None
        else:
            HH_orb = np.asarray(HH_GG_orbital, dtype=np.float64)
            expected = (n_G, n_G, dim_orb, dim_orb)
            if HH_orb.shape != expected:
                raise ValueError(
                    f"HH_GG_orbital must have shape {expected}; got "
                    f"{HH_orb.shape}"
                )
            diag_blocks = HH_orb[np.arange(n_G), np.arange(n_G)]
            if float(np.max(np.abs(diag_blocks))) > 1e-12:
                raise ValueError(
                    "HH_GG_orbital must have zero diagonal-G blocks (q=0 "
                    "piece dropped); got nonzero HH_GG_orbital[i, i, :, :]."
                )
            self._HH_GG_orbital = np.ascontiguousarray(HH_orb, dtype=np.float64)

        # Optional orbital-resolved Fock kernel — pre-FFT'd lag Coulomb on
        # the extended grid, shape (N_ext_x, N_ext_y, dim_orb, dim_orb).
        # When supplied, the streaming Fock primitive uses
        # ``V_{αβ}(q)`` per orbital pair instead of the scalar V from
        # ``layout.V_lag_fft``.
        if V_lag_fft_orbital is None:
            self._V_lag_fft_orbital = None
        else:
            V_orb = np.asarray(V_lag_fft_orbital, dtype=np.complex128)
            expected = (layout.N_ext_x, layout.N_ext_y,
                        dim_orb, dim_orb)
            if V_orb.shape != expected:
                raise ValueError(
                    f"V_lag_fft_orbital must have shape {expected}; got "
                    f"{V_orb.shape}"
                )
            self._V_lag_fft_orbital = np.ascontiguousarray(V_orb)

        # Reference density (normal ordering).
        if reference_density is not None:
            ref = np.ascontiguousarray(np.asarray(reference_density), dtype=h.dtype)
            if ref.shape != h.shape:
                raise ValueError(
                    f"reference_density must have shape {h.shape}; got {ref.shape}"
                )
            self._refP = ref
        else:
            self._refP = None
        self._empty_refP = np.empty((0,), dtype=h.dtype)
        self._zero_refP = np.broadcast_to(np.array(0, dtype=h.dtype), h.shape)

        # Dummy fields that satisfy _kernel_args_for_native's existing
        # contract.  The native kernel ignores ``VR`` and ``HH`` when the
        # superlattice_* flags are set, but we still need shape-correct
        # arrays so make_kernel's validators pass.
        self.h = h
        self.weights_b = w2d[..., None, None]
        self.w2d = w2d
        self.weight_sum = float(np.sum(w2d))
        self._VR_shifted = np.zeros((nkx, nky, 1, 1), dtype=h.dtype)
        self.exchange_hermitian_channel_packing = False
        self.HH = np.zeros((D, D), dtype=np.float64)
        self.contact_g = np.zeros((1,), dtype=np.float64)
        self.contact_Oi = np.zeros((1, D, D), dtype=h.dtype)
        self.contact_Oj = np.zeros((1, D, D), dtype=h.dtype)

        # Superlattice-specific config exposed via as_args() and the
        # _kernel_args_for_native extension in cpp_hf.{reference_scf,solver}.
        self._layout = layout
        self.n_G = int(n_G)
        self.dim_orb = int(dim_orb)
        self.hartree_degeneracy = float(hartree_degeneracy)
        self.superlattice_fock_active = bool(self.include_exchange)
        self.superlattice_hartree_active = bool(self.include_hartree)

        # Embedded reference potential.  Computes V_HF[refP] = Σ_F[refP] + Σ_H[refP]
        # once and bakes it into h, so the solver runs standard HF where the
        # iteration acts on (P - refP) and the gradient is h + V_HF[P].
        self.embed_reference_potential = bool(embed_reference_potential)
        self.center_embedded_hartree = bool(center_embedded_hartree)
        self.embedded_energy_offset = 0.0
        if self.embed_reference_potential:
            if self._refP is None:
                raise ValueError(
                    "embed_reference_potential=True requires reference_density."
                )
            offset = 0.0
            V_ref = np.zeros_like(h, dtype=h.dtype)
            if self.include_exchange:
                sigma_F_ref = superlattice_fock(
                    self._refP, layout, n_G, dim_orb, nkx, nky,
                    V_lag_fft_orbital=self._V_lag_fft_orbital,
                )
                V_ref = V_ref + sigma_F_ref
                offset += float(
                    np.einsum("ij,ijab,ijba->", w2d, sigma_F_ref, self._refP).real
                )
            if self.include_hartree:
                sigma_H_ref_full = _hartree_full_from_layout(
                    self._refP, w2d, layout, n_G, dim_orb,
                    HH_GG=self._HH_GG, degeneracy=self.hartree_degeneracy,
                    HH_GG_orbital=self._HH_GG_orbital,
                )
                if self.center_embedded_hartree:
                    mean_diag = float(np.mean(np.real(np.diag(sigma_H_ref_full))))
                    sigma_H_ref_full = sigma_H_ref_full - mean_diag * np.eye(D, dtype=h.dtype)
                V_ref = V_ref + sigma_H_ref_full[None, None, :, :]
                offset += float(
                    np.einsum("ij,ab,ijba->", w2d, sigma_H_ref_full, self._refP).real
                )
            self.h = np.ascontiguousarray(self.h + V_ref, dtype=np.complex128)
            self.embedded_energy_offset = 0.5 * offset

    @property
    def has_reference_density(self) -> bool:
        return self._refP is not None

    @property
    def refP(self) -> np.ndarray:
        if self._refP is None:
            return self._zero_refP
        return self._refP

    @property
    def layout(self) -> ExtendedGridLayout:
        return self._layout

    @property
    def HH_GG(self) -> np.ndarray:
        return self._HH_GG

    @property
    def HH_GG_orbital(self) -> np.ndarray | None:
        return self._HH_GG_orbital

    @property
    def V_lag_fft_orbital(self) -> np.ndarray | None:
        return self._V_lag_fft_orbital


def _hartree_full_from_layout(
    rho: np.ndarray,
    weights: np.ndarray,
    layout: ExtendedGridLayout,
    n_G: int,
    dim_orb: int,
    *,
    HH_GG: np.ndarray,
    degeneracy: float,
    HH_GG_orbital: np.ndarray | None = None,
) -> np.ndarray:
    """Build a (D, D) k-independent Hartree shift matrix from a density.

    Mirrors the C++ ``build_fock_compact`` superlattice-Hartree branch.  Used
    only for ``embed_reference_potential=True`` (one-time precompute).
    When ``HH_GG_orbital`` is supplied, evaluates the orbital-resolved
    Hartree formula instead of the scalar one.
    """
    D = n_G * dim_orb
    rho_bar = np.einsum("ij,ijab->ab", weights, np.asarray(rho)).reshape(
        n_G, dim_orb, n_G, dim_orb
    )
    if HH_GG_orbital is None:
        rho_GG = np.einsum("AxBx->AB", rho_bar)
        rho_for_delta = np.zeros(layout.n_delta, dtype=complex)
        np.add.at(rho_for_delta, layout.pair_to_delta.ravel(), rho_GG.ravel())
        sigma_GG = degeneracy * HH_GG * rho_for_delta[layout.pair_to_delta]
        sigma_H = np.zeros((n_G, dim_orb, n_G, dim_orb), dtype=complex)
        for xi in range(dim_orb):
            sigma_H[:, xi, :, xi] = sigma_GG
        return sigma_H.reshape(D, D)
    rho_perorb = np.einsum("AxBx->ABx", rho_bar)  # (n_G, n_G, dim_orb)
    rho_for_delta_perorb = np.zeros((layout.n_delta, dim_orb), dtype=complex)
    np.add.at(
        rho_for_delta_perorb,
        layout.pair_to_delta.ravel(),
        rho_perorb.reshape(n_G * n_G, dim_orb),
    )
    rho_at_pair = rho_for_delta_perorb[layout.pair_to_delta]
    HH_orb = np.asarray(HH_GG_orbital).astype(complex)
    sigma_perorb = degeneracy * np.einsum("ABag,ABg->ABa", HH_orb, rho_at_pair)
    sigma_H = np.zeros((n_G, dim_orb, n_G, dim_orb), dtype=complex)
    for alpha in range(dim_orb):
        sigma_H[:, alpha, :, alpha] = sigma_perorb[:, :, alpha]
    return sigma_H.reshape(D, D)


__all__ = [
    "ExtendedGridLayout",
    "build_extended_layout",
    "superlattice_fock",
    "SuperlatticeHartreeFockKernel",
]
