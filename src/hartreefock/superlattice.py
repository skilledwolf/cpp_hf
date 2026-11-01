"""Hartree-Fock for a superlattice (moiré) plane-wave basis, for either backend.

The Hamiltonian is written in a basis ``|k, G_a, α⟩`` with ``G_a`` from a
discrete reciprocal-lattice basis and ``α`` an orbital index, so
``D = n_G · dim_orb``.  The basis state carries absolute momentum
``p = k - G_a`` and the Fock self-energy

    Σ_F[(G_a α), (G_b β)](k) = -Σ_{q, G_0} V(|k - q + G_0|) w ρ[(G_a+G_0) α, (G_b+G_0) β](q)

is one aperiodic 2D convolution in ``p`` per channel ``(ΔG = G_a - G_b, α, β)``.
:func:`build_extended_layout` precomputes the zero-padded extended grid, the
FFT of ``V`` on it and the CSR grouping of ``(G_a, G_b)`` pairs by ``ΔG``; the
native and JAX backends stream the channels through one scratch buffer.  The
Hartree term is k-independent and couples ``ΔG``-components of the BZ-averaged
density.

:class:`SuperlatticeHartreeFockKernel` is solved by :func:`hartreefock.solve`
and :func:`hartreefock.solve_scf` on either backend, exactly like the dense
:class:`~hartreefock.HartreeFockKernel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .kernel import validate_temperature
from .symmetry import array_namespace

__all__ = [
    "ExtendedGridLayout",
    "SuperlatticeHartreeFockKernel",
    "build_extended_layout",
    "superlattice_hartree",
    "upsample_density",
]


@dataclass(frozen=True)
class ExtendedGridLayout:
    """Index data and interaction tables for the extended-momentum-grid Fock.

    Attributes
    ----------
    N_ext_x, N_ext_y :
        Extended grid size, twice the ``(k, G)`` support ``nk · G_span``.
    g_a_off : (n_G, 2) int64
        Tile offset (in units of the BZ mesh) of the momenta ``k - G_a``.
    n_delta :
        Number of distinct ``ΔG = G_a - G_b``.
    pair_to_delta : (n_G, n_G) int64
        ``ΔG`` index of each ``(G_a, G_b)`` pair.
    delta_pair_i, delta_pair_j, delta_pair_start :
        CSR grouping of the pairs by ``ΔG``: pairs of channel ``d`` are
        ``[delta_pair_start[d] : delta_pair_start[d + 1]]``.
    V_lag_fft : (N_ext_x, N_ext_y) complex128
        FFT of ``w · V`` on the lag grid (``w`` the uniform BZ weight).
    V_lag_fft_orbital : (N_ext_x, N_ext_y, dim_orb, dim_orb) complex128, optional
        Orbital-pair version of ``V_lag_fft``; replaces it for exchange.
    """

    N_ext_x: int
    N_ext_y: int
    g_a_off: np.ndarray
    n_delta: int
    pair_to_delta: np.ndarray
    delta_pair_i: np.ndarray
    delta_pair_j: np.ndarray
    delta_pair_start: np.ndarray
    V_lag_fft: np.ndarray
    V_lag_fft_orbital: np.ndarray | None = None


def build_extended_layout(
    g_fractional,
    reciprocal_basis,
    nkx: int,
    nky: int,
    coulomb_V: Callable[[np.ndarray], np.ndarray],
    w_scalar: float,
    *,
    coulomb_V_orbital: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ExtendedGridLayout:
    """Precompute the extended grid for G-basis ``g_fractional`` on an ``nkx × nky`` BZ mesh.

    ``reciprocal_basis`` is the 2×2 matrix ``B`` with ``G_cart = B @ G_frac``.
    ``coulomb_V(|q|)`` maps momentum magnitudes to the interaction;
    ``coulomb_V_orbital(|q|)`` optionally returns ``(*q.shape, dim_orb,
    dim_orb)`` for an orbital-resolved exchange.
    """
    g = np.asarray(g_fractional, dtype=np.int64)
    n_G = g.shape[0]
    p = -g  # basis state G_a carries momentum k - G_a
    p_min = p.min(axis=0)
    span = p.max(axis=0) - p_min + 1
    N_ext_x, N_ext_y = 2 * int(nkx * span[0]), 2 * int(nky * span[1])
    g_a_off = np.ascontiguousarray(p - p_min, dtype=np.int64)

    dg = (g[:, None, :] - g[None, :, :]).reshape(n_G * n_G, 2)
    _, pair_to_delta = np.unique(dg, axis=0, return_inverse=True)
    pair_to_delta = pair_to_delta.reshape(n_G * n_G).astype(np.int64)
    n_delta = int(pair_to_delta.max()) + 1
    order = np.argsort(pair_to_delta, kind="stable")
    delta_pair_start = np.zeros(n_delta + 1, dtype=np.int64)
    delta_pair_start[1:] = np.cumsum(np.bincount(pair_to_delta, minlength=n_delta))

    def lag(n_ext, nk):
        i = np.arange(n_ext)
        return np.where(i < n_ext // 2, i, i - n_ext) / float(nk)

    fx, fy = np.meshgrid(lag(N_ext_x, nkx), lag(N_ext_y, nky), indexing="ij")
    B = np.asarray(reciprocal_basis, dtype=float)
    qmag = np.hypot(B[0, 0] * fx + B[0, 1] * fy, B[1, 0] * fx + B[1, 1] * fy)
    V_lag_fft = np.fft.fftn(float(w_scalar) * np.asarray(coulomb_V(qmag), dtype=float))

    V_lag_fft_orbital = None
    if coulomb_V_orbital is not None:
        V_orb = np.asarray(coulomb_V_orbital(qmag))
        if V_orb.ndim != 4 or V_orb.shape[:2] != qmag.shape or V_orb.shape[2] != V_orb.shape[3]:
            raise ValueError(
                f"coulomb_V_orbital must map a {qmag.shape} grid to "
                f"{qmag.shape} + (dim_orb, dim_orb); got {V_orb.shape}"
            )
        if not np.all(np.isfinite(V_orb)):
            raise ValueError("coulomb_V_orbital must return finite values")
        V_lag_fft_orbital = np.ascontiguousarray(
            np.fft.fftn(float(w_scalar) * V_orb, axes=(0, 1)), dtype=np.complex128
        )

    return ExtendedGridLayout(
        N_ext_x=N_ext_x,
        N_ext_y=N_ext_y,
        g_a_off=g_a_off,
        n_delta=n_delta,
        pair_to_delta=pair_to_delta.reshape(n_G, n_G),
        delta_pair_i=np.ascontiguousarray(order // n_G, dtype=np.int64),
        delta_pair_j=np.ascontiguousarray(order % n_G, dtype=np.int64),
        delta_pair_start=delta_pair_start,
        V_lag_fft=np.ascontiguousarray(V_lag_fft, dtype=np.complex128),
        V_lag_fft_orbital=V_lag_fft_orbital,
    )


def superlattice_hartree(
    rho_bar,
    pair_to_delta: np.ndarray,
    *,
    HH_GG: np.ndarray,
    HH_GG_orbital: np.ndarray | None = None,
    degeneracy: float = 1.0,
):
    """k-independent Hartree potential ``(D, D)`` of the BZ-averaged density ``rho_bar``.

    Scalar coupling: ``σ_H[(A α), (B α)] = g · HH_GG[A, B] · Σ_γ ρ_γ(ΔG_AB)``.
    Orbital coupling: ``σ_H[(A α), (B α)] = g · Σ_γ HH_GG_orbital[A, B, α, γ] ρ_γ(ΔG_AB)``,
    with ``ρ_γ(d)`` the sum of ``rho_bar[(A' γ), (B' γ)]`` over pairs with
    ``ΔG = d``.  Works on NumPy and JAX arrays (also under ``jax.jit``).
    """
    xp = array_namespace(rho_bar)
    n_G = pair_to_delta.shape[0]
    D = rho_bar.shape[-1]
    dim_orb = D // n_G
    n_delta = int(pair_to_delta.max()) + 1
    real = xp.zeros((), dtype=rho_bar.dtype).real.dtype
    # Pair -> ΔG one-hot matrix: sums pairs into ΔG components and back.
    onehot = np.zeros((n_delta, n_G * n_G))
    onehot[pair_to_delta.ravel(), np.arange(n_G * n_G)] = 1.0
    onehot = xp.asarray(onehot).astype(real)
    eye = xp.eye(dim_orb, dtype=rho_bar.dtype)

    rho_orb = xp.einsum("AxBx->ABx", rho_bar.reshape(n_G, dim_orb, n_G, dim_orb))
    rho_orb = rho_orb.reshape(n_G * n_G, dim_orb)
    rho_at_pair = (onehot.T @ (onehot @ rho_orb)).reshape(n_G, n_G, dim_orb)
    if HH_GG_orbital is None:
        HH = xp.asarray(HH_GG).astype(rho_bar.dtype)
        sigma_GG = degeneracy * HH * xp.sum(rho_at_pair, axis=-1)
        sigma = xp.einsum("AB,xy->AxBy", sigma_GG, eye)
    else:
        HH = xp.asarray(HH_GG_orbital).astype(rho_bar.dtype)
        sigma_orb = degeneracy * xp.einsum("ABag,ABg->ABa", HH, rho_at_pair)
        sigma = xp.einsum("ABa,ab->AaBb", sigma_orb, eye)
    return sigma.reshape(D, D)


def _herm(A):
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


class SuperlatticeHartreeFockKernel:
    """Precomputed superlattice Hartree-Fock problem.

    Parameters
    ----------
    weights : (nkx, nky) array
        Uniform BZ quadrature weights (the FFT convolution folds the weight
        into the interaction table).
    hamiltonian : (nkx, nky, D, D) array
        Single-particle Hamiltonian, ``D = n_G · dim_orb`` with the G index
        slow.
    coulomb_V : callable
        ``|q| -> V(|q|)``, tabulated on the extended lag grid.
    g_fractional : (n_G, 2) int array
        G-basis in units of the reciprocal lattice vectors.
    reciprocal_basis : (2, 2) array
        ``B`` with ``G_cart = B @ G_frac``.
    T : float
        Temperature.
    include_hartree, include_exchange : bool
    reference_density : (nkx, nky, D, D) array, optional
        Subtracted from ``P`` before building the self-energy.
    hartree_matrix : (n_G, n_G) array, optional
        Scalar Hartree coupling, used as given.  Default: ``coulomb_V`` at
        the G differences with the diagonal (q = 0) dropped.
    hartree_matrix_orbital : (n_G, n_G, dim_orb, dim_orb) array, optional
        Orbital Hartree coupling ``V_αγ(G_A - G_B)``, used as given (a finite
        diagonal-G block keeps the q = 0 layer Hartree of gated systems).
        Default with ``coulomb_V_orbital``: ``coulomb_V_orbital`` at the G
        differences, q = 0 kept.
    coulomb_V_orbital : callable, optional
        ``|q| -> V_αβ(|q|)`` of shape ``(*q.shape, dim_orb, dim_orb)``; used
        for exchange (and, by default, Hartree) instead of ``coulomb_V``.
    hartree_degeneracy : float
        Flavour multiplicity of the Hartree term (spin × valley folded into
        the basis).
    embed_reference_potential : bool
        Add ``V_HF[refP]`` to ``h`` once so the loop solves standard HF.  The
        reported energy then exceeds the standard HF energy by
        ``embedded_energy_offset - embedded_hartree_shift * n_electrons``;
        ``mu_standard = mu + embedded_hartree_shift``.
    center_embedded_hartree : bool
        With embedding, subtract the mean diagonal of ``σ_H[refP]``
        (``embedded_hartree_shift``; a gauge choice, zero when q = 0 is dropped).
    k_origin : (float, float)
        Fractional coordinates of the first BZ mesh point; locates the mesh
        for :meth:`potential_at` and for continuation between meshes.
    """

    def __init__(
        self,
        weights,
        hamiltonian,
        coulomb_V: Callable[[np.ndarray], np.ndarray],
        g_fractional,
        reciprocal_basis,
        *,
        T: float,
        include_hartree: bool = False,
        include_exchange: bool = True,
        reference_density=None,
        hartree_matrix=None,
        hartree_matrix_orbital=None,
        coulomb_V_orbital: Callable[[np.ndarray], np.ndarray] | None = None,
        hartree_degeneracy: float = 1.0,
        embed_reference_potential: bool = False,
        center_embedded_hartree: bool = True,
        k_origin: tuple[float, float] = (-0.5, -0.5),
    ):
        if not (include_hartree or include_exchange):
            raise ValueError("SuperlatticeHartreeFockKernel must include Hartree or exchange.")
        if not callable(coulomb_V):
            raise ValueError("coulomb_V must be callable (|q| -> V)")
        if coulomb_V_orbital is not None and not callable(coulomb_V_orbital):
            raise ValueError("coulomb_V_orbital must be callable or None")
        self.T = validate_temperature(T)
        self.include_hartree = bool(include_hartree)
        self.include_exchange = bool(include_exchange)
        self.coulomb_V = coulomb_V
        self.coulomb_V_orbital = coulomb_V_orbital

        h = np.ascontiguousarray(hamiltonian, dtype=np.complex128)
        if h.ndim != 4 or h.shape[-1] != h.shape[-2]:
            raise ValueError(f"hamiltonian must be (nkx, nky, D, D); got {h.shape}")
        nkx, nky, D, _ = h.shape
        self.g_fractional = np.asarray(g_fractional, dtype=np.int64)
        self.reciprocal_basis = np.asarray(reciprocal_basis, dtype=float)
        self.g_cart = self.g_fractional @ self.reciprocal_basis.T
        n_G = self.g_fractional.shape[0]
        if D % n_G:
            raise ValueError(f"hamiltonian dimension D={D} is not a multiple of n_G={n_G}")
        self.nkx, self.nky, self.D, self.n_G, self.dim_orb = nkx, nky, D, n_G, D // n_G

        w2d = np.ascontiguousarray(weights, dtype=np.float64)
        if w2d.shape != (nkx, nky):
            raise ValueError(f"weights must have shape ({nkx}, {nky}); got {w2d.shape}")
        w_scalar = float(w2d.flat[0])
        if w_scalar <= 0 or not np.allclose(w2d, w_scalar):
            raise ValueError("superlattice kernels need uniform positive BZ weights")
        self.w2d = w2d
        self.weights_b = w2d[..., None, None]
        self.weight_sum = float(w2d.sum())
        self.k_origin = (float(k_origin[0]), float(k_origin[1]))

        self.has_reference_density = reference_density is not None
        if self.has_reference_density:
            ref = np.ascontiguousarray(reference_density, dtype=np.complex128)
            if ref.shape != h.shape:
                raise ValueError(f"reference_density must have shape {h.shape}; got {ref.shape}")
            self.refP = ref
        else:
            self.refP = np.broadcast_to(np.zeros((), dtype=np.complex128), h.shape)

        self.layout = build_extended_layout(
            self.g_fractional, self.reciprocal_basis, nkx, nky, coulomb_V, w_scalar,
            coulomb_V_orbital=coulomb_V_orbital,
        )
        orb = self.layout.V_lag_fft_orbital
        if orb is not None and orb.shape[-2:] != (self.dim_orb, self.dim_orb):
            raise ValueError(
                f"coulomb_V_orbital returns {orb.shape[-2:]} blocks but dim_orb={self.dim_orb}"
            )

        self.hartree_degeneracy = float(hartree_degeneracy)
        self.HH_GG, self.HH_GG_orbital = self._hartree_couplings(
            hartree_matrix, hartree_matrix_orbital
        )

        # Dense-kernel fields: the backends read them, the superlattice paths
        # bypass them.
        self.VR_shifted = np.zeros((nkx, nky, 1, 1), dtype=np.complex128)
        self.HH = np.zeros((D, D))
        self.exchange_hermitian_channel_packing = False
        self.contact_g = np.zeros(1)
        self.contact_Oi = np.zeros((1, D, D), dtype=np.complex128)
        self.contact_Oj = self.contact_Oi

        self.embed_reference_potential = bool(embed_reference_potential)
        self.center_embedded_hartree = bool(center_embedded_hartree)
        self.embedded_energy_offset = 0.0
        self.embedded_hartree_shift = 0.0
        self.h = h
        if self.embed_reference_potential:
            if not self.has_reference_density:
                raise ValueError("embed_reference_potential=True requires reference_density.")
            V_ref, self.embedded_energy_offset = self._reference_potential()
            self.h = np.ascontiguousarray(h + V_ref)

    def _hartree_couplings(self, hartree_matrix, hartree_matrix_orbital):
        n_G, no = self.n_G, self.dim_orb
        if not self.include_hartree:
            return np.zeros((n_G, n_G)), None
        g_diff = np.linalg.norm(self.g_cart[:, None, :] - self.g_cart[None, :, :], axis=-1)
        if hartree_matrix_orbital is None and self.coulomb_V_orbital is not None:
            hartree_matrix_orbital = self.coulomb_V_orbital(g_diff)
        if hartree_matrix_orbital is not None:
            HH_orb = np.ascontiguousarray(hartree_matrix_orbital, dtype=np.float64)
            if HH_orb.shape != (n_G, n_G, no, no):
                raise ValueError(
                    f"orbital Hartree matrix must have shape {(n_G, n_G, no, no)}; got {HH_orb.shape}"
                )
            if not np.all(np.isfinite(HH_orb)):
                raise ValueError("orbital Hartree matrix must be finite")
            return np.zeros((n_G, n_G)), HH_orb
        if hartree_matrix is None:
            HH = np.asarray(self.coulomb_V(g_diff), dtype=np.float64)
            np.fill_diagonal(HH, 0.0)  # the gate neutralises the uniform charge
        else:
            HH = np.asarray(hartree_matrix, dtype=np.float64)
            if HH.shape != (n_G, n_G):
                raise ValueError(f"hartree_matrix must have shape ({n_G}, {n_G}); got {HH.shape}")
        return np.ascontiguousarray(HH), None

    # ----- linear maps (no reference subtraction) ---------------------------------

    def exchange(self, rho) -> np.ndarray:
        """Fock self-energy ``Σ_F[rho]`` on the BZ mesh."""
        if not self.include_exchange:
            return np.zeros(self.h.shape, dtype=np.complex128)
        from ._backends.native.fock import superlattice_exchange

        return superlattice_exchange(rho, self.layout, self.n_G)

    def hartree(self, rho) -> np.ndarray:
        """k-independent Hartree potential ``(D, D)`` of ``rho``."""
        if not self.include_hartree:
            return np.zeros((self.D, self.D), dtype=np.complex128)
        rho_bar = np.einsum("ij,ijab->ab", self.w2d, np.asarray(rho, dtype=np.complex128))
        return superlattice_hartree(
            rho_bar, self.layout.pair_to_delta, HH_GG=self.HH_GG,
            HH_GG_orbital=self.HH_GG_orbital, degeneracy=self.hartree_degeneracy,
        )

    def potential(self, rho) -> np.ndarray:
        """Hermitized ``V_HF[rho] = Σ_F[rho] + σ_H[rho]`` on the BZ mesh."""
        return _herm(self.exchange(rho) + self.hartree(rho)[None, None])

    def _reference_potential(self) -> tuple[np.ndarray, float]:
        """``V_HF[refP]`` as embedded into ``h`` (Hartree optionally centred) and ½Tr(V·refP)."""
        ref = self.refP
        V = self.exchange(ref) + self.hartree(ref)[None, None]
        # The offset is the uncentred ½Tr(V·refP); centring is a separate gauge shift.
        offset = 0.5 * float(np.einsum("ij,ijab,ijba->", self.w2d, V, ref).real)
        if self.center_embedded_hartree and self.include_hartree:
            self.embedded_hartree_shift = float(np.mean(np.diag(self.hartree(ref)).real))
            V = V - self.embedded_hartree_shift * np.eye(self.D)
        return V, offset

    def _as_args(self) -> dict:
        """Dense-kernel solver inputs (see :meth:`HartreeFockKernel._as_args`)."""
        return dict(
            h=self.h,
            weights_b=self.weights_b,
            weight_sum=self.weight_sum,
            VR=self.VR_shifted,
            T=self.T,
            refP=self.refP,
            HH=self.HH,
            include_hartree=self.include_hartree,
            include_exchange=self.include_exchange,
            exchange_hermitian_channel_packing=False,
            contact_g=self.contact_g,
            contact_Oi=self.contact_Oi,
            contact_Oj=self.contact_Oj,
        )

    # ----- off-mesh evaluation --------------------------------------------------

    def potential_at(self, P, k_fractional, *, include_embedded_reference: bool = True):
        """``V_HF[P - refP]`` at arbitrary fractional k (e.g. a band path).

        Direct plane-wave sum ``O(n_k · nk² · n_G²)``: matrix elements of
        ``Σ(k)`` in a truncated G basis are not gauge-periodic, so they cannot
        be interpolated from the mesh.  With ``include_embedded_reference``
        the embedded ``V_HF[refP]`` is added, so ``h_bare + potential_at``
        reproduces the mean-field Hamiltonian of an embedded kernel.
        """
        k = np.asarray(k_fractional, dtype=float)
        squeeze = k.ndim == 1
        k = np.atleast_2d(k)
        if k.shape[-1] != 2:
            raise ValueError(f"k_fractional must have shape (n, 2); got {k.shape}")
        P = np.asarray(P, dtype=np.complex128)
        refP = np.asarray(self.refP)
        embedded = include_embedded_reference and self.embed_reference_potential
        rho = P if embedded else P - refP
        sigma = self._exchange_at(rho, k) if self.include_exchange else 0.0
        if self.include_hartree:
            sigma = sigma + self.hartree(P - refP)[None]
            if embedded:
                sigma_ref = self.hartree(refP)
                if self.center_embedded_hartree:
                    sigma_ref = sigma_ref - np.mean(np.diag(sigma_ref).real) * np.eye(self.D)
                sigma = sigma + sigma_ref[None]
        sigma = _herm(np.broadcast_to(sigma, (k.shape[0], self.D, self.D)))
        return sigma[0] if squeeze else sigma

    def _exchange_at(self, rho, k):
        n_G, no, nkx, nky = self.n_G, self.dim_orb, self.nkx, self.nky
        rho = rho.reshape(nkx, nky, n_G, no, n_G, no)
        g = self.g_fractional
        lookup = {tuple(v): i for i, v in enumerate(g)}
        q = np.stack(
            np.meshgrid(
                self.k_origin[0] + np.arange(nkx) / nkx,
                self.k_origin[1] + np.arange(nky) / nky,
                indexing="ij",
            ),
            axis=-1,
        )
        sigma = np.zeros((k.shape[0], n_G, no, n_G, no), dtype=np.complex128)
        for shift in {tuple(a - b) for a in g for b in g}:
            pairs = [(i, lookup[tuple(gi + shift)]) for i, gi in enumerate(g) if tuple(gi + shift) in lookup]
            out, src = (np.asarray(x) for x in zip(*pairs))
            lag = np.einsum("ab,...b->...a", self.reciprocal_basis, k[:, None, None] - q[None] + shift)
            qmag = np.linalg.norm(lag, axis=-1)
            rho_sub = rho[:, :, src][:, :, :, :, src]
            if self.coulomb_V_orbital is not None:
                W = -self.w2d[None, ..., None, None] * np.asarray(self.coulomb_V_orbital(qmag))
                block = np.einsum("nxyab,xyuavb->nuavb", W, rho_sub, optimize=True)
            else:
                W = -self.w2d[None] * np.asarray(self.coulomb_V(qmag), dtype=float)
                block = np.einsum("nxy,xyuavb->nuavb", W, rho_sub, optimize=True)
            # Separated advanced indices move to the front: (u, v, n, α, β).
            sigma[:, out[:, None], :, out[None, :], :] += block.transpose(1, 3, 0, 2, 4)
        return sigma.reshape(k.shape[0], self.D, self.D)


def upsample_density(P_coarse, coarse: SuperlatticeHartreeFockKernel, fine: SuperlatticeHartreeFockKernel):
    """Spectral (FFT zero-padding) upsample of a periodic BZ density to the fine mesh.

    The coarse spectrum is phase-shifted by the offset between the two mesh
    origins (non-zero for Monkhorst-Pack meshes, whose half-cell shift
    depends on nk), so the result lands on the fine mesh points.
    """
    P = np.asarray(P_coarse, dtype=np.complex128)
    ncx, ncy, D, _ = P.shape
    nfx, nfy = fine.nkx, fine.nky
    if nfx < ncx or nfy < ncy:
        raise ValueError("the fine mesh must be at least as large as the coarse mesh")
    spec = np.fft.fftn(P, axes=(0, 1))
    dx = fine.k_origin[0] - coarse.k_origin[0]
    dy = fine.k_origin[1] - coarse.k_origin[1]
    if dx or dy:
        mx = np.fft.fftfreq(ncx) * ncx
        my = np.fft.fftfreq(ncy) * ncy
        spec = spec * np.exp(2j * np.pi * (mx[:, None] * dx + my[None, :] * dy))[..., None, None]
    pad = np.zeros((nfx, nfy, D, D), dtype=np.complex128)
    hx, hy = ncx // 2, ncy // 2
    tx, ty = nfx - (ncx - hx), nfy - (ncy - hy)
    pad[:hx, :hy] = spec[:hx, :hy]
    pad[:hx, ty:] = spec[:hx, hy:]
    pad[tx:, :hy] = spec[hx:, :hy]
    pad[tx:, ty:] = spec[hx:, hy:]
    return _herm(np.fft.ifftn(pad * (nfx * nfy) / (ncx * ncy), axes=(0, 1)))
