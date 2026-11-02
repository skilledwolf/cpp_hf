"""Plane-wave (superlattice) pieces of :class:`~hartreefock.HartreeFockKernel`.

With ``plane_waves`` the basis is ``|k, G_a, α⟩`` with ``G_a`` from a
discrete reciprocal-lattice basis and ``α`` an orbital index, so
``D = n_G · n_orb``.  The basis state carries absolute momentum
``p = k - G_a`` and the Fock self-energy::

    Σ_F[(G_a α), (G_b β)](k) = -Σ_{q, G_0} V(|k - q + G_0|) w ρ[(G_a+G_0) α, (G_b+G_0) β](q)

is one aperiodic 2D convolution in ``p`` per channel ``(ΔG = G_a - G_b, α, β)``.
:func:`build_extended_layout` precomputes the zero-padded extended grid, the
FFT of ``V`` on it and the CSR grouping of ``(G_a, G_b)`` pairs by ``ΔG``; the
exchange streams the channels through one scratch buffer.  The
Hartree term is k-independent and couples ``ΔG``-components of the BZ-averaged
density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .mesh import Mesh
from .symmetry import array_namespace

__all__ = [
    "ExtendedGridLayout",
    "build_extended_layout",
    "exchange_at",
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
    V_lag_fft : (N_ext_x, N_ext_y) complex128, or None
        FFT of ``w · V`` on the lag grid (``w`` the uniform BZ weight).
    V_lag_fft_orbital : (N_ext_x, N_ext_y, dim_orb, dim_orb) complex128, or None
        The same for an orbital-resolved ``V``; exactly one of the two is set.
    """

    N_ext_x: int
    N_ext_y: int
    g_a_off: np.ndarray
    n_delta: int
    pair_to_delta: np.ndarray
    delta_pair_i: np.ndarray
    delta_pair_j: np.ndarray
    delta_pair_start: np.ndarray
    V_lag_fft: np.ndarray | None
    V_lag_fft_orbital: np.ndarray | None


def build_extended_layout(
    g_fractional,
    reciprocal_basis,
    nkx: int,
    nky: int,
    V: Callable[[np.ndarray], np.ndarray],
    w_scalar: float,
) -> ExtendedGridLayout:
    """Precompute the extended grid for G-basis ``g_fractional`` on an ``nkx × nky`` BZ mesh.

    ``reciprocal_basis`` is the 2×2 matrix ``B`` with ``G_cart = B @ G_frac``.
    ``V(|q|)`` maps momentum magnitudes to the interaction, of shape
    ``q.shape`` or ``q.shape + (dim_orb, dim_orb)`` for an orbital-resolved
    exchange.
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
    V_q = np.asarray(V(qmag))
    if V_q.shape != qmag.shape and (V_q.ndim != 4 or V_q.shape[:2] != qmag.shape
                                    or V_q.shape[2] != V_q.shape[3]):
        raise ValueError(
            f"V must map a {qmag.shape} grid to {qmag.shape} or {qmag.shape} + (n, n); "
            f"got {V_q.shape}"
        )
    if not np.all(np.isfinite(V_q)):
        raise ValueError("V must return finite values")
    V_fft = np.ascontiguousarray(
        np.fft.fftn(float(w_scalar) * V_q, axes=(0, 1)), dtype=np.complex128
    )
    orbital = V_q.ndim == 4

    return ExtendedGridLayout(
        N_ext_x=N_ext_x,
        N_ext_y=N_ext_y,
        g_a_off=g_a_off,
        n_delta=n_delta,
        pair_to_delta=pair_to_delta.reshape(n_G, n_G),
        delta_pair_i=np.ascontiguousarray(order // n_G, dtype=np.int64),
        delta_pair_j=np.ascontiguousarray(order % n_G, dtype=np.int64),
        delta_pair_start=delta_pair_start,
        V_lag_fft=None if orbital else V_fft,
        V_lag_fft_orbital=V_fft if orbital else None,
    )


def superlattice_hartree(rho_bar, pair_to_delta: np.ndarray, HH: np.ndarray, *, degeneracy: float = 1.0):
    """k-independent Hartree potential ``(D, D)`` of the BZ-averaged density ``rho_bar``.

    Scalar coupling ``HH`` of shape ``(n_G, n_G)``:
    ``σ_H[(A α), (B α)] = g · HH[A, B] · Σ_γ ρ_γ(ΔG_AB)``.  Orbital coupling
    of shape ``(n_G, n_G, n, n)``:
    ``σ_H[(A α), (B α)] = g · Σ_γ HH[A, B, α, γ] ρ_γ(ΔG_AB)``, with ``ρ_γ(d)``
    the sum of ``rho_bar[(A' γ), (B' γ)]`` over pairs with ``ΔG = d``.
    Works on NumPy and JAX arrays (also under ``jax.jit``).
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
    HH = xp.asarray(HH).astype(rho_bar.dtype)
    if HH.ndim == 2:
        sigma_GG = degeneracy * HH * xp.sum(rho_at_pair, axis=-1)
        sigma = xp.einsum("AB,xy->AxBy", sigma_GG, eye)
    else:
        sigma_orb = degeneracy * xp.einsum("ABag,ABg->ABa", HH, rho_at_pair)
        sigma = xp.einsum("ABa,ab->AaBb", sigma_orb, eye)
    return sigma.reshape(D, D)


def _herm(A):
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


def exchange_at(kernel, rho, k):
    """Exchange ``Σ_F[rho]`` of a plane-wave ``kernel`` at fractional momenta ``k`` ``(n, 2)``."""
    n_G, no, mesh = kernel.n_G, kernel.dim_orb, kernel.mesh
    nk, D = mesh.nk, kernel.h.shape[-1]
    rho = rho.reshape(nk, nk, n_G, no, n_G, no)
    g = kernel.plane_waves
    lookup = {tuple(v): i for i, v in enumerate(g)}
    q = mesh.frac_grid
    sigma = np.zeros((k.shape[0], n_G, no, n_G, no), dtype=np.complex128)
    for shift in {tuple(a - b) for a in g for b in g}:
        pairs = [(i, lookup[tuple(gi + shift)]) for i, gi in enumerate(g) if tuple(gi + shift) in lookup]
        out, src = (np.asarray(x) for x in zip(*pairs))
        lag = np.einsum("ab,...b->...a", mesh.cell, k[:, None, None] - q[None] + shift)
        V = -mesh.weight * np.asarray(kernel.V(np.linalg.norm(lag, axis=-1)))
        rho_sub = rho[:, :, src][:, :, :, :, src]
        if kernel.orbital_V:
            block = np.einsum("nxyab,xyuavb->nuavb", V, rho_sub, optimize=True)
        else:
            block = np.einsum("nxy,xyuavb->nuavb", V, rho_sub, optimize=True)
        # Separated advanced indices move to the front: (u, v, n, α, β).
        sigma[:, out[:, None], :, out[None, :], :] += block.transpose(1, 3, 0, 2, 4)
    return sigma.reshape(k.shape[0], D, D)


def upsample_density(P_coarse, coarse: Mesh, fine: Mesh):
    """Spectral (FFT zero-padding) upsample of a periodic BZ density from mesh ``coarse`` to ``fine``.

    The coarse spectrum is phase-shifted by the offset between the two mesh
    origins (non-zero for Monkhorst-Pack meshes, whose half-cell shift
    depends on nk), so the result lands on the fine mesh points.
    """
    P = np.asarray(P_coarse, dtype=np.complex128)
    ncx, ncy, D, _ = P.shape
    nfx = nfy = fine.nk
    if nfx < ncx or nfy < ncy:
        raise ValueError("the fine mesh must be at least as large as the coarse mesh")
    spec = np.fft.fftn(P, axes=(0, 1))
    dx = dy = fine.frac[0] - coarse.frac[0]
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
