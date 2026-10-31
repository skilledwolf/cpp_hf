"""Streaming superlattice Fock self-energy (JAX).

For Hamiltonians in a plane-wave basis indexed by a moire/superlattice
reciprocal-vector basis ``g_basis`` (TBG, modulated bilayer, twisted MoTe_2,
generic SuperlatticeModel), the Fock self-energy

    Σ_F[α G_a, β G_b](k) = -Σ_{q, G_0} V(|k - q + G_0|)
                           · ρ[α (G_a+G_0), β (G_b+G_0)](q)

is a 2D convolution in absolute momentum p = k - G, decoupled into
independent (α, β, ΔG = G_a - G_b) channels.  This module provides a
self-contained extended-grid layout builder (pure NumPy) and a
:func:`make_superlattice_fock_fn` factory that returns a JIT-compiled JAX
function streaming one ΔG channel at a time through a single small scratch
buffer.  Peak device memory is O(N_ext^2 · dim_orb^2), independent of
n_delta — the HLO graph is also smaller than a single batched FFT over all
channels, which cuts compile time on moderate-to-large layouts. An optional
orbital kernel replaces V by V_{αβ} and streams one (ΔG, α, β) channel at
a time through a 2D scratch buffer. Its kernel table costs
O(N_ext_x N_ext_y dim_orb²); scratch memory is O(N_ext_x N_ext_y), without
an additional n_delta factor. Both paths retain every orbital coherence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp

from .fock import _contact_self_energy


@dataclass(frozen=True)
class ExtendedGridLayout:
    """Indexing data for the extended-momentum-grid superlattice Fock.

    Pure-NumPy dataclass; converted to JAX arrays at JIT-trace time inside
    :func:`make_superlattice_fock_fn`.

    Attributes
    ----------
    N_ext_x, N_ext_y : int
        Extended-grid dimensions (each ``= 2 · nk · G_dim``).
    n_supp_x, n_supp_y : int
        Support dimensions: ``nk · G_dim``.
    g_a_off : (n_G, 2) int64 array
        Per-G tile offset (in units of ``nkx``, ``nky``).
    n_delta : int
        Number of unique ΔG = G_a − G_b values.
    pair_to_delta : (n_G, n_G) int64 array
        For each (G_a, G_b), the index into the unique-ΔG list.
    V_lag_fft : (N_ext_x, N_ext_y) complex128 array
        FFT of the Cartesian Coulomb kernel on the FFT-natural lag grid,
        scaled by the BZ-integration weight scalar.
    V_lag_fft_orbital : (N_ext_x, N_ext_y, dim_orb, dim_orb) complex array, optional
        Orbital-pair interaction on the same weighted lag grid. When
        supplied it replaces the scalar exchange kernel.
    delta_pair_count : (n_delta,) int64 array
    delta_pair_i, delta_pair_j : (n_pairs,) int64 arrays
        (G_a, G_b) index pairs grouped by ΔG.
    delta_pair_start : (n_delta + 1,) int64 array
        CSR-style row pointer.
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
    # Optional density-density interaction resolved over orbital pairs.
    # Appended so existing positional construction remains compatible.
    V_lag_fft_orbital: np.ndarray | None = None


def build_extended_layout(
    g_basis_fractional: np.ndarray,
    g_basis_B: np.ndarray,
    nkx: int,
    nky: int,
    coulomb_V: Callable[[np.ndarray], np.ndarray],
    w_scalar: float,
    *,
    coulomb_V_orbital: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ExtendedGridLayout:
    """Precompute index arrays + FFT'd Coulomb kernel for the extended grid.

    Parameters mirror :func:`cpp_hf.superlattice.build_extended_layout` so
    contimod (or any external wrapper) can swap backends without rebuilding
    its layout-construction call. ``coulomb_V_orbital(qmag)`` optionally
    returns ``(*qmag.shape, dim_orb, dim_orb)`` and takes precedence over
    the scalar Coulomb for exchange. A physical radial density-density
    interaction is real and symmetric in the orbital indices.
    """
    g_frac = np.asarray(g_basis_fractional, dtype=np.int64)
    n_G = int(g_frac.shape[0])
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
    V_lag_fft_orbital = None
    if coulomb_V_orbital is not None:
        if not callable(coulomb_V_orbital):
            raise ValueError("coulomb_V_orbital must be callable or None")
        V_orbital = np.asarray(coulomb_V_orbital(qmag))
        if (V_orbital.ndim != 4 or V_orbital.shape[:2] != qmag.shape
                or V_orbital.shape[-2] != V_orbital.shape[-1]
                or V_orbital.shape[-1] == 0):
            raise ValueError(
                "coulomb_V_orbital must return shape "
                f"{qmag.shape} + (dim_orb, dim_orb); got {V_orbital.shape}"
            )
        if not np.all(np.isfinite(V_orbital)):
            raise ValueError("coulomb_V_orbital must return finite values")
        V_lag_fft_orbital = np.ascontiguousarray(
            np.fft.fftn(float(w_scalar) * V_orbital, axes=(0, 1)),
            dtype=np.complex128,
        )

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
        V_lag_fft_orbital=V_lag_fft_orbital,
    )


def make_superlattice_fock_fn(
    layout: ExtendedGridLayout,
    n_G: int,
    dim_orb: int,
    nkx: int,
    nky: int,
    *,
    V_lag_fft_orbital: np.ndarray | None = None,
) -> Callable[[jax.Array], jax.Array]:
    """Build a JIT-compiled streaming superlattice Fock function.

    The returned function accepts the density matrix shape
    ``(nkx, nky, n_G * dim_orb, n_G * dim_orb)`` and returns the Fock
    self-energy with the same shape.  The JIT'd body iterates ΔG channels
    with one shared ``(N_ext_x, N_ext_y, dim_orb, dim_orb)`` scratch buffer
    — XLA reuses the buffer across iterations so peak device memory is
    O(N_ext^2 · dim_orb^2), independent of n_delta. If an orbital kernel
    is supplied explicitly or stored in ``layout.V_lag_fft_orbital``, it
    replaces the scalar exchange kernel. That path uses structured JAX
    loops and one 2D scratch buffer per (ΔG, α, β), with no tensor carrying
    all ΔG channels. Input/output densities still require
    O(nkx nky (n_G dim_orb)²) memory. The returned map supports JIT and JVP.
    """
    if V_lag_fft_orbital is None:
        V_lag_fft_orbital = getattr(layout, "V_lag_fft_orbital", None)
    if V_lag_fft_orbital is not None:
        return _make_orbital_superlattice_fock_fn(
            layout, n_G, dim_orb, nkx, nky, V_lag_fft_orbital,
        )
    g_a_off = np.asarray(layout.g_a_off)
    delta_pair_i = np.asarray(layout.delta_pair_i)
    delta_pair_j = np.asarray(layout.delta_pair_j)
    delta_pair_start = np.asarray(layout.delta_pair_start)
    V_lag_fft = jnp.asarray(layout.V_lag_fft)
    N_ext_x = int(layout.N_ext_x)
    N_ext_y = int(layout.N_ext_y)
    n_delta = int(layout.n_delta)

    pairs_for_delta: list[list[tuple[int, int]]] = []
    for d in range(n_delta):
        a = int(delta_pair_start[d])
        b = int(delta_pair_start[d + 1])
        pairs_for_delta.append([
            (int(delta_pair_i[k]), int(delta_pair_j[k])) for k in range(a, b)
        ])

    D = n_G * dim_orb

    @jax.jit
    def fock_fn(rho_kk: jax.Array) -> jax.Array:
        rho_r = rho_kk.reshape(nkx, nky, n_G, dim_orb, n_G, dim_orb)
        sigma = jnp.zeros(
            (nkx, nky, n_G, dim_orb, n_G, dim_orb), dtype=rho_r.dtype,
        )
        for d in range(n_delta):
            rho_dg = jnp.zeros(
                (N_ext_x, N_ext_y, dim_orb, dim_orb), dtype=rho_r.dtype,
            )
            for (i, j) in pairs_for_delta[d]:
                ox = int(nkx * int(g_a_off[i, 0]))
                oy = int(nky * int(g_a_off[i, 1]))
                rho_dg = rho_dg.at[ox:ox + nkx, oy:oy + nky, :, :].set(
                    rho_r[:, :, i, :, j, :]
                )
            rho_fft = jnp.fft.fftn(rho_dg, axes=(0, 1))
            sigma_dg = -jnp.fft.ifftn(
                rho_fft * V_lag_fft[:, :, None, None], axes=(0, 1),
            )
            for (i, j) in pairs_for_delta[d]:
                ox = int(nkx * int(g_a_off[i, 0]))
                oy = int(nky * int(g_a_off[i, 1]))
                sigma = sigma.at[:, :, i, :, j, :].set(
                    sigma_dg[ox:ox + nkx, oy:oy + nky, :, :]
                )
        return sigma.reshape(nkx, nky, D, D)

    return fock_fn


def _make_orbital_superlattice_fock_fn(
    layout, n_G, dim_orb, nkx, nky, V_lag_fft_orbital,
):
    """Stream orbital-pair channels without unrolling the ΔG layout in HLO."""
    nx, ny, n_delta = int(layout.N_ext_x), int(layout.N_ext_y), int(layout.n_delta)
    expected = (nx, ny, dim_orb, dim_orb)
    V_np = np.asarray(V_lag_fft_orbital)
    if V_np.shape != expected:
        raise ValueError(f"V_lag_fft_orbital must have shape {expected}; got {V_np.shape}")
    if not np.all(np.isfinite(V_np)):
        raise ValueError("V_lag_fft_orbital must contain only finite values")
    V_fft = jnp.asarray(V_np)
    pair_i = jnp.asarray(layout.delta_pair_i, dtype=jnp.int32)
    pair_j = jnp.asarray(layout.delta_pair_j, dtype=jnp.int32)
    pair_start = jnp.asarray(layout.delta_pair_start, dtype=jnp.int32)
    offsets = jnp.asarray(layout.g_a_off, dtype=jnp.int32)
    D = n_G * dim_orb
    expected_density = (nkx, nky, D, D)

    @jax.jit
    def orbital_fock(rho_kk):
        if rho_kk.shape != expected_density:
            raise ValueError(f"density must have shape {expected_density}; got {rho_kk.shape}")
        dtype = jnp.result_type(rho_kk.dtype, jnp.complex64)
        rho = jnp.asarray(rho_kk, dtype=dtype).reshape(nkx, nky, n_G, dim_orb, n_G, dim_orb)
        zero = jnp.int32(0)

        def channel_body(channel, sigma):
            channel = jnp.asarray(channel, dtype=jnp.int32)
            d = channel // (dim_orb * dim_orb)
            alpha = (channel // dim_orb) % dim_orb
            beta = channel % dim_orb

            def scatter(k, scratch):
                i, j = pair_i[k], pair_j[k]
                tile = jax.lax.dynamic_slice(
                    rho, (zero, zero, i, alpha, j, beta), (nkx, nky, 1, 1, 1, 1),
                ).reshape(nkx, nky)
                return jax.lax.dynamic_update_slice(
                    scratch, tile, (nkx * offsets[i, 0], nky * offsets[i, 1]),
                )

            scratch = jax.lax.fori_loop(
                pair_start[d], pair_start[d + 1], scatter,
                jnp.zeros((nx, ny), dtype=dtype),
            )
            interaction = jax.lax.dynamic_slice(
                V_fft, (zero, zero, alpha, beta), (nx, ny, 1, 1),
            ).reshape(nx, ny).astype(dtype)
            response = -jnp.fft.ifftn(jnp.fft.fftn(scratch) * interaction)

            def gather(k, result):
                i, j = pair_i[k], pair_j[k]
                tile = jax.lax.dynamic_slice(
                    response, (nkx * offsets[i, 0], nky * offsets[i, 1]), (nkx, nky),
                ).reshape(nkx, nky, 1, 1, 1, 1)
                return jax.lax.dynamic_update_slice(
                    result, tile, (zero, zero, i, alpha, j, beta),
                )

            return jax.lax.fori_loop(pair_start[d], pair_start[d + 1], gather, sigma)

        sigma = jax.lax.fori_loop(
            0, n_delta * dim_orb * dim_orb, channel_body, jnp.zeros_like(rho),
        )
        return sigma.reshape(expected_density)

    return orbital_fock


def make_superlattice_build_fock_fn(
    layout: ExtendedGridLayout,
    n_G: int,
    dim_orb: int,
    nkx: int,
    nky: int,
    HH_GG: np.ndarray,
    hartree_degeneracy: float,
    HH_GG_orbital: np.ndarray | None = None,
    *,
    V_lag_fft_orbital: np.ndarray | None = None,
) -> Callable:
    """Adapter matching the signature of :func:`jax_hf.fock.build_fock`.

    The returned function plugs into :func:`jax_hf.solve_scf` as its
    ``fock_build_fn``.  It computes the streaming superlattice exchange and
    absorbs the k-independent Hartree shift directly into ``Sigma``; the
    returned ``H`` is zero so :func:`jax_hf.fock.hf_energy` (which uses
    ``Sigma + H`` interchangeably) keeps working unchanged.  Contact terms
    passed through the standard build-Fock signature are also accumulated
    into ``Sigma``. Orbital exchange is read from the layout or from the
    optional ``V_lag_fft_orbital`` override, including response-Fock JVPs.

    Parameters
    ----------
    HH_GG : (n_G, n_G) real array
        Scalar Coulomb on G-basis differences, diagonal zeroed.  Used when
        ``HH_GG_orbital`` is ``None``.
    HH_GG_orbital : (n_G, n_G, dim_orb, dim_orb) real array, optional
        Orbital-resolved Coulomb on G-basis differences.  When supplied,
        the Hartree shift uses the per-orbital formula

            σ_H[A, α, B, α] = degeneracy · Σ_γ HH_GG_orbital[A, B, α, γ]
                                              · ρ_γ(ΔG_{AB}),

        where ``ρ_γ(d) = Σ_{(A',B'): pair_to_delta(A',B')=d} ρ_bar[(A', γ), (B', γ)]``.
        The diagonal-G blocks (``HH_GG_orbital[i, i, :, :]``) are the q=0 layer
        Hartree.  Whether to keep them is the caller's physical convention: the
        canonical moiré/TBG convention drops them (the gate neutralises the
        uniform charge, so they are zero), while a layer-resolved system at
        finite displacement field keeps the gate-screened q=0 layer Hartree.
        A nonzero diagonal is therefore allowed but *warned* about, since a
        stray nonzero diagonal is also a common bug (forgetting to drop q=0).
        Matches the CPU ``cpp_hf.superlattice`` semantics.
    """
    fock_fn = make_superlattice_fock_fn(
        layout, n_G, dim_orb, nkx, nky, V_lag_fft_orbital=V_lag_fft_orbital,
    )
    pair_to_delta_jax = jnp.asarray(np.asarray(layout.pair_to_delta, dtype=np.int64))
    HH_GG_jax = jnp.asarray(np.asarray(HH_GG, dtype=np.float64))
    n_delta = int(layout.n_delta)
    degeneracy = float(hartree_degeneracy)
    D = n_G * dim_orb

    orbital_path = HH_GG_orbital is not None
    if orbital_path:
        HH_orb_np = np.asarray(HH_GG_orbital, dtype=np.float64)
        expected = (n_G, n_G, dim_orb, dim_orb)
        if HH_orb_np.shape != expected:
            raise ValueError(
                f"HH_GG_orbital must have shape {expected}; got {HH_orb_np.shape}"
            )
        diag_blocks = HH_orb_np[np.arange(n_G), np.arange(n_G)]
        if float(np.max(np.abs(diag_blocks))) > 1e-12:
            # The diagonal-G blocks are the q=0 layer Hartree.  Keeping them is
            # a valid convention (layer-resolved system at finite displacement
            # field); dropping them is the moiré/TBG convention.  Warn rather
            # than raise so both work, since a stray nonzero diagonal is also a
            # common bug (forgetting to drop q=0).
            warnings.warn(
                "HH_GG_orbital has nonzero diagonal-G blocks; the q=0 layer "
                "Hartree will be included.  This is intended for a layer-resolved "
                "system at finite displacement field; for the moiré/TBG "
                "convention pass HH_GG_orbital[i, i, :, :] = 0.",
                stacklevel=2,
            )
        HH_orb_jax = jnp.asarray(HH_orb_np)
    else:
        HH_orb_jax = None

    def build_fock_sl(
        P,
        *,
        h, VR, refP, HH, w2d,
        include_exchange,
        include_hartree,
        exchange_hermitian_channel_packing,
        contact_g, contact_Oi, contact_Oj,
        project_fn=None,
        exchange_block_specs=None,
        exchange_check_offdiag=None,
        exchange_offdiag_atol=1e-12,
        exchange_offdiag_rtol=0.0,
    ):
        rho = P - refP
        rho_bar_full = jnp.einsum("ij,ijab->ab", w2d, rho)
        # Exchange channel (streaming).
        if include_exchange:
            sigma_F = fock_fn(rho)
        else:
            sigma_F = jnp.zeros_like(P)

        # Superlattice Hartree (k-independent shift), absorbed into Sigma.
        if include_hartree:
            rho_bar = rho_bar_full.reshape(
                n_G, dim_orb, n_G, dim_orb,
            )
            if orbital_path:
                # Per-orbital ΔG density: ρ_γ(d) = Σ_{(A,B): pair_to_delta(A,B)=d}
                #                                  rho_bar[(A, γ), (B, γ)].
                rho_perorb = jnp.einsum("AxBx->ABx", rho_bar)  # (n_G, n_G, dim_orb)
                rho_for_delta_perorb = (
                    jnp.zeros((n_delta, dim_orb), dtype=rho_perorb.dtype)
                    .at[pair_to_delta_jax.ravel()]
                    .add(rho_perorb.reshape(n_G * n_G, dim_orb))
                )
                rho_at_pair = rho_for_delta_perorb[pair_to_delta_jax]  # (n_G, n_G, dim_orb)
                # σ_H[A, α, B, α] = degeneracy · Σ_γ HH_GG_orbital[A, B, α, γ] · ρ_γ(d(A, B)).
                sigma_perorb = degeneracy * jnp.einsum(
                    "ABag,ABg->ABa", HH_orb_jax.astype(rho_at_pair.dtype), rho_at_pair,
                )
                sigma_H_blocks = jnp.zeros(
                    (n_G, dim_orb, n_G, dim_orb), dtype=sigma_perorb.dtype,
                )
                for alpha in range(dim_orb):
                    sigma_H_blocks = sigma_H_blocks.at[:, alpha, :, alpha].set(
                        sigma_perorb[:, :, alpha]
                    )
            else:
                rho_GG = jnp.einsum("AxBx->AB", rho_bar)
                rho_for_delta = jnp.zeros(n_delta, dtype=rho_GG.dtype).at[
                    pair_to_delta_jax.ravel()
                ].add(rho_GG.ravel())
                sigma_GG = degeneracy * HH_GG_jax * rho_for_delta[pair_to_delta_jax]
                sigma_H_blocks = jnp.zeros(
                    (n_G, dim_orb, n_G, dim_orb), dtype=sigma_GG.dtype,
                )
                for xi in range(dim_orb):
                    sigma_H_blocks = sigma_H_blocks.at[:, xi, :, xi].set(sigma_GG)
            sigma_H = sigma_H_blocks.reshape(D, D)
            Sigma = sigma_F + sigma_H[None, None, :, :]
        else:
            Sigma = sigma_F

        sigma_contact = _contact_self_energy(
            rho_bar_full,
            contact_g,
            contact_Oi,
            contact_Oj,
        ).astype(P.dtype)
        Sigma = Sigma + sigma_contact[None, None, :, :]

        H_zero = jnp.zeros_like(P)
        F = h + Sigma
        F = 0.5 * (F + jnp.conj(jnp.swapaxes(F, -1, -2)))
        if project_fn is not None:
            F = project_fn(F)
            F = 0.5 * (F + jnp.conj(jnp.swapaxes(F, -1, -2)))
        return Sigma, H_zero, F

    return build_fock_sl


__all__ = [
    "ExtendedGridLayout",
    "build_extended_layout",
    "make_superlattice_fock_fn",
    "make_superlattice_build_fock_fn",
]
