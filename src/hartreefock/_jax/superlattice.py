"""Superlattice Fock builder.

:func:`superlattice_build_fock` turns a plane-wave
:class:`~hartreefock.HartreeFockKernel` into a Fock
builder with the signature of :func:`~hartreefock._jax.fock.build_fock`,
so the jitted SCF, direct-minimization and stability loops run on it
unchanged.  Exchange streams one ``ΔG`` channel (or one ``(ΔG, α, β)``
channel with an orbital interaction) through a single extended-grid scratch
buffer, so peak device memory does not grow with the number of channels.
The builder supports JIT and forward-mode JVP.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..superlattice import ExtendedGridLayout, superlattice_hartree
from . import ffi


def make_superlattice_fock_fn(
    layout: ExtendedGridLayout, n_G: int, dim_orb: int, nkx: int, nky: int,
) -> Callable[[jax.Array], jax.Array]:
    """JIT-compiled ``rho -> Σ_F[rho]`` for densities of shape ``(nkx, nky, D, D)``.

    Uses ``layout.V_lag_fft_orbital`` when present, else the scalar table.
    On CPU in double precision this is the C++ OpenMP streaming kernel
    through ``jax.ffi``.
    """
    if jax.config.jax_enable_x64 and ffi.on_cpu() and ffi.available(ffi.SUPERLATTICE_FOCK):
        return _make_ffi_superlattice_fock_fn(layout, n_G, dim_orb, nkx, nky)
    if layout.V_lag_fft_orbital is not None:
        return _make_orbital_superlattice_fock_fn(
            layout, n_G, dim_orb, nkx, nky, layout.V_lag_fft_orbital,
        )
    return _make_scalar_superlattice_fock_fn(layout, n_G, dim_orb, nkx, nky)


def _make_scalar_superlattice_fock_fn(layout, n_G, dim_orb, nkx, nky):
    """Stream the ``ΔG`` channels of a scalar interaction, unrolled in HLO."""
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


def _make_ffi_superlattice_fock_fn(layout, n_G, dim_orb, nkx, nky):
    """``rho -> Σ_F[rho]`` through the C++ streaming kernel (CPU only).

    The Hermitian ΔG halving applies only to Hermitian inputs.
    """
    D = n_G * dim_orb
    shape = (nkx, nky, D, D)
    V_orb = layout.V_lag_fft_orbital
    # The handler reads the extended grid size off the scalar table.
    V_scalar = layout.V_lag_fft
    if V_scalar is None:
        V_scalar = np.zeros((layout.N_ext_x, layout.N_ext_y), np.complex128)
    tables = (
        jnp.asarray(V_scalar, dtype=jnp.complex128),
        jnp.asarray(
            np.zeros((0,), np.complex128) if V_orb is None else V_orb, dtype=jnp.complex128
        ),
        jnp.asarray(layout.g_a_off, dtype=jnp.int64),
        jnp.asarray(layout.delta_pair_i, dtype=jnp.int64),
        jnp.asarray(layout.delta_pair_j, dtype=jnp.int64),
        jnp.asarray(layout.delta_pair_start, dtype=jnp.int64),
    )
    call = jax.ffi.ffi_call(
        ffi.SUPERLATTICE_FOCK,
        jax.ShapeDtypeStruct(shape, jnp.complex128),
        vmap_method="sequential",
    )

    def apply(rho):
        if rho.shape != shape:
            raise ValueError(f"density must have shape {shape}; got {rho.shape}")
        rho = jnp.asarray(rho, dtype=jnp.complex128)
        # Densities and most response tangents are Hermitian; those take the
        # ΔG halving, anything else the full channel set.
        skew = jnp.max(jnp.abs(rho - jnp.conj(jnp.swapaxes(rho, -1, -2))))
        hermitian = skew <= 1e-14 * jnp.maximum(jnp.max(jnp.abs(rho)), 1e-300)
        return lax.cond(
            hermitian,
            lambda r: call(r, *tables, hermitian_rho=True),
            lambda r: call(r, *tables, hermitian_rho=False),
            rho,
        )

    # Σ_F is linear in rho, so its JVP is the same kernel on the tangent.
    fock_fn = jax.custom_jvp(apply)
    fock_fn.defjvp(lambda primals, tangents: (apply(primals[0]), apply(tangents[0])))

    return jax.jit(fock_fn)


_BUILDERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def superlattice_build_fock(kernel) -> Callable:
    """Fock builder for a superlattice kernel (cached per kernel so jit caches hit).

    Exchange is the streaming convolution above; the k-independent Hartree
    term is folded into ``Sigma`` and the returned ``H`` is zero, so
    :func:`~hartreefock._jax.fock.hf_energy` is unchanged.
    """
    cached = _BUILDERS.get(kernel)
    if cached is not None:
        return cached
    n_G, dim_orb = kernel.n_G, kernel.dim_orb
    nk = kernel.mesh.nk
    fock_fn = make_superlattice_fock_fn(kernel.layout, n_G, dim_orb, nk, nk)
    pair_to_delta = np.asarray(kernel.layout.pair_to_delta)
    HH, degeneracy = kernel.HH, kernel.degeneracy

    def build_fock_sl(P, *, h, refP, w2d, include_exchange, include_hartree,
                      project_fn=None, **_dense_only):
        rho = P - refP
        Sigma = fock_fn(rho) if include_exchange else jnp.zeros_like(P)
        if include_hartree:
            rho_bar = jnp.einsum("ij,ijab->ab", w2d, rho)
            Sigma = Sigma + superlattice_hartree(
                rho_bar, pair_to_delta, HH, degeneracy=degeneracy
            )[None, None]
        F = h + Sigma
        F = 0.5 * (F + jnp.conj(jnp.swapaxes(F, -1, -2)))
        if project_fn is not None:
            F = project_fn(F)
            F = 0.5 * (F + jnp.conj(jnp.swapaxes(F, -1, -2)))
        return Sigma, jnp.zeros_like(P), F

    _BUILDERS[kernel] = build_fock_sl
    return build_fock_sl
