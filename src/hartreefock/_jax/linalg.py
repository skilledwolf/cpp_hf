"""Linear algebra helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from . import ffi


def hermitian_eigh(array: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Batched Hermitian eigh over the last two axes: ascending ``w`` and column eigenvectors.

    On CPU a complex128 batch goes to the native OpenMP kernel through
    ``jax.ffi`` (XLA CPU loops over the batch serially); elsewhere, and for
    other dtypes, it is ``jnp.linalg.eigh``.  Not differentiable.
    """
    array = jnp.asarray(array)
    if array.dtype != jnp.complex128 or array.ndim < 3 or not ffi.available(ffi.EIGH):
        w, v = jnp.linalg.eigh(array)
        return w, v
    call = jax.ffi.ffi_call(
        ffi.EIGH,
        (
            jax.ShapeDtypeStruct(array.shape[:-1], jnp.float64),
            jax.ShapeDtypeStruct(array.shape, jnp.complex128),
        ),
        vmap_method="broadcast_all",
    )

    def native(a):
        return tuple(call(a))

    def xla(a):
        return tuple(jnp.linalg.eigh(a))

    w, v = lax.platform_dependent(array, cpu=native, default=xla)
    return w, v



def hermitian_eigvalsh(array: jax.Array) -> jax.Array:
    """Ascending eigenvalues of a batch of Hermitian matrices (last two axes).

    The eigenvalue-only counterpart of :func:`hermitian_eigh`, on the native
    kernel for complex128 stacks on CPU.  Not differentiable.
    """
    array = jnp.asarray(array)
    if array.dtype != jnp.complex128 or array.ndim < 3 or not ffi.available(ffi.EIGVALSH):
        return jnp.linalg.eigvalsh(array)
    call = jax.ffi.ffi_call(
        ffi.EIGVALSH, jax.ShapeDtypeStruct(array.shape[:-1], jnp.float64),
        vmap_method="broadcast_all",
    )
    return lax.platform_dependent(array, cpu=call, default=jnp.linalg.eigvalsh)


def _native_matmul(a, b):
    out = jax.ShapeDtypeStruct(a.shape[:-1] + b.shape[-1:], jnp.complex128)
    return jax.ffi.ffi_call(ffi.MATMUL, out, vmap_method="broadcast_all")(a, b)


@jax.custom_jvp
def _matmul_c128(a, b):
    return lax.platform_dependent(a, b, cpu=_native_matmul, default=jnp.matmul)


@_matmul_c128.defjvp
def _matmul_c128_jvp(primals, tangents):
    (a, b), (da, db) = primals, tangents
    return _matmul_c128(a, b), _matmul_c128(da, b) + _matmul_c128(a, db)


def adjoint(a: jax.Array) -> jax.Array:
    """Conjugate transpose over the last two axes."""
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def bmm(a: jax.Array, b: jax.Array) -> jax.Array:
    """Batched ``a @ b`` over the last two axes.

    On CPU a complex128 product of stacked matrices goes to the native
    OpenMP kernel through ``jax.ffi`` (XLA CPU runs a batched dot serially);
    otherwise it is ``jnp.matmul``.  Forward-mode differentiable.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    if (
        a.dtype != jnp.complex128 or b.dtype != jnp.complex128
        or max(a.ndim, b.ndim) < 3 or min(a.ndim, b.ndim) < 2
        or not ffi.available(ffi.MATMUL)
    ):
        return jnp.matmul(a, b)
    batch = jnp.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = jnp.broadcast_to(a, batch + a.shape[-2:])
    b = jnp.broadcast_to(b, batch + b.shape[-2:])
    return _matmul_c128(a, b)

def eigh(array: jax.Array, sectors: tuple[tuple[int, ...], ...] | None = None):
    """:func:`hermitian_eigh`, one block per sector when ``sectors`` are given.

    ``sectors`` partitions the basis into index tuples that the matrix does
    not couple (``HartreeFockKernel.sectors``); the blocks are diagonalized
    separately and the eigenvalues come back ascending across all of them.
    """
    if sectors is None:
        return hermitian_eigh(array)
    array = jnp.asarray(array)
    ws = []
    v_full = jnp.zeros(array.shape, dtype=array.dtype)
    col = 0
    for idx in sectors:
        i = jnp.asarray(idx, dtype=jnp.int32)
        w, v = hermitian_eigh(array[..., i[:, None], i[None, :]])
        ws.append(w)
        v_full = v_full.at[..., i, col : col + len(idx)].set(v)
        col += len(idx)
    w = jnp.concatenate(ws, axis=-1)
    order = jnp.argsort(w, axis=-1)
    return (
        jnp.take_along_axis(w, order, axis=-1),
        jnp.take_along_axis(v_full, order[..., None, :], axis=-1),
    )
