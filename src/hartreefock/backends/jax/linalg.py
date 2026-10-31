"""Linear algebra helpers for jax_hf."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

# Canonical, JIT-friendly block spec representation:
#   ("sizes",   (n1, n2, ...)) or
#   ("indices", ((i1, i2, ...), (j1, j2, ...), ...))
BlockSpec = tuple[str, tuple[Any, ...]]


def normalize_block_specs(block_specs: Any) -> tuple[BlockSpec, ...] | None:
    """Normalize loose block specs into a hashable tuple form.

    Supports contimod-style specs such as:
        [{"block_sizes": [...]}, {"block_indices": [[...],[...]]}, ...]
    as well as the canonical form used internally:
        (("sizes",(…)), ("indices",((…), (…))), …)

    The returned object is hashable and can be passed as a JAX `static_argname`.
    """
    if block_specs is None:
        return None

    if isinstance(block_specs, dict):
        items: Sequence[Any] = (block_specs,)
    else:
        items = block_specs  # may raise below if not a sequence

    if not isinstance(items, (list, tuple)):
        raise TypeError(
            "block_specs must be a sequence of specs or a single spec dict; "
            f"got {type(block_specs)}."
        )

    normalized: list[BlockSpec] = []
    for spec in items:
        normalized.append(_normalize_one_spec(spec))
    return tuple(normalized)


def _normalize_one_spec(spec: Any) -> BlockSpec:
    if spec is None:
        raise TypeError("block_specs entries must be non-None.")

    # Canonical form.
    if (
        isinstance(spec, tuple)
        and len(spec) == 2
        and isinstance(spec[0], str)
        and spec[0].strip().lower() in ("sizes", "indices")
    ):
        kind = spec[0].strip().lower()
        data = spec[1]
        if kind == "sizes":
            return ("sizes", tuple(int(x) for x in _as_sequence(data)))
        return ("indices", tuple(tuple(int(i) for i in _as_sequence(b)) for b in _as_sequence(data)))

    # contimod-style dict specs: {"block_sizes": ...} or {"block_indices": ...}
    if isinstance(spec, dict):
        if "block_sizes" in spec:
            return ("sizes", tuple(int(x) for x in _as_sequence(spec["block_sizes"])))
        if "block_indices" in spec:
            return (
                "indices",
                tuple(tuple(int(i) for i in _as_sequence(b)) for b in _as_sequence(spec["block_indices"])),
            )
        raise TypeError("Spec dict must contain 'block_sizes' or 'block_indices'.")

    # Bare sequences: treat as sizes if it is a flat int sequence; otherwise as indices.
    seq = _as_sequence(spec)
    if len(seq) == 0:
        raise TypeError("Spec sequences must be non-empty.")
    if all(isinstance(x, (int, np.integer)) for x in seq):
        return ("sizes", tuple(int(x) for x in seq))
    return ("indices", tuple(tuple(int(i) for i in _as_sequence(b)) for b in seq))


def _as_sequence(obj: Any) -> Sequence[Any]:
    if isinstance(obj, (list, tuple)):
        return obj
    if hasattr(obj, "tolist"):
        out = obj.tolist()
        if isinstance(out, list):
            return out
    raise TypeError(f"Expected a sequence; got {type(obj)}.")


def eigh(
    array: jax.Array,
    *,
    block_specs: Any | None = None,
    block_sizes: Any | None = None,
    block_indices: Any | None = None,
    sort: bool = True,
    check_offdiag: bool | None = None,
    offdiag_atol: float = 1e-12,
    offdiag_rtol: float = 0.0,
    **kwargs,
) -> tuple[jax.Array, jax.Array]:
    """Eigen-decomposition of a Hermitian matrix with optional block structure.

    This is a JAX-compatible variant of contimod's `utils.linalg.eigh`, designed
    for use under `jax.jit`. When block structure is requested, we compute a
    cheap off-diagonal block metric and *fall back* to full diagonalization if
    the matrix violates the requested structure.
    """
    array = jnp.asarray(array)

    if block_specs is not None:
        specs = normalize_block_specs(block_specs)
        if specs:
            check = bool(check_offdiag) if check_offdiag is not None else True
            return _eigh_block_specs(
                array,
                specs,
                sort=bool(sort),
                check_offdiag=check,
                offdiag_atol=float(offdiag_atol),
                offdiag_rtol=float(offdiag_rtol),
                **kwargs,
            )
        w, v = jnp.linalg.eigh(array, **kwargs)
        return w, v

    if block_sizes is not None:
        spec = normalize_block_specs(({"block_sizes": block_sizes},))
        check = bool(check_offdiag) if check_offdiag is not None else True
        return _eigh_block_specs(
            array,
            spec or (),
            sort=bool(sort),
            check_offdiag=check,
            offdiag_atol=float(offdiag_atol),
            offdiag_rtol=float(offdiag_rtol),
            **kwargs,
        )

    if block_indices is not None:
        spec = normalize_block_specs(({"block_indices": block_indices},))
        check = bool(check_offdiag) if check_offdiag is not None else True
        return _eigh_block_specs(
            array,
            spec or (),
            sort=bool(sort),
            check_offdiag=check,
            offdiag_atol=float(offdiag_atol),
            offdiag_rtol=float(offdiag_rtol),
            **kwargs,
        )

    w, v = jnp.linalg.eigh(array, **kwargs)
    return w, v


def _validate_block_sizes(block_sizes: tuple[int, ...], n: int) -> tuple[int, ...]:
    sizes = tuple(int(s) for s in block_sizes)
    if any(s <= 0 for s in sizes):
        raise ValueError("block_sizes entries must be positive integers.")
    if sum(sizes) != int(n):
        raise ValueError(f"block_sizes must sum to {int(n)} (got {sum(sizes)}).")
    return sizes


def _validate_block_indices(block_indices: tuple[tuple[int, ...], ...], n: int) -> tuple[tuple[int, ...], ...]:
    blocks = tuple(tuple(int(i) for i in b) for b in block_indices)
    used: set[int] = set()
    for b in blocks:
        if len(b) == 0:
            raise ValueError("block_indices entries must be non-empty.")
        for i in b:
            if i < 0 or i >= int(n):
                raise ValueError(f"block_indices contains index {i} outside [0, {int(n) - 1}].")
            if i in used:
                raise ValueError("block_indices must form a disjoint partition of the basis.")
            used.add(int(i))
    if len(used) != int(n):
        raise ValueError(f"block_indices must cover all basis indices 0..{int(n) - 1}.")
    return blocks


def _block_slices(block_sizes: tuple[int, ...]) -> tuple[slice, ...]:
    start = 0
    out: list[slice] = []
    for size in block_sizes:
        stop = start + int(size)
        out.append(slice(start, stop))
        start = stop
    return tuple(out)


def _mask_from_block_sizes(block_sizes: tuple[int, ...], n: int) -> jax.Array:
    mask = np.ones((int(n), int(n)), dtype=bool)
    for s in _block_slices(block_sizes):
        mask[s, s] = False
    return jnp.asarray(mask)


def _mask_from_block_indices(block_indices: tuple[tuple[int, ...], ...], n: int) -> jax.Array:
    mask = np.ones((int(n), int(n)), dtype=bool)
    for idx in block_indices:
        idx_arr = np.asarray(idx, dtype=int)
        mask[np.ix_(idx_arr, idx_arr)] = False
    return jnp.asarray(mask)


def _eigh_block_sizes(
    array: jax.Array, block_sizes: tuple[int, ...], *, sort: bool, **kwargs
) -> tuple[jax.Array, jax.Array]:
    n = int(array.shape[-1])
    sizes = _validate_block_sizes(block_sizes, n)
    blocks = _block_slices(sizes)

    eigenvals = []
    eigenvecs = []
    for s in blocks:
        w, v = jnp.linalg.eigh(array[..., s, s], **kwargs)
        eigenvals.append(w)
        eigenvecs.append(v)

    w_full = jnp.concatenate(eigenvals, axis=-1)

    v_full = jnp.zeros(array.shape, dtype=array.dtype)
    col_start = 0
    for s, v in zip(blocks, eigenvecs):
        size = int(s.stop - s.start)
        v_full = v_full.at[..., s, col_start : col_start + size].set(v)
        col_start += size

    if sort:
        idx = jnp.argsort(w_full, axis=-1)
        w_full = jnp.take_along_axis(w_full, idx, axis=-1)
        v_full = jnp.take_along_axis(v_full, idx[..., None, :], axis=-1)

    return w_full, v_full


def _eigh_block_indices(
    array: jax.Array, block_indices: tuple[tuple[int, ...], ...], *, sort: bool, **kwargs
) -> tuple[jax.Array, jax.Array]:
    n = int(array.shape[-1])
    blocks = _validate_block_indices(block_indices, n)

    eigenvals = []
    eigenvecs = []
    for idx in blocks:
        idx_j = jnp.asarray(idx, dtype=jnp.int32)
        sub = jnp.take(jnp.take(array, idx_j, axis=-2), idx_j, axis=-1)
        w, v = jnp.linalg.eigh(sub, **kwargs)
        eigenvals.append(w)
        eigenvecs.append(v)

    w_full = jnp.concatenate(eigenvals, axis=-1)

    v_full = jnp.zeros(array.shape, dtype=array.dtype)
    col_start = 0
    for idx, v in zip(blocks, eigenvecs):
        size = int(len(idx))
        idx_j = jnp.asarray(idx, dtype=jnp.int32)
        v_full = v_full.at[..., idx_j, col_start : col_start + size].set(v)
        col_start += size

    if sort:
        order = jnp.argsort(w_full, axis=-1)
        w_full = jnp.take_along_axis(w_full, order, axis=-1)
        v_full = jnp.take_along_axis(v_full, order[..., None, :], axis=-1)

    return w_full, v_full


def _eigh_block_specs(
    array: jax.Array,
    block_specs: tuple[BlockSpec, ...],
    *,
    sort: bool,
    check_offdiag: bool,
    offdiag_atol: float,
    offdiag_rtol: float,
    **kwargs,
) -> tuple[jax.Array, jax.Array]:
    n = int(array.shape[-1])

    if check_offdiag:
        abs_array = jnp.abs(array)
        scale = jnp.max(abs_array)
        tol = jnp.asarray(offdiag_atol, dtype=abs_array.dtype) + jnp.asarray(offdiag_rtol, dtype=abs_array.dtype) * scale

    def do_full(_):
        w, v = jnp.linalg.eigh(array, **kwargs)
        return w, v

    fn = do_full

    # Build a nested cond chain that picks the first compatible spec, without
    # performing redundant diagonalizations.
    for kind, data in reversed(block_specs):
        kind = str(kind).strip().lower()
        if kind == "sizes":
            sizes = _validate_block_sizes(tuple(int(x) for x in data), n)
            mask = _mask_from_block_sizes(sizes, n)
            ok = (
                (jnp.max(abs_array * mask) <= tol)
                if check_offdiag
                else jnp.array(True)
            )

            def do_block(_, *, _sizes=sizes):
                return _eigh_block_sizes(array, _sizes, sort=sort, **kwargs)

        elif kind == "indices":
            blocks = _validate_block_indices(
                tuple(tuple(int(i) for i in b) for b in data),
                n,
            )
            mask = _mask_from_block_indices(blocks, n)
            ok = (
                (jnp.max(abs_array * mask) <= tol)
                if check_offdiag
                else jnp.array(True)
            )

            def do_block(_, *, _blocks=blocks):
                return _eigh_block_indices(array, _blocks, sort=sort, **kwargs)

        else:
            raise ValueError("block_specs kind must be 'sizes' or 'indices'.")

        prev_fn = fn
        fn = (lambda _prev, _ok, _do: (lambda _: lax.cond(_ok, _do, _prev, operand=None)))(prev_fn, ok, do_block)

    return fn(None)
