"""Linear algebra helpers (block-diagonal eigh, block-spec normalization).

Mirrors ``jax_hf.linalg``.  Per-slice Hermitian eigh is delegated to the C++
extension (``cpp_hf._native.eigh_batched``) via :func:`cpp_hf._compat.batched_eigh`;
the block-spec dispatch (off-block-metric check, fall-back to full eigh) is
plain Python.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._compat import batched_eigh

BlockSpec = tuple[str, tuple[Any, ...]]


def normalize_block_specs(block_specs: Any) -> tuple[BlockSpec, ...] | None:
    if block_specs is None:
        return None

    if isinstance(block_specs, dict):
        items: Sequence[Any] = (block_specs,)
    else:
        items = block_specs

    if not isinstance(items, (list, tuple)):
        raise TypeError(
            "block_specs must be a sequence of specs or a single spec dict; "
            f"got {type(block_specs)}."
        )

    return tuple(_normalize_one_spec(spec) for spec in items)


def _normalize_one_spec(spec: Any) -> BlockSpec:
    if spec is None:
        raise TypeError("block_specs entries must be non-None.")

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
        return (
            "indices",
            tuple(tuple(int(i) for i in _as_sequence(b)) for b in _as_sequence(data)),
        )

    if isinstance(spec, dict):
        if "block_sizes" in spec:
            return ("sizes", tuple(int(x) for x in _as_sequence(spec["block_sizes"])))
        if "block_indices" in spec:
            return (
                "indices",
                tuple(
                    tuple(int(i) for i in _as_sequence(b))
                    for b in _as_sequence(spec["block_indices"])
                ),
            )
        raise TypeError("Spec dict must contain 'block_sizes' or 'block_indices'.")

    seq = _as_sequence(spec)
    if len(seq) == 0:
        raise TypeError("Spec sequences must be non-empty.")
    if all(isinstance(x, (int, np.integer)) for x in seq):
        return ("sizes", tuple(int(x) for x in seq))
    return (
        "indices",
        tuple(tuple(int(i) for i in _as_sequence(b)) for b in seq),
    )


def _as_sequence(obj: Any) -> Sequence[Any]:
    if isinstance(obj, (list, tuple)):
        return obj
    if hasattr(obj, "tolist"):
        out = obj.tolist()
        if isinstance(out, list):
            return out
    raise TypeError(f"Expected a sequence; got {type(obj)}.")


def _validate_block_sizes(block_sizes: tuple[int, ...], n: int) -> tuple[int, ...]:
    sizes = tuple(int(s) for s in block_sizes)
    if any(s <= 0 for s in sizes):
        raise ValueError("block_sizes entries must be positive integers.")
    if sum(sizes) != int(n):
        raise ValueError(f"block_sizes must sum to {int(n)} (got {sum(sizes)}).")
    return sizes


def _validate_block_indices(
    block_indices: tuple[tuple[int, ...], ...], n: int
) -> tuple[tuple[int, ...], ...]:
    blocks = tuple(tuple(int(i) for i in b) for b in block_indices)
    used: set[int] = set()
    for b in blocks:
        if len(b) == 0:
            raise ValueError("block_indices entries must be non-empty.")
        for i in b:
            if i < 0 or i >= int(n):
                raise ValueError(
                    f"block_indices contains index {i} outside [0, {int(n) - 1}]."
                )
            if i in used:
                raise ValueError(
                    "block_indices must form a disjoint partition of the basis."
                )
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


def _mask_from_block_sizes(block_sizes: tuple[int, ...], n: int) -> np.ndarray:
    mask = np.ones((int(n), int(n)), dtype=bool)
    for s in _block_slices(block_sizes):
        mask[s, s] = False
    return mask


def _mask_from_block_indices(
    block_indices: tuple[tuple[int, ...], ...], n: int
) -> np.ndarray:
    mask = np.ones((int(n), int(n)), dtype=bool)
    for idx in block_indices:
        idx_arr = np.asarray(idx, dtype=int)
        mask[np.ix_(idx_arr, idx_arr)] = False
    return mask


def _eigh_block_sizes(
    array: np.ndarray, block_sizes: tuple[int, ...], *, sort: bool
) -> tuple[np.ndarray, np.ndarray]:
    n = int(array.shape[-1])
    sizes = _validate_block_sizes(block_sizes, n)
    blocks = _block_slices(sizes)

    eigenvals = []
    eigenvecs = []
    for s in blocks:
        w, v = batched_eigh(array[..., s, s])
        eigenvals.append(w)
        eigenvecs.append(v)

    w_full = np.concatenate(eigenvals, axis=-1)
    v_full = np.zeros(array.shape, dtype=array.dtype)
    col_start = 0
    for s, v in zip(blocks, eigenvecs):
        size = int(s.stop - s.start)
        v_full[..., s, col_start : col_start + size] = v
        col_start += size

    if sort:
        idx = np.argsort(w_full, axis=-1)
        w_full = np.take_along_axis(w_full, idx, axis=-1)
        v_full = np.take_along_axis(v_full, idx[..., None, :], axis=-1)

    return w_full, v_full


def _eigh_block_indices(
    array: np.ndarray, block_indices: tuple[tuple[int, ...], ...], *, sort: bool
) -> tuple[np.ndarray, np.ndarray]:
    n = int(array.shape[-1])
    blocks = _validate_block_indices(block_indices, n)

    eigenvals = []
    eigenvecs = []
    for idx in blocks:
        idx_arr = np.asarray(idx, dtype=np.intp)
        sub = array[..., idx_arr[:, None], idx_arr[None, :]]
        w, v = batched_eigh(sub)
        eigenvals.append(w)
        eigenvecs.append(v)

    w_full = np.concatenate(eigenvals, axis=-1)
    v_full = np.zeros(array.shape, dtype=array.dtype)
    col_start = 0
    for idx, v in zip(blocks, eigenvecs):
        size = int(len(idx))
        idx_arr = np.asarray(idx, dtype=np.intp)
        v_full[..., idx_arr[:, None], col_start : col_start + size] = v
        col_start += size

    if sort:
        order = np.argsort(w_full, axis=-1)
        w_full = np.take_along_axis(w_full, order, axis=-1)
        v_full = np.take_along_axis(v_full, order[..., None, :], axis=-1)

    return w_full, v_full


def _eigh_block_specs(
    array: np.ndarray,
    block_specs: tuple[BlockSpec, ...],
    *,
    sort: bool,
    check_offdiag: bool,
    offdiag_atol: float,
    offdiag_rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(array.shape[-1])

    if check_offdiag:
        abs_array = np.abs(array)
        scale = np.max(abs_array)
        tol = float(offdiag_atol) + float(offdiag_rtol) * float(scale)

    for kind, data in block_specs:
        kind = str(kind).strip().lower()
        if kind == "sizes":
            sizes = _validate_block_sizes(tuple(int(x) for x in data), n)
            mask = _mask_from_block_sizes(sizes, n)
            ok = (not check_offdiag) or (float(np.max(abs_array * mask)) <= tol)
            if ok:
                return _eigh_block_sizes(array, sizes, sort=sort)
        elif kind == "indices":
            blocks = _validate_block_indices(
                tuple(tuple(int(i) for i in b) for b in data), n
            )
            mask = _mask_from_block_indices(blocks, n)
            ok = (not check_offdiag) or (float(np.max(abs_array * mask)) <= tol)
            if ok:
                return _eigh_block_indices(array, blocks, sort=sort)
        else:
            raise ValueError("block_specs kind must be 'sizes' or 'indices'.")

    return batched_eigh(array)


def eigh(
    array: np.ndarray,
    *,
    block_specs: Any | None = None,
    block_sizes: Any | None = None,
    block_indices: Any | None = None,
    sort: bool = True,
    check_offdiag: bool | None = None,
    offdiag_atol: float = 1e-12,
    offdiag_rtol: float = 0.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian eigendecomposition with optional block-diagonal acceleration.

    A numpy/scipy variant of ``jax_hf.linalg.eigh``.  When ``block_specs`` is
    provided we cheaply check the off-block magnitude and dispatch a reduced
    block solve when the matrix is compatible.
    """
    array = np.asarray(array)

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
            )
        return batched_eigh(array)

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
        )

    return batched_eigh(array)
