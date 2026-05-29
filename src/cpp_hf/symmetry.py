"""Generic symmetry projectors (numpy port of ``jax_hf.symmetry``).

``make_project_fn`` builds a callable that averages a density / Fock matrix
over unitary, spatial, and time-reversal symmetries.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ProjectFn = Callable[[np.ndarray], np.ndarray]

__all__ = ["ProjectFn", "make_project_fn"]


def _flip_k(
    A: np.ndarray,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
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


def _sum_unitary_conj(A: np.ndarray, G: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(A)
    for i in range(G.shape[0]):
        g = G[i]
        gH = np.conj(np.swapaxes(g, -1, -2))
        acc += (g @ A) @ gH
    return acc


def _avg_unitary_conj(A: np.ndarray, G: np.ndarray) -> np.ndarray:
    ng = G.shape[0]
    return _sum_unitary_conj(A, G) / np.asarray(float(ng), dtype=A.dtype)


def _avg_combined_group(
    A: np.ndarray,
    G_same: np.ndarray,
    G_flip: np.ndarray,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    acc = _sum_unitary_conj(A, G_same)
    A_neg = _flip_k(A, k_convention, flip_axes)
    acc += _sum_unitary_conj(A_neg, G_flip)
    N = G_same.shape[0] + G_flip.shape[0]
    return acc / np.asarray(float(N), dtype=A.dtype)


def _avg_time_reversal(
    A: np.ndarray,
    U: np.ndarray,
    k_convention: str,
    flip_axes: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    UH = np.conj(np.swapaxes(U, -1, -2))
    A_neg = _flip_k(A, k_convention, flip_axes)
    A_tr = U @ np.conj(A_neg) @ UH
    return 0.5 * (A + A_tr)


def make_project_fn(
    *,
    unitary_group: np.ndarray | None = None,
    spatial_group: np.ndarray | None = None,
    spatial_k_convention: str = "mod",
    spatial_k_flip_axes: tuple[int, ...] = (0, 1),
    time_reversal_U: np.ndarray | None = None,
    time_reversal_k_convention: str = "mod",
    time_reversal_k_flip_axes: tuple[int, ...] = (0, 1),
) -> ProjectFn:
    """Build a symmetry-averaging projection function."""
    has_group = unitary_group is not None or spatial_group is not None
    if not has_group and time_reversal_U is None:
        return lambda A: A

    G = None if unitary_group is None else np.asarray(unitary_group)
    S = None if spatial_group is None else np.asarray(spatial_group)
    U = None if time_reversal_U is None else np.asarray(time_reversal_U)
    s_k_conv = str(spatial_k_convention)
    s_k_axes = tuple(spatial_k_flip_axes)
    t_k_conv = str(time_reversal_k_convention)
    t_k_axes = tuple(time_reversal_k_flip_axes)

    def project(A: np.ndarray) -> np.ndarray:
        out = np.asarray(A)
        if G is not None and S is not None:
            out = _avg_combined_group(out, G, S, s_k_conv, s_k_axes)
        elif G is not None:
            out = _avg_unitary_conj(out, G)
        elif S is not None:
            I_mat = np.eye(S.shape[-1], dtype=S.dtype)[None]
            out = _avg_combined_group(out, I_mat, S, s_k_conv, s_k_axes)
        if U is not None:
            out = _avg_time_reversal(out, U, t_k_conv, t_k_axes)
        return out

    return project
