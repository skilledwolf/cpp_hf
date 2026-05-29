"""Internal helpers shared across the Python wrapper layer.

The C++ extension ``cpp_hf._native`` (built by CMake from ``cpp/``) provides
the algorithm core: FFT-based exchange, batched eigh, the SCF and direct-
minimisation solver loops.  The package is C++-first — importing without the
extension built leaves :data:`HAVE_NATIVE` ``False`` and any solver call
raises a clear :class:`RuntimeError` via :func:`native_required`.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional native accelerator
    from . import _native  # type: ignore[attr-defined]

    HAVE_NATIVE = True
except Exception:  # pragma: no cover - native build optional at import time
    _native = None  # type: ignore[assignment]
    HAVE_NATIVE = False


def native_required() -> None:
    """Raise if the C++ extension is not loaded."""
    if not HAVE_NATIVE:
        raise RuntimeError(
            "The cpp_hf C++ extension (cpp_hf._native) is not available. "
            "Build it via 'pip install -e .' or 'pip install cpp_hf'."
        )


def hermitize(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))


def expit(x):
    """Numerically stable sigmoid; mirrors ``jax.scipy.special.expit``."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.result_type(x.dtype, np.float64))
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def batched_eigh(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched Hermitian eigh via the C++ extension."""
    native_required()
    F = np.asarray(F)
    Farr = np.ascontiguousarray(F, dtype=np.complex128)
    w, v = _native.eigh_batched(Farr)
    return (
        w.astype(F.real.dtype, copy=False),
        v.astype(F.dtype, copy=False),
    )


def fftn2d(x: np.ndarray) -> np.ndarray:
    return np.fft.fftn(x, axes=(0, 1))


def ifftn2d(x: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(x, axes=(0, 1))


def ifftshift2d(x: np.ndarray) -> np.ndarray:
    return np.fft.ifftshift(x, axes=(0, 1))
