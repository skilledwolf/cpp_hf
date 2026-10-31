"""Internal helpers shared across the Python wrapper layer.

The C++ extension ``hartreefock.backends.native._native`` (built by CMake from ``cpp/``) provides
the algorithm core: FFT-based exchange, batched eigh, the SCF and direct-
minimisation solver loops.  The package is C++-first — importing without the
extension built leaves :data:`HAVE_NATIVE` ``False`` and any solver call
raises a clear :class:`RuntimeError` via :func:`native_required`.
"""

from __future__ import annotations

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
            "The native C++ extension (hartreefock.backends.native._native) is not available. "
            "Build it via 'pip install -e .' or 'pip install hartreefock'."
        )
