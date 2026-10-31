"""Fock build and HF energy through the C++ ``build_fock``."""

from __future__ import annotations

import numpy as np

from ._compat import _native, native_required
from ._lower import kernel_args


def build_fock(kernel, P, *, project_fn=None):
    """Return ``(Sigma, H, F, E)`` for density ``P``.

    ``F = project(hermitize(h + Sigma + H))`` and ``E`` is the HF energy
    ``Σ_k w_k Tr[h P] + ½ Σ_k w_k Tr[(Sigma + H)(P - refP)]``.
    """
    native_required()
    P = np.ascontiguousarray(P, dtype=np.complex128)
    return _native.build_fock_apply(kernel_args(kernel), P, project_fn)
