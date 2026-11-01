"""Fock build and HF energy through the C++ ``build_fock``."""

from __future__ import annotations

from types import SimpleNamespace

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


def fock_operators(kernel):
    """``evaluate(P) -> (F, E)`` and the linear ``response(P, dP) -> dF``.

    The HF Fock is affine in P, so the response is a Fock build of ``dP``
    with ``h`` and the reference density removed (which also avoids
    subtractive cancellation).
    """
    native_required()
    args = kernel_args(kernel)
    response_args = dict(
        args, h=np.zeros_like(args["h"]), has_refP=False, refP=np.empty(0, dtype=np.complex128)
    )

    def evaluate(P):
        out = _native.build_fock_apply(args, np.ascontiguousarray(P, dtype=np.complex128), None)
        return np.asarray(out[2]), float(out[3])

    def response(P, dP):
        dP = np.ascontiguousarray(dP, dtype=np.complex128)
        return np.asarray(_native.build_fock_apply(response_args, dP, None)[2])

    return SimpleNamespace(evaluate=evaluate, response=response, dtype=np.dtype(np.complex128))


def superlattice_exchange(rho, layout, n_G: int, *, hermitian_rho: bool = False):
    """Streaming superlattice Fock ``Σ_F[rho]`` (see :mod:`hartreefock.superlattice`).

    ``rho`` has shape ``(nkx, nky, D, D)`` with ``D = n_G · dim_orb``.  The C++
    loop streams one ``ΔG`` channel at a time, so peak memory is
    ``O(N_ext² · dim_orb²)`` independent of the number of channels.
    """
    native_required()
    rho = np.ascontiguousarray(rho, dtype=np.complex128)
    nkx, nky, D, _ = rho.shape
    dim_orb = D // int(n_G)
    V_orb = layout.V_lag_fft_orbital
    sigma = _native.selfenergy_superlattice_streamed(
        rho=rho.reshape(nkx, nky, n_G, dim_orb, n_G, dim_orb),
        VR_fft=layout.V_lag_fft,
        g_a_off=layout.g_a_off,
        pair_i=layout.delta_pair_i,
        pair_j=layout.delta_pair_j,
        pair_start=layout.delta_pair_start,
        nkx=int(nkx), nky=int(nky), n_G=int(n_G), dim_orb=int(dim_orb),
        N_ext_x=int(layout.N_ext_x), N_ext_y=int(layout.N_ext_y),
        VR_fft_orbital=V_orb,
        hermitian_rho=bool(hermitian_rho),
    )
    return np.asarray(sigma).reshape(nkx, nky, D, D)
