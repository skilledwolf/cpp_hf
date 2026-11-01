"""Lower a kernel into the argument dict of the C++ ``HFKernel``."""

from __future__ import annotations

import numpy as np

from ...superlattice import SuperlatticeHartreeFockKernel


def _c(a, dtype):
    return np.ascontiguousarray(a, dtype=dtype)


def kernel_args(kernel) -> dict:
    """Pack a dense or superlattice kernel into the dict the C++ binding expects.

    For a :class:`SuperlatticeHartreeFockKernel` the ``superlattice_*`` flags
    route ``build_fock_compact`` to the streaming Fock and the ΔG Hartree.
    """
    a = kernel._as_args()
    has_refP = bool(kernel.has_reference_density)
    args = dict(
        h=_c(a["h"], np.complex128),
        VR=_c(a["VR"], np.complex128),
        refP=_c(a["refP"] if has_refP else np.empty(0), np.complex128),
        has_refP=has_refP,
        w2d=_c(kernel.w2d, np.float64),
        HH=_c(a["HH"], np.float64),
        contact_g=_c(a["contact_g"], np.float64),
        contact_Oi=_c(a["contact_Oi"], np.complex128),
        contact_Oj=_c(a["contact_Oj"], np.complex128),
        weight_sum=float(a["weight_sum"]),
        T=float(a["T"]),
        include_hartree=bool(a["include_hartree"]),
        include_exchange=bool(a["include_exchange"]),
        exchange_hcp=bool(a["exchange_hermitian_channel_packing"]),
    )
    if not isinstance(kernel, SuperlatticeHartreeFockKernel):
        return args
    layout = kernel.layout
    args.update(
        superlattice_fock_active=kernel.include_exchange,
        superlattice_hartree_active=kernel.include_hartree,
        hartree_degeneracy=kernel.hartree_degeneracy,
        n_G=kernel.n_G,
        dim_orb=kernel.dim_orb,
        n_delta=layout.n_delta,
        N_ext_x=layout.N_ext_x,
        N_ext_y=layout.N_ext_y,
        V_lag_fft=layout.V_lag_fft,
        g_a_off=layout.g_a_off,
        pair_i=layout.delta_pair_i,
        pair_j=layout.delta_pair_j,
        pair_start=layout.delta_pair_start,
        pair_to_delta=_c(layout.pair_to_delta, np.int64),
        HH_GG=kernel.HH_GG,
    )
    if kernel.HH_GG_orbital is not None:
        args["HH_GG_orbital"] = kernel.HH_GG_orbital
    if layout.V_lag_fft_orbital is not None:
        args["V_lag_fft_orbital"] = layout.V_lag_fft_orbital
    return args
