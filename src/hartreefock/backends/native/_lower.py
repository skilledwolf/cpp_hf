"""Lower a kernel into the argument dict of the C++ ``HFKernel``."""

from __future__ import annotations

import numpy as np


def kernel_args(kernel) -> dict:
    """Pack a HartreeFockKernel into the dict shape that the C++ binding expects.

    Also packs the optional superlattice layout fields when the kernel exposes
    them (see :class:`SuperlatticeHartreeFockKernel`).
    The native solver then dispatches the streaming Fock + full Hartree paths
    inside ``build_fock_compact``.
    """
    has_refP = bool(kernel.has_reference_density)
    refP = kernel.refP if has_refP else np.empty((0,), dtype=np.complex128)
    args = dict(
        h=np.ascontiguousarray(kernel.h, dtype=np.complex128),
        VR=np.ascontiguousarray(kernel.VR_shifted, dtype=np.complex128),
        refP=np.ascontiguousarray(refP, dtype=np.complex128),
        has_refP=has_refP,
        w2d=np.ascontiguousarray(kernel.w2d, dtype=np.float64),
        HH=np.ascontiguousarray(kernel.HH, dtype=np.float64),
        contact_g=np.ascontiguousarray(kernel.contact_g, dtype=np.float64),
        contact_Oi=np.ascontiguousarray(kernel.contact_Oi, dtype=np.complex128),
        contact_Oj=np.ascontiguousarray(kernel.contact_Oj, dtype=np.complex128),
        weight_sum=float(kernel.weight_sum),
        T=float(kernel.T),
        include_hartree=bool(kernel.include_hartree),
        include_exchange=bool(kernel.include_exchange),
        exchange_hcp=bool(kernel.exchange_hermitian_channel_packing),
    )
    if getattr(kernel, "superlattice_fock_active", False) \
            or getattr(kernel, "superlattice_hartree_active", False):
        layout = kernel.layout
        args.update(dict(
            superlattice_fock_active=bool(kernel.superlattice_fock_active),
            superlattice_hartree_active=bool(kernel.superlattice_hartree_active),
            hartree_degeneracy=float(kernel.hartree_degeneracy),
            n_G=int(kernel.n_G),
            dim_orb=int(kernel.dim_orb),
            n_delta=int(layout.n_delta),
            N_ext_x=int(layout.N_ext_x),
            N_ext_y=int(layout.N_ext_y),
            V_lag_fft=np.ascontiguousarray(layout.V_lag_fft, dtype=np.complex128),
            g_a_off=np.ascontiguousarray(layout.g_a_off, dtype=np.int64),
            pair_i=np.ascontiguousarray(layout.delta_pair_i, dtype=np.int64),
            pair_j=np.ascontiguousarray(layout.delta_pair_j, dtype=np.int64),
            pair_start=np.ascontiguousarray(layout.delta_pair_start, dtype=np.int64),
            pair_to_delta=np.ascontiguousarray(layout.pair_to_delta, dtype=np.int64),
            HH_GG=np.ascontiguousarray(kernel.HH_GG, dtype=np.float64),
        ))
        HH_orb = getattr(kernel, "HH_GG_orbital", None)
        if HH_orb is not None:
            args["HH_GG_orbital"] = np.ascontiguousarray(
                np.asarray(HH_orb), dtype=np.float64,
            )
        V_orb = getattr(kernel, "V_lag_fft_orbital", None)
        if V_orb is not None:
            args["V_lag_fft_orbital"] = np.ascontiguousarray(
                np.asarray(V_orb), dtype=np.complex128,
            )
    return args
