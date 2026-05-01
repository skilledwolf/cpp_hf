"""Hartree-Fock kernel: precomputed arrays for the solver loop.

Numpy port of ``jax_hf.problem``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._compat import fftn2d


class HartreeFockKernel:
    """Precomputed arrays for the HF solver loop.

    Mirrors :class:`jax_hf.HartreeFockKernel` exactly: holds the FFT of the
    interaction kernel (with the ifftshift phase pre-absorbed into VR), the
    Hartree matrix, weights, and the single-particle Hamiltonian.
    """

    def __init__(
        self,
        weights,
        hamiltonian,
        coulomb_q,
        T: float,
        include_hartree: bool = False,
        include_exchange: bool = True,
        reference_density=None,
        hartree_matrix=None,
        contact_terms=None,
    ):
        # We work in double precision throughout (complex128 / float64).
        h = np.ascontiguousarray(np.asarray(hamiltonian), dtype=np.complex128)
        self.h = h
        real_dtype = np.float64
        w2d = np.ascontiguousarray(np.asarray(weights), dtype=real_dtype)
        self.weights_b = w2d[..., None, None]
        self.weight_sum = np.sum(w2d)
        self.w2d = w2d

        Vq = np.asarray(coulomb_q)
        if np.iscomplexobj(Vq):
            imag_max = float(np.max(np.abs(np.imag(Vq))))
            if imag_max <= 1e-8:
                Vq = np.real(Vq)
        else:
            Vq = Vq.astype(real_dtype)
        self.exchange_hermitian_channel_packing = bool(
            Vq.shape[-2:] == (1, 1) and not np.iscomplexobj(Vq)
        )
        self.VR = fftn2d(self.weights_b * Vq.astype(h.dtype))

        n0, n1 = int(h.shape[0]), int(h.shape[1])
        s0, s1 = n0 // 2, n1 // 2
        phase0 = np.exp(2j * np.pi * np.arange(n0) * s0 / n0)
        phase1 = np.exp(2j * np.pi * np.arange(n1) * s1 / n1)
        shift_phase = (phase0[:, None] * phase1[None, :])[..., None, None].astype(self.VR.dtype)
        self._VR_shifted = self.VR * shift_phase

        self.T = float(T)
        self.include_hartree = bool(include_hartree)
        self.include_exchange = bool(include_exchange)
        if (not self.include_hartree) and (not self.include_exchange):
            raise ValueError(
                "HartreeFockKernel must include at least one of Hartree or exchange."
            )

        if reference_density is not None:
            ref = np.asarray(reference_density, dtype=h.dtype)
            if ref.shape != h.shape:
                raise ValueError(
                    f"reference_density must have shape {h.shape}, got {ref.shape}"
                )
            self.refP = ref
        else:
            self.refP = np.zeros_like(h)

        if self.include_hartree:
            if hartree_matrix is None:
                raise ValueError(
                    "include_hartree=True requires hartree_matrix to be provided"
                )
            if reference_density is None:
                raise ValueError(
                    "include_hartree=True requires reference_density to be provided"
                )
            HH = np.asarray(hartree_matrix, dtype=real_dtype)
            if HH.shape != h.shape[-2:]:
                raise ValueError(
                    f"hartree_matrix must have shape {h.shape[-2:]}, got {HH.shape}"
                )
            self.HH = HH
        else:
            self.HH = np.zeros(h.shape[-2:], dtype=real_dtype)

        n_orb = int(h.shape[-1])
        if contact_terms is None or len(contact_terms) == 0:
            self.contact_g = np.zeros((1,), dtype=real_dtype)
            self.contact_Oi = np.zeros((1, n_orb, n_orb), dtype=h.dtype)
            self.contact_Oj = np.zeros((1, n_orb, n_orb), dtype=h.dtype)
        else:
            gs, Ois, Ojs = [], [], []
            for k, term in enumerate(contact_terms):
                if len(term) != 3:
                    raise ValueError(
                        f"contact_terms[{k}] must be (g, O_i, O_j); got len={len(term)}"
                    )
                g_t, Oi_t, Oj_t = term
                Oi_arr = np.asarray(Oi_t, dtype=h.dtype)
                Oj_arr = np.asarray(Oj_t, dtype=h.dtype)
                if Oi_arr.shape != (n_orb, n_orb):
                    raise ValueError(
                        f"contact_terms[{k}].O_i must have shape {(n_orb, n_orb)}, "
                        f"got {Oi_arr.shape}"
                    )
                if Oj_arr.shape != (n_orb, n_orb):
                    raise ValueError(
                        f"contact_terms[{k}].O_j must have shape {(n_orb, n_orb)}, "
                        f"got {Oj_arr.shape}"
                    )
                gs.append(np.asarray(g_t, dtype=real_dtype))
                Ois.append(Oi_arr)
                Ojs.append(Oj_arr)
            self.contact_g = np.stack(gs)
            self.contact_Oi = np.stack(Ois)
            self.contact_Oj = np.stack(Ojs)

    def as_args(self) -> dict:
        """Dynamic inputs for the solver functions (matches jax_hf signature)."""
        return dict(
            h=self.h,
            weights_b=self.weights_b,
            weight_sum=self.weight_sum,
            VR=self._VR_shifted,
            T=self.T,
            refP=self.refP,
            HH=self.HH,
            include_hartree=self.include_hartree,
            include_exchange=self.include_exchange,
            exchange_hermitian_channel_packing=self.exchange_hermitian_channel_packing,
            contact_g=self.contact_g,
            contact_Oi=self.contact_Oi,
            contact_Oj=self.contact_Oj,
        )
