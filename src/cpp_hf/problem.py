"""Hartree-Fock kernel: precomputed arrays for the solver loop.

Numpy port of ``jax_hf.problem``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._compat import fftn2d
from .utils import selfenergy_fft


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
        embed_reference_potential: bool = False,
        center_embedded_hartree: bool = True,
    ):
        """Construct the Hartree-Fock kernel.

        ``embed_reference_potential``: when True, precompute the reference
        Hartree-Fock potential ``V_HF[refP] = J[refP] + Σ[refP]`` (where
        Σ = −K is the kernel-convention sign for exchange) and add it to
        the bare ``hamiltonian`` once at construction time.  The kernel
        then runs reference-subtracted updates ``Σ[P−refP] + H[P−refP]``
        on this enlarged ``h_eff``, which equals
        ``h + V_HF[refP] + V_HF[P−refP] = h + V_HF[P]`` — i.e. the
        standard HF Fock matrix.  Without this flag, the kernel solves a
        *modified* HF where the gradient is ``h + V_HF[P−refP]``; the two
        problems have different minima by an amount that scales with
        the layer-screening contribution from refP.

        ``center_embedded_hartree``: when ``embed_reference_potential`` is
        also True, subtract the orbital-uniform mean of ``J[refP]`` from
        ``h_eff``.  This is a gauge choice that brings absolute energies
        to a sensible scale (E_F near 0 instead of near +V_H[refP].mean
        which can be very large for strong-Coulomb dual-gate setups).
        Does not affect band asymmetries or DOS at E_F.

        Setting ``embed_reference_potential=True`` requires
        ``reference_density`` to be provided.  Energy reported by
        :func:`cpp_hf.fock.hf_energy` is then equal to the standard HF
        energy plus a P-independent constant
        ``½ Tr[V_HF[refP]·refP]`` — phase comparisons (energy
        differences) are unaffected.
        """
        # We work in double precision throughout (complex128 / float64).
        h = np.ascontiguousarray(np.asarray(hamiltonian), dtype=np.complex128)
        self.h = h
        real_dtype = np.float64
        w2d = np.ascontiguousarray(np.asarray(weights), dtype=real_dtype)
        self.weights_b = w2d[..., None, None]
        self.weight_sum = np.sum(w2d)
        self.w2d = w2d

        self.T = float(T)
        self.include_hartree = bool(include_hartree)
        self.include_exchange = bool(include_exchange)
        if (not self.include_hartree) and (not self.include_exchange):
            raise ValueError(
                "HartreeFockKernel must include at least one of Hartree or exchange."
            )

        n0, n1 = int(h.shape[0]), int(h.shape[1])
        if self.include_exchange:
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
            VR = fftn2d(self.weights_b * Vq.astype(h.dtype, copy=False))

            s0, s1 = n0 // 2, n1 // 2
            phase0 = np.exp(2j * np.pi * np.arange(n0) * s0 / n0)
            phase1 = np.exp(2j * np.pi * np.arange(n1) * s1 / n1)
            shift_phase = (phase0[:, None] * phase1[None, :])[..., None, None].astype(
                VR.dtype
            )
            VR *= shift_phase
            self._VR_shifted = np.ascontiguousarray(VR)
        else:
            self.exchange_hermitian_channel_packing = False
            self._VR_shifted = np.zeros((n0, n1, 1, 1), dtype=h.dtype)

        if reference_density is not None:
            ref = np.ascontiguousarray(np.asarray(reference_density), dtype=h.dtype)
            if ref.shape != h.shape:
                raise ValueError(
                    f"reference_density must have shape {h.shape}, got {ref.shape}"
                )
            self._refP = ref
        else:
            self._refP = None
        self._empty_refP = np.empty((0,), dtype=h.dtype)
        self._zero_refP = np.broadcast_to(np.array(0, dtype=h.dtype), h.shape)

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

        # Embedded reference potential: bake V_HF[refP] = J[refP] + Σ[refP]
        # into h once so the kernel naturally solves standard HF.  See the
        # constructor docstring.
        self.embed_reference_potential = bool(embed_reference_potential)
        self.center_embedded_hartree = bool(center_embedded_hartree)
        self.embedded_energy_offset = 0.0
        if self.embed_reference_potential:
            if self._refP is None:
                raise ValueError(
                    "embed_reference_potential=True requires reference_density."
                )
            ref = self._refP
            n_orb = int(h.shape[-1])
            # Hartree contribution from refP:
            #   J_ref_diag[a] = Σ_b HH[a, b] · ⟨n_orbital_b⟩_BZ(refP)
            # is a per-orbital constant (k-independent).  In our (Σ = −K)
            # convention, the kernel's "H" is +J, so this enters h with
            # the same sign.
            H_ref_diag_addition = np.zeros(n_orb, dtype=real_dtype)
            const_J_refP_refP = 0.0
            if self.include_hartree:
                diag_R = np.einsum("ij,ijaa->a", w2d, ref).real
                J_ref_diag = self.HH @ diag_R
                if self.center_embedded_hartree:
                    J_ref_diag = J_ref_diag - float(np.mean(J_ref_diag))
                H_ref_diag_addition = J_ref_diag.astype(real_dtype, copy=False)
                # Tr[J[refP] · refP] = Σ_a J_diag[a] · diag_R[a]   (over orbitals)
                # Note: this Tr is BZ-integrated already because J is k-independent
                # and refP[k]'s diagonal is summed with weights into diag_R.
                const_J_refP_refP = float(np.sum(J_ref_diag * diag_R))
            # Exchange contribution from refP:
            #   Σ_ref[k] = (−K[refP])[k]  is a full (nk, nk, nb, nb) tensor
            const_Sigma_refP_refP = 0.0
            Sigma_ref = None
            if self.include_exchange:
                Sigma_ref = selfenergy_fft(
                    self._VR_shifted, ref,
                    _apply_ifftshift=False,
                    hermitian_channel_packing=self.exchange_hermitian_channel_packing,
                )
                Sigma_ref = np.ascontiguousarray(Sigma_ref, dtype=h.dtype)
                # Tr[Σ[refP] · refP] = Σ_k w_k Σ_ij Σ_ref[k]_ij refP[k]_ji
                const_Sigma_refP_refP = float(np.einsum(
                    "ij,ijab,ijba->", w2d, Sigma_ref, ref,
                ).real)
            # Build h_eff in place (overwrite self.h).
            h_eff = self.h.copy()
            if Sigma_ref is not None:
                h_eff += Sigma_ref
            for a in range(n_orb):
                h_eff[..., a, a] += H_ref_diag_addition[a]
            self.h = np.ascontiguousarray(h_eff, dtype=np.complex128)
            # Energy reported by hf_energy(h=h_eff, ...) equals
            # E_std + ½ Tr[V_HF[refP] refP].  We track the offset so callers
            # can recover E_std if desired.
            self.embedded_energy_offset = 0.5 * (
                const_J_refP_refP + const_Sigma_refP_refP
            )

    @property
    def has_reference_density(self) -> bool:
        return self._refP is not None

    @property
    def refP(self) -> np.ndarray:
        """Reference density, exposed as a zero broadcast view when absent."""
        if self._refP is None:
            return self._zero_refP
        return self._refP

    def as_args(self) -> dict:
        """Dynamic inputs for the solver functions (matches jax_hf signature)."""
        return dict(
            h=self.h,
            weights_b=self.weights_b,
            weight_sum=self.weight_sum,
            VR=self._VR_shifted,
            T=self.T,
            refP=self.refP,
            has_refP=self.has_reference_density,
            HH=self.HH,
            include_hartree=self.include_hartree,
            include_exchange=self.include_exchange,
            exchange_hermitian_channel_packing=self.exchange_hermitian_channel_packing,
            contact_g=self.contact_g,
            contact_Oi=self.contact_Oi,
            contact_Oj=self.contact_Oj,
        )
