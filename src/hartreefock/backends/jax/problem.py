"""Hartree-Fock kernel: precomputed arrays for the JIT solver loop."""

from __future__ import annotations

from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from .fock import _contact_self_energy
from .utils import selfenergy_fft, validate_temperature


class HartreeFockKernel:
    """Precomputed arrays for the jitted solver loop.

    Absorbs the ifftshift phase into VR so the hot loop skips the data
    permutation.  All arrays are exposed via ``as_args()`` for passing
    as dynamic inputs to ``jax.jit``-compiled functions.
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
        """
        Parameters
        ----------
        contact_terms : sequence of (g, O_i, O_j) tuples, optional
            Additional flavor-bilinear *contact* (q-independent) interactions
            of the form ``H = (1/2) g · O_i ⊗ O_j``.  Each term contributes
            both Hartree and Fock channels via the BZ-averaged density:
                σ_H = g · O_i · tr(O_j ρ̄)         (k-independent)
                σ_F = -g · O_i · ρ̄ · O_j         (k-independent)
            where ρ̄ = Σ_k w_k (P_k - refP_k).  ``O_i`` and ``O_j`` are
            ``(n_orb, n_orb)`` operator matrices in flavor space.  Pass an
            asymmetric pair plus its h.c. partner to model a Hermitian term
            with O_i ≠ O_j (matches contimod's ``Shortrange`` convention,
            cf. ``get_shortrange_MF_valley`` in contimod.meanfield).
        embed_reference_potential : bool, default False
            When True, precompute the reference Hartree-Fock potential
            ``V_HF[refP]``, including Hartree, long-range exchange, and
            contact terms, and add it to the bare ``hamiltonian`` once at
            construction time.  The kernel then runs reference-subtracted
            updates on this enlarged ``h_eff``, giving
            ``F = h + V_HF[P]`` — i.e. the standard HF Fock matrix.
            Without this flag, the kernel solves a *modified* HF problem
            whose minimum differs from standard HF by the reference
            potential.  Recover standard HF energy using
            ``E_standard = E_reported - embedded_energy_offset
            + embedded_hartree_shift * n_electrons``.  The offset is the
            uncentered ``½ Tr[V_HF[refP] · refP]``; the shift is the removed
            Hartree mean, or zero without centering.  Here ``n_electrons``
            is ``sum_k weights[k] Tr[P[k]]``, not divided by weight_sum.
            Similarly, ``mu_standard = mu_reported + embedded_hartree_shift``.
            Phase energy differences at fixed particle number are unaffected.
        center_embedded_hartree : bool, default True
            When ``embed_reference_potential`` is True, subtract the
            orbital-uniform mean of ``J[refP]`` from ``h_eff``.  This
            is a gauge choice that brings absolute eigenvalues to a
            sensible scale (E_F near 0 instead of near +mean(J[refP])
            which can be huge for strong-Coulomb dual-gate setups).
            Does not affect band asymmetries or DOS at E_F.
        """
        validate_temperature(T, context="T")
        h = jnp.asarray(hamiltonian)
        self.h = h
        w2d = jnp.asarray(weights, dtype=h.real.dtype)
        weights_host = np.asarray(w2d)
        if weights_host.shape != h.shape[:2]:
            raise ValueError(f"weights must have shape {h.shape[:2]}, got {w2d.shape}")
        if not np.all(np.isfinite(weights_host)):
            raise ValueError("weights must be finite")
        if np.any(weights_host < 0):
            raise ValueError("weights must be non-negative")
        if float(weights_host.sum()) <= 0:
            raise ValueError("weights must have a positive sum")
        self.weights_b = w2d[..., None, None]
        self.weight_sum = jnp.sum(w2d)
        self.w2d = w2d

        Vq = jnp.asarray(coulomb_q)
        if jnp.iscomplexobj(Vq):
            imag_max = float(jnp.max(jnp.abs(jnp.imag(Vq))))
            if imag_max <= 1e-8:
                Vq = jnp.real(Vq)
        else:
            Vq = Vq.astype(h.real.dtype)
        self.exchange_hermitian_channel_packing = bool(
            Vq.shape[-2:] == (1, 1) and not jnp.iscomplexobj(Vq)
        )
        # The quadrature weight belongs to the integrated density P(q), not
        # the transfer-momentum kernel V(k-q).  build_fock applies w(q) before
        # transforming the density.
        self.VR = jnp.fft.fftn(jnp.asarray(Vq, dtype=h.dtype), axes=(0, 1))
        # Pre-absorb ifftshift phase into VR.
        n0, n1 = int(h.shape[0]), int(h.shape[1])
        s0, s1 = n0 // 2, n1 // 2
        phase0 = np.exp(2j * np.pi * np.arange(n0) * s0 / n0)
        phase1 = np.exp(2j * np.pi * np.arange(n1) * s1 / n1)
        shift_phase = jnp.asarray(
            (phase0[:, None] * phase1[None, :])[..., None, None],
            dtype=self.VR.dtype,
        )
        shifted = self.VR * shift_phase
        # A sampled radial V(q) need not obey *periodic* inversion on even
        # oblique grids. The quadratic energy uses only the self-adjoint
        # convolution; retaining its skew part in F breaks dE/dP = F.
        # Convolution's adjoint multiplier is shifted.conj(). This preserves
        # the scalar quadratic energy at fixed h and reference density.
        shifted_host = np.asarray(shifted)
        self.exchange_reciprocity_defect = float(
            np.linalg.norm(shifted_host.imag)
            / max(np.linalg.norm(shifted_host), np.finfo(float).tiny)
        )
        self._VR_shifted = shifted.real.astype(shifted.dtype)
        # Keep the public unshifted representation consistent with as_args().
        self.VR = self._VR_shifted / shift_phase
        self.T = float(T)
        self.include_hartree = bool(include_hartree)
        self.include_exchange = bool(include_exchange)
        if (not self.include_hartree) and (not self.include_exchange):
            raise ValueError("HartreeFockKernel must include at least one of Hartree or exchange.")

        if reference_density is not None:
            ref = jnp.asarray(reference_density, dtype=h.dtype)
            if ref.shape != h.shape:
                raise ValueError(f"reference_density must have shape {h.shape}, got {ref.shape}")
            self.refP = ref
        else:
            self.refP = jnp.zeros_like(h)

        if self.include_hartree:
            if hartree_matrix is None:
                raise ValueError("include_hartree=True requires hartree_matrix to be provided")
            if reference_density is None:
                raise ValueError("include_hartree=True requires reference_density to be provided")
            HH = jnp.asarray(hartree_matrix, dtype=h.real.dtype)
            if HH.shape != h.shape[-2:]:
                raise ValueError(f"hartree_matrix must have shape {h.shape[-2:]}, got {HH.shape}")
            self.HH = HH
        else:
            self.HH = jnp.zeros(h.shape[-2:], dtype=h.real.dtype)

        # Contact (q-independent) flavor-bilinear interactions. Always at
        # least one term in the stack so that JIT shapes are static; if
        # the user passes nothing, we install a zero-coupling dummy.
        n_orb = int(h.shape[-1])
        if contact_terms is None or len(contact_terms) == 0:
            self.contact_g  = jnp.zeros((1,), dtype=h.real.dtype)
            self.contact_Oi = jnp.zeros((1, n_orb, n_orb), dtype=h.dtype)
            self.contact_Oj = jnp.zeros((1, n_orb, n_orb), dtype=h.dtype)
        else:
            gs, Ois, Ojs = [], [], []
            for k, term in enumerate(contact_terms):
                if len(term) != 3:
                    raise ValueError(
                        f"contact_terms[{k}] must be (g, O_i, O_j); got len={len(term)}"
                    )
                g_t, Oi_t, Oj_t = term
                Oi_arr = jnp.asarray(Oi_t, dtype=h.dtype)
                Oj_arr = jnp.asarray(Oj_t, dtype=h.dtype)
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
                gs.append(jnp.asarray(g_t, dtype=h.real.dtype))
                Ois.append(Oi_arr)
                Ojs.append(Oj_arr)
            self.contact_g  = jnp.stack(gs)
            self.contact_Oi = jnp.stack(Ois)
            self.contact_Oj = jnp.stack(Ojs)

        # Embedded reference potential — see constructor docstring.
        self.embed_reference_potential = bool(embed_reference_potential)
        self.center_embedded_hartree = bool(center_embedded_hartree)
        self.embedded_energy_offset = 0.0
        self.embedded_hartree_shift = 0.0
        if self.embed_reference_potential:
            if reference_density is None:
                raise ValueError(
                    "embed_reference_potential=True requires reference_density."
                )
            ref = self.refP
            n_orb = int(h.shape[-1])
            real_dtype = h.real.dtype
            const_J_refP_refP = 0.0
            J_ref_diag = jnp.zeros(n_orb, dtype=real_dtype)
            if self.include_hartree:
                diag_R = jnp.einsum("ij,ijaa->a", w2d, ref).real
                J_ref_diag = self.HH @ diag_R
                # The constant offset uses the uncentered potential;
                # centering separately subtracts c*N(P) from reported energy.
                const_J_refP_refP = float(jnp.sum(J_ref_diag * diag_R))
                if self.center_embedded_hartree:
                    self.embedded_hartree_shift = float(jnp.mean(J_ref_diag))
                    J_ref_diag = J_ref_diag - self.embedded_hartree_shift
            const_Sigma_refP_refP = 0.0
            Sigma_ref = None
            if self.include_exchange:
                Sigma_ref = selfenergy_fft(
                    self._VR_shifted, self.weights_b * ref,
                    _apply_ifftshift=False,
                    hermitian_channel_packing=self.exchange_hermitian_channel_packing,
                )
                const_Sigma_refP_refP = float(jnp.einsum(
                    "ij,ijab,ijba->", w2d, Sigma_ref, ref,
                ).real)
            rho_bar_ref = jnp.einsum("ij,ijab->ab", w2d, ref)
            Sigma_contact_ref = _contact_self_energy(
                rho_bar_ref,
                self.contact_g,
                self.contact_Oi,
                self.contact_Oj,
            ).astype(h.dtype)
            const_contact_refP_refP = float(
                jnp.einsum("ab,ba->", Sigma_contact_ref, rho_bar_ref).real
            )
            h_eff = self.h
            if Sigma_ref is not None:
                h_eff = h_eff + Sigma_ref
            h_eff = h_eff + Sigma_contact_ref[None, None, ...]
            # Add per-orbital diagonal in JAX-compatible way.
            h_eff = h_eff + jnp.diag(J_ref_diag.astype(h.dtype))[None, None, :, :]
            self.h = h_eff
            self.embedded_energy_offset = 0.5 * (
                const_J_refP_refP
                + const_Sigma_refP_refP
                + const_contact_refP_refP
            )

    def as_args(self):
        """Dynamic inputs for jitted solver functions (no constant capture)."""
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
