"""Hartree-Fock kernel: the precomputed problem the solvers run on.

One kernel covers a dense orbital basis and a superlattice plane-wave basis.
The exchange it describes is::

    Σ(k) = -Σ_q V(k - q) w [P(q) - refP(q)]

with ``w`` the mesh weight.  On a dense basis it is a periodic FFT
convolution on the mesh; with ``plane_waves`` the basis state ``|k, G, α⟩``
carries momentum ``k - G`` and the convolution runs on an extended grid
(:mod:`hartreefock.superlattice`).

The kernel stores NumPy arrays; the jitted solvers read ``_as_args()``, and
its potentials are the same JAX functions the solvers call
(:mod:`hartreefock._jax.fock`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from ._jax import fock as _fock
from ._jax.utils import chemical_potential, hermitize, validate_temperature
from .mesh import Mesh


def _shift_phase(n0: int, n1: int) -> np.ndarray:
    """Phase that absorbs the meshgrid ifftshift into the FFT of V."""
    s0, s1 = n0 // 2, n1 // 2
    phase0 = np.exp(2j * np.pi * np.arange(n0) * s0 / n0)
    phase1 = np.exp(2j * np.pi * np.arange(n1) * s1 / n1)
    return (phase0[:, None] * phase1[None, :])[..., None, None]


def fermi_density(fock, weights, number, temperature):
    """Host diagonalization and a fixed-particle Fermi solve in double precision."""
    eps, orbitals = np.linalg.eigh(fock)
    mu = float(chemical_potential(eps, weights, number, temperature))
    p = expit((mu - eps) / temperature)
    density = (orbitals * p[..., None, :]) @ orbitals.conj().swapaxes(-2, -1)
    return density, eps, orbitals, p


def _is_jax(A) -> bool:
    return type(A).__module__.startswith("jax")


class HartreeFockKernel:
    """The Hartree-Fock problem on a k-mesh, precomputed for the solvers.

    Parameters
    ----------
    mesh : Mesh
        The k-mesh; its weights are the BZ quadrature.
    h : (nk, nk, D, D) array
        Single-particle Hamiltonian on the mesh.  With ``plane_waves``,
        ``D = n_G · n_orb`` with the G index slow.
    V : callable or array, optional
        The interaction.  A callable maps ``|q|`` to ``V`` of shape
        ``|q|.shape`` (one coupling) or ``|q|.shape + (n, n)`` (orbital
        resolved, ``n`` the orbitals of one plane-wave block).  A dense
        kernel also takes the table on :attr:`Mesh.transfer_grid`, shape
        ``(nk, nk)``, ``(nk, nk, 1, 1)`` or ``(nk, nk, D, D)``.  ``None``:
        no long-range term, for contact-only models.
    T : float
        Temperature (energy units).
    plane_waves : (n_G, 2) int array, optional
        Superlattice basis ``G`` in units of the reciprocal lattice vectors,
        which are the columns of ``mesh.cell``.
    exchange : bool
        Include the exchange (Fock) term.
    hartree : bool or array
        Include the Hartree term.  ``True`` derives the coupling from ``V``:
        ``V(q = 0)`` on a dense basis (all orbitals coupled for a scalar
        ``V``); ``V`` at the ``G`` differences with a plane-wave basis (the
        uniform ``G = 0`` term dropped for a scalar ``V``, kept for an
        orbital one).  An array sets it: ``(D, D)`` dense, ``(n_G, n_G)`` or
        ``(n_G, n_G, n, n)`` with plane waves.  Needs ``reference``.
    degeneracy : float
        Flavour multiplicity of the Hartree term, for flavours folded out of
        the basis.
    reference : (nk, nk, D, D) array, optional
        Density subtracted before building the self-energy.
    embed_reference : bool
        Add ``V_HF[reference]`` to ``h`` once, so the loop solves standard
        HF: ``F = h + V_HF[P]``.  Solvers report the standard HF energy; the
        constant they remove is ``embedded_energy_offset``.
    contact_terms : sequence of (g, O_i, O_j), optional
        q-independent interactions ``H = ½ g O_i ⊗ O_j`` (dense basis).  Each
        contributes ``σ_H = g O_i tr(O_j ρ̄)`` and ``σ_F = -g O_i ρ̄ O_j`` with
        ``ρ̄ = Σ_k w_k (P_k - refP_k)``.  A term other than its own h.c.
        partner ``(g, O_j†, O_i†)`` gets the partner added, so
        ``(U, n_up, n_down)`` is the Hubbard ``U n_up n_down``.
    sectors : sequence of index sequences, optional
        A partition of the basis that ``h``, the reference and the contact
        terms do not couple: a conserved quantum number, such as a valley no
        term mixes.  The solvers diagonalize each sector on its own and
        transform only the exchange channels inside a sector, so the density
        stays block diagonal.
    """

    def __init__(
        self,
        mesh: Mesh,
        h,
        V=None,
        *,
        T: float,
        plane_waves=None,
        exchange: bool = True,
        hartree=False,
        degeneracy: float = 1.0,
        reference=None,
        embed_reference: bool = False,
        contact_terms=None,
        sectors=None,
    ):
        if not isinstance(mesh, Mesh):
            raise TypeError(f"mesh must be a hartreefock.Mesh, got {type(mesh).__name__}")
        self.mesh = mesh
        self.T = validate_temperature(T)
        self.superlattice = plane_waves is not None
        h = np.asarray(h)
        if self.superlattice or not np.iscomplexobj(h):
            h = h.astype(np.complex128 if self.superlattice else np.result_type(h.dtype, np.complex64))
        if h.ndim != 4 or h.shape[:2] != mesh.shape or h.shape[-1] != h.shape[-2]:
            raise ValueError(f"h must have shape {mesh.shape} + (D, D), got {h.shape}")
        if h.dtype == np.complex128 and not jax.config.jax_enable_x64:
            raise ValueError(
                "a complex128 kernel needs jax_enable_x64, which importing hartreefock "
                "turns on; with it off, build the kernel from complex64 arrays"
            )
        self.h = np.ascontiguousarray(h)
        D = h.shape[-1]
        self.w2d = mesh.weights.astype(h.real.dtype)
        self.weights_b = self.w2d[..., None, None]
        self.weight_sum = float(self.w2d.sum())

        self.include_exchange = bool(exchange)
        self.include_hartree = hartree is not False and hartree is not None
        if not (self.include_exchange or self.include_hartree):
            raise ValueError("HartreeFockKernel needs exchange or Hartree.")
        if V is None and (self.superlattice or (self.include_hartree and hartree is True)):
            raise ValueError("V is required with plane_waves or hartree=True")
        self.degeneracy = float(degeneracy)

        if reference is not None:
            ref = np.asarray(reference, dtype=h.dtype)
            if ref.shape != h.shape:
                raise ValueError(f"reference must have shape {h.shape}, got {ref.shape}")
            self.refP = ref
        else:
            if self.include_hartree:
                raise ValueError("hartree needs a reference density")
            # Read-only zero view: no (nk, nk, D, D) allocation.
            self.refP = np.broadcast_to(np.zeros((), dtype=h.dtype), h.shape)
        self.has_reference_density = reference is not None

        self._contact(contact_terms, D)
        if self.superlattice:
            self._init_plane_waves(np.asarray(plane_waves, dtype=np.int64), V, hartree)
        else:
            self._init_dense(V, hartree)

        self.sectors = self._sectors(sectors)

        self.embed_reference = bool(embed_reference)
        self.embedded_energy_offset = 0.0
        if self.embed_reference:
            if reference is None:
                raise ValueError("embed_reference needs a reference density")
            self._embed_reference()

    # ----- construction -------------------------------------------------------------

    def _sectors(self, sectors):
        if sectors is None:
            return None
        sectors = tuple(tuple(int(i) for i in idx) for idx in sectors)
        D = self.h.shape[-1]
        if sorted(i for idx in sectors for i in idx) != list(range(D)):
            raise ValueError(f"sectors must partition the {D} basis states")
        coupled = np.ones((D, D), dtype=bool)
        for idx in sectors:
            coupled[np.ix_(idx, idx)] = False
        for name, A in (("h", self.h), ("reference", self.refP),
                        ("contact O_i", self.contact_Oi), ("contact O_j", self.contact_Oj)):
            leak = float(np.max(np.abs(A[..., coupled]), initial=0.0))
            if leak > 1e-10 * max(float(np.max(np.abs(A), initial=0.0)), 1.0):
                raise ValueError(f"{name} couples the sectors (|off-sector| = {leak:.2e})")
        self._sector_mask = (~coupled).astype(self.h.real.dtype)
        return sectors

    def _contact(self, terms, D) -> None:
        # Always at least one (possibly zero) term, so jitted shapes are static.
        terms = list(terms or ()) or [(0.0, np.zeros((D, D)), np.zeros((D, D)))]
        if self.superlattice and any(float(t[0]) for t in terms):
            raise ValueError("contact_terms need a dense basis")
        gs, Ois, Ojs = [], [], []
        for k, term in enumerate(terms):
            if len(term) != 3:
                raise ValueError(f"contact_terms[{k}] must be (g, O_i, O_j); got len={len(term)}")
            g_t, Oi_t, Oj_t = term
            for name, arr in (("O_i", Oi_t), ("O_j", Oj_t)):
                if np.shape(arr) != (D, D):
                    raise ValueError(
                        f"contact_terms[{k}].{name} must have shape {(D, D)}, got {np.shape(arr)}"
                    )
            Oi_t, Oj_t = (np.asarray(O, dtype=self.h.dtype) for O in (Oi_t, Oj_t))
            pairs = [(Oi_t, Oj_t)]
            partner = (Oj_t.conj().T, Oi_t.conj().T)
            if not (np.allclose(partner[0], Oi_t) and np.allclose(partner[1], Oj_t)):
                pairs.append(partner)  # the h.c. partner makes the term Hermitian
            for Oi_p, Oj_p in pairs:
                gs.append(float(g_t))
                Ois.append(Oi_p)
                Ojs.append(Oj_p)
        self.contact_g = np.asarray(gs, dtype=self.h.real.dtype)
        self.contact_Oi = np.stack(Ois)
        self.contact_Oj = np.stack(Ojs)

    def _init_dense(self, V, hartree) -> None:
        h, mesh = self.h, self.mesh
        D = h.shape[-1]
        if V is None:
            Vq = np.zeros((*mesh.shape, 1, 1))
        else:
            Vq = np.asarray(V(mesh.transfer_distance_grid) if callable(V) else V)
            if Vq.shape == mesh.shape:
                Vq = Vq[..., None, None]
            if Vq.shape not in ((*mesh.shape, 1, 1), (*mesh.shape, D, D)):
                raise ValueError(
                    f"V on the mesh must have shape {mesh.shape}, {(*mesh.shape, 1, 1)} "
                    f"or {(*mesh.shape, D, D)}, got {Vq.shape}"
                )
        if np.iscomplexobj(Vq) and float(np.max(np.abs(Vq.imag))) <= 1e-8:
            Vq = Vq.real
        self.exchange_hermitian_channel_packing = bool(
            Vq.shape[-2:] == (1, 1) and not np.iscomplexobj(Vq)
        )
        phase = _shift_phase(*mesh.shape)
        shifted = np.fft.fftn(Vq.astype(h.dtype), axes=(0, 1)) * phase
        # A sampled radial V(q) need not obey periodic inversion on even
        # oblique grids.  The quadratic energy only sees the self-adjoint part
        # of the convolution; keeping the skew part in F breaks dE/dP = F.
        self.exchange_reciprocity_defect = float(
            np.linalg.norm(shifted.imag) / max(np.linalg.norm(shifted), np.finfo(float).tiny)
        )
        self.VR_shifted = np.ascontiguousarray(shifted.real.astype(h.dtype))

        real = h.real.dtype
        if not self.include_hartree:
            HH = np.zeros((D, D))
        elif hartree is True:
            c = mesh.center_index
            V0 = np.real(Vq[c, c])
            HH = V0[0, 0] * np.ones((D, D)) if V0.shape == (1, 1) else V0
        else:
            HH = np.asarray(hartree, dtype=real)
            if HH.shape != (D, D):
                raise ValueError(f"hartree must have shape {(D, D)}, got {HH.shape}")
        self.HH = self.degeneracy * np.asarray(HH, dtype=real)

    def _init_plane_waves(self, g, V, hartree) -> None:
        from .superlattice import build_extended_layout

        mesh, D = self.mesh, self.h.shape[-1]
        if not mesh.periodic:
            raise ValueError("plane_waves need a periodic (Brillouin-zone) mesh")
        if not callable(V):
            raise ValueError("with plane_waves, V must be a callable of |q|")
        n_G = g.shape[0]
        if D % n_G:
            raise ValueError(f"h dimension D={D} is not a multiple of n_G={n_G}")
        self.plane_waves, self.n_G, self.dim_orb = g, n_G, D // n_G
        probe = np.asarray(V(np.ones(1)))
        self.orbital_V = probe.ndim == 3
        if self.orbital_V and probe.shape[1:] != (self.dim_orb, self.dim_orb):
            raise ValueError(
                f"V returns {probe.shape[1:]} blocks but a plane-wave block has "
                f"{self.dim_orb} orbitals"
            )
        self.V = V
        self.layout = build_extended_layout(g, mesh.cell, mesh.nk, mesh.nk, V, mesh.weight)
        self._exchange_fn = None
        self.HH = self._plane_wave_hartree(hartree)
        # Dense-kernel solver inputs; the plane-wave Fock builder ignores them.
        self.VR_shifted = np.zeros((*mesh.shape, 1, 1), dtype=np.complex128)
        self.exchange_hermitian_channel_packing = False

    def _plane_wave_hartree(self, hartree):
        n_G, no = self.n_G, self.dim_orb
        if not self.include_hartree:
            return np.zeros((n_G, n_G))
        if hartree is True:
            g_cart = self.plane_waves @ self.mesh.cell.T
            g_diff = np.linalg.norm(g_cart[:, None, :] - g_cart[None, :, :], axis=-1)
            HH = np.asarray(self.V(g_diff), dtype=np.float64)
            if not self.orbital_V:
                np.fill_diagonal(HH, 0.0)  # the gate neutralises the uniform charge
        else:
            HH = np.asarray(hartree, dtype=np.float64)
        if HH.shape not in ((n_G, n_G), (n_G, n_G, no, no)):
            raise ValueError(
                f"hartree must have shape {(n_G, n_G)} or {(n_G, n_G, no, no)}, got {HH.shape}"
            )
        if not np.all(np.isfinite(HH)):
            raise ValueError("hartree must be finite")
        return np.ascontiguousarray(HH)

    def _embed_reference(self) -> None:
        ref = self.refP
        V_ref = self.potential(ref)
        self.embedded_energy_offset = 0.5 * self._trace(V_ref, ref)
        self.h = np.ascontiguousarray(self.h + V_ref.astype(self.h.dtype))

    # ----- observables ---------------------------------------------------------------

    def noninteracting_density(self, n_electrons, field=None):
        """The Fermi-filled density of ``h + field`` with ``n_electrons`` at ``T``.

        ``hf.solve`` starts from it when ``P0`` is ``None``.  A ``field``, a
        ``(D, D)`` matrix or one per k-point, biases the start, for example
        ``diag(-δ, δ)`` toward spin polarization.
        """
        H = np.asarray(self.h, dtype=np.complex128) + (0.0 if field is None else np.asarray(field))
        return fermi_density(H, self.w2d.astype(np.float64), float(n_electrons), self.T)[0]

    def within_sectors(self, P):
        """``P`` with the blocks between different :attr:`sectors` set to zero."""
        return P if self.sectors is None else P * self._sector_mask

    def mean_density(self, P):
        """The k-summed density ``ρ̄ = Σ_k w_k P_k`` ``(D, D)``; its diagonal holds the orbital occupations."""
        return (jnp if _is_jax(P) else np).einsum("ij,ijab->ab", self.w2d, P)

    def energies(self, P) -> dict[str, float]:
        """Kinetic, Hartree and exchange parts of the standard HF energy at ``P``.

        They sum to the ``energy`` that :func:`hartreefock.solve` reports; the
        contact terms count in the Hartree and exchange parts.
        """
        P = np.asarray(P)
        rho = P if self.embed_reference else P - self.refP
        kinetic = self._trace(self.h, P)
        if self.embed_reference:
            kinetic -= self._trace(self.potential(self.refP), P)
        hartree = np.broadcast_to(self.hartree(rho), P.shape)
        return {"kinetic": kinetic, "hartree": 0.5 * self._trace(hartree, rho),
                "exchange": 0.5 * self._trace(self.exchange(rho), rho)}

    def _trace(self, A, B) -> float:
        return float(np.einsum("ij,ijab,ijba->", self.w2d, A, B).real)

    # ----- linear maps (no reference subtraction) -----------------------------------

    def exchange(self, rho) -> np.ndarray:
        """Fock self-energy ``Σ_F[rho]`` on the mesh, with the contact Fock part.

        ``rho`` need not be Hermitian (a response coherence, say).  NumPy in,
        NumPy out; JAX input, a tracer under ``jax.jit`` included, gives JAX.
        """
        traced = _is_jax(rho)
        rho = jnp.asarray(rho, dtype=self.h.dtype)
        if self.superlattice:
            if not self.include_exchange:
                out = jnp.zeros(self.h.shape, dtype=self.h.dtype)
            else:
                if self._exchange_fn is None:
                    from ._jax.superlattice import make_superlattice_fock_fn

                    self._exchange_fn = make_superlattice_fock_fn(
                        self.layout, self.n_G, self.dim_orb, self.mesh.nk, self.mesh.nk
                    )
                out = self._exchange_fn(rho)
        else:
            out = jnp.zeros(self.h.shape, dtype=self.h.dtype)
            if self.include_exchange:
                out = out + _fock.dense_exchange(rho, VR=self.VR_shifted, w2d=self.w2d)
            out = out + self._contact_potentials(rho)[1][None, None]
        return out if traced else np.asarray(out)

    def hartree(self, rho) -> np.ndarray:
        """k-independent Hartree potential ``(D, D)``, with the contact Hartree part."""
        traced = _is_jax(rho)
        rho = jnp.asarray(rho, dtype=self.h.dtype)
        D = self.h.shape[-1]
        out = jnp.zeros((D, D), dtype=self.h.dtype)
        if self.include_hartree and self.superlattice:
            from .superlattice import superlattice_hartree

            out = superlattice_hartree(
                self.mean_density(rho), self.layout.pair_to_delta, self.HH, degeneracy=self.degeneracy
            )
        elif self.include_hartree:
            out = _fock.dense_hartree(rho, HH=self.HH, w2d=self.w2d, dtype=self.h.dtype)
        if not self.superlattice:
            out = out + self._contact_potentials(rho)[0]
        return out if traced else np.asarray(out)

    def potential(self, rho) -> np.ndarray:
        """Hermitized ``V_HF[rho] = Σ_F[rho] + σ_H[rho]``; the solver's Fock is ``h + potential(P - refP)``."""
        out = hermitize(jnp.asarray(self.exchange(rho)) + self.hartree(rho)[None, None])
        return out if _is_jax(rho) else np.asarray(out)

    def _contact_potentials(self, rho):
        return _fock.contact(
            rho, w2d=self.w2d, contact_g=self.contact_g, contact_Oi=self.contact_Oi, contact_Oj=self.contact_Oj
        )

    def potential_at(self, P, k_fractional, *, include_embedded_reference: bool = True):
        """``V_HF[P - refP]`` at arbitrary fractional k (e.g. a band path); plane waves only.

        Direct plane-wave sum ``O(n_k · nk² · n_G²)``: matrix elements of
        ``Σ(k)`` in a truncated G basis are not gauge-periodic, so they cannot
        be interpolated from the mesh.  With ``include_embedded_reference``
        the embedded ``V_HF[refP]`` is added, so ``h_bare + potential_at``
        reproduces the mean-field Hamiltonian of an embedded kernel.
        """
        if not self.superlattice:
            raise ValueError("potential_at needs a plane-wave kernel")
        from .superlattice import exchange_at

        k = np.asarray(k_fractional, dtype=float)
        squeeze = k.ndim == 1
        k = np.atleast_2d(k)
        if k.shape[-1] != 2:
            raise ValueError(f"k_fractional must have shape (n, 2); got {k.shape}")
        P = np.asarray(P, dtype=np.complex128)
        refP = np.asarray(self.refP)
        embedded = include_embedded_reference and self.embed_reference
        rho = P if embedded else P - refP
        sigma = exchange_at(self, rho, k) if self.include_exchange else 0.0
        if self.include_hartree:
            sigma = sigma + self.hartree(P - refP)[None]
            if embedded:
                sigma = sigma + self.hartree(refP)[None]
        D = self.h.shape[-1]
        sigma = np.asarray(hermitize(np.broadcast_to(sigma, (k.shape[0], D, D))))
        return sigma[0] if squeeze else sigma

    def _as_args(self) -> dict:
        """Dynamic solver inputs (passed to jitted functions, never captured)."""
        D = self.h.shape[-1]
        return dict(
            h=self.h,
            weights_b=self.weights_b,
            weight_sum=self.weight_sum,
            VR=self.VR_shifted,
            T=self.T,
            refP=self.refP,
            HH=np.zeros((D, D)) if self.superlattice else self.HH,
            include_hartree=self.include_hartree,
            include_exchange=self.include_exchange,
            exchange_hermitian_channel_packing=self.exchange_hermitian_channel_packing,
            contact_g=self.contact_g,
            contact_Oi=self.contact_Oi,
            contact_Oj=self.contact_Oj,
            sectors=self.sectors,
        )
