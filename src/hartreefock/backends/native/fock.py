"""Fock matrix construction and energy evaluation (numpy port of ``jax_hf.fock``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import selfenergy_fft


def _herm(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))


def build_fock(
    P: np.ndarray,
    *,
    h: np.ndarray,
    VR: np.ndarray,
    refP: np.ndarray,
    HH: np.ndarray,
    w2d: np.ndarray,
    include_exchange: bool,
    include_hartree: bool,
    exchange_hermitian_channel_packing: bool,
    contact_g: np.ndarray | None = None,
    contact_Oi: np.ndarray | None = None,
    contact_Oj: np.ndarray | None = None,
    exchange_block_specs: Any | None = None,
    exchange_check_offdiag: bool | None = None,
    exchange_offdiag_atol: float = 1e-12,
    exchange_offdiag_rtol: float = 0.0,
    project_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Fock matrix and return ``(Sigma, H_hartree, F)``.

    F = project(hermitize(h + Sigma[P] + H[P])) — see jax_hf docstring for the
    contact-term formula.
    """
    if refP is None:
        dP = P
    else:
        refP_arr = np.asarray(refP)
        if refP_arr.size and all(s == 0 for s in refP_arr.strides) and refP_arr.flat[0] == 0:
            dP = P
        else:
            dP = P - refP_arr

    if include_exchange:
        Sigma = selfenergy_fft(
            VR,
            dP,
            block_specs=exchange_block_specs,
            check_offdiag=exchange_check_offdiag,
            offdiag_atol=exchange_offdiag_atol,
            offdiag_rtol=exchange_offdiag_rtol,
            _apply_ifftshift=False,
            hermitian_channel_packing=exchange_hermitian_channel_packing,
        )
    else:
        Sigma = np.zeros_like(h)

    if include_hartree:
        diag_real = np.real(np.diagonal(dP, axis1=-2, axis2=-1))
        n_vec = np.sum(w2d[..., None] * diag_real, axis=(0, 1))
        sigma_diag = HH @ n_vec
        H_mat = np.diag(sigma_diag.astype(h.real.dtype))
        H = H_mat[None, None, ...]
    else:
        H = np.zeros_like(h)

    if contact_g is not None and contact_Oi is not None and contact_Oj is not None:
        rho_bar = np.einsum("ij,ijab->ab", w2d, dP)
        tr_Oj_rho = np.einsum("tij,ji->t", contact_Oj, rho_bar)
        sigma_h_contact = np.einsum(
            "t,tij->ij", contact_g * tr_Oj_rho, contact_Oi
        )
        oi_rho = np.einsum("tij,jk->tik", contact_Oi, rho_bar)
        oi_rho_oj = np.einsum("tik,tkl->til", oi_rho, contact_Oj)
        sigma_f_contact = np.einsum("t,tij->ij", -contact_g, oi_rho_oj)
        sigma_contact = (sigma_h_contact + sigma_f_contact).astype(h.dtype)
        Sigma = Sigma + sigma_contact[None, None, ...]

    F = _herm(h + Sigma + H)
    if project_fn is not None:
        F = _herm(np.asarray(project_fn(F), dtype=F.dtype))
    return Sigma, H, F


def hf_energy(
    P: np.ndarray,
    *,
    h: np.ndarray,
    Sigma: np.ndarray,
    H: np.ndarray,
    weights_b: np.ndarray,
    refP: np.ndarray | None = None,
) -> np.ndarray:
    """E = Σ_k w_k Tr[hP] + ½Σ_k w_k Tr[(Σ+H)(P-refP)]."""
    dP = P if refP is None else P - refP
    weights = np.asarray(weights_b)[..., 0, 0]
    one_body = weights * np.einsum("...ij,...ji->...", h, P)
    sigma_term = np.einsum("...ij,...ji->...", Sigma, dP)
    hartree_term = np.einsum("...ij,...ji->...", H, dP)
    interaction = 0.5 * weights * (sigma_term + hartree_term)
    return np.sum(np.real(one_body + interaction))


def occupation_entropy(p: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
    p_safe = np.clip(p, 1e-14, 1.0 - 1e-14)
    s = p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log1p(-p_safe)
    return -np.sum(w_norm[..., None] * s)


def free_energy(
    E: np.ndarray,
    p: np.ndarray,
    w_norm: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Free energy Ω = E − T·S(p)."""
    T_val = max(float(T), 1e-14)
    return E - T_val * occupation_entropy(p, w_norm)
