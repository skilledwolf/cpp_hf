"""Tests for cpp_hf.utils (port of jax_hf.tests.test_main)."""

import numpy as np
import pytest

from cpp_hf import HartreeFockKernel
from cpp_hf._compat import _native
from cpp_hf.fock import build_fock, hf_energy
from cpp_hf.solver import _kernel_args_for_native
from cpp_hf.utils import (
    density_matrix_from_fock,
    fermidirac,
    find_chemical_potential,
    selfenergy_fft,
)


def test_find_chemical_potential_hits_target_density():
    bands = np.array([[[0.0, 1.0]]], dtype=np.float32)
    weights = np.ones((1, 1), dtype=np.float32)
    mu = find_chemical_potential(bands, weights, n_electrons=1.0, T=0.1)
    occ = fermidirac(bands - mu, 0.1)
    total = float(np.sum(weights[..., None] * occ))
    assert abs(total - 1.0) < 1e-4


@pytest.mark.parametrize("method", ["bisection", "newton", "native"])
@pytest.mark.parametrize("target", [1e-25, 1e-6, 2.0 - 1e-6])
@pytest.mark.parametrize("bandwidth", [0.0, 0.2])
def test_chemical_potential_bounds_cover_low_and_high_fillings(method, target, bandwidth):
    bands = np.array([[[0.0, bandwidth]]])
    weights = np.ones((1, 1))
    T = 0.1
    if method == "native":
        mu = _native.find_mu_bisection(bands, weights, target, T)
    else:
        mu = find_chemical_potential(bands, weights, target, T, method=method)
    # Test holes directly near full filling, without subtracting two counts
    # of order one and hiding a relative error in the unoccupied fraction.
    if target > 1.0:
        count = np.sum(fermidirac(mu - bands, T))
        expected = 2.0 - target
    else:
        count = np.sum(fermidirac(bands - mu, T))
        expected = target
    np.testing.assert_allclose(count, expected, rtol=2e-8, atol=0.0)


@pytest.mark.parametrize("acceleration", ["linear", "diis", "oda"])
def test_zero_temperature_scf_preserves_flat_band_particle_count(acceleration):
    from cpp_hf import SCFConfig, solve_scf

    h = np.zeros((1, 1, 2, 2), complex)
    kernel = HartreeFockKernel(np.ones((1, 1)), h, np.zeros((1, 1, 1, 1)), T=0.0)
    density = np.eye(2)[None, None] * 0.05
    result = solve_scf(kernel, density, 0.1, config=SCFConfig(
        acceleration=acceleration, max_iter=4, mixing=1.0,
        density_tol=1e-12, comm_tol=1e-12,
    ))
    assert result.converged
    np.testing.assert_allclose(np.trace(result.density_matrix[0, 0]), 0.1, atol=1e-14)


def test_find_chemical_potential_rejects_unreachable_density():
    bands = np.array([[[0.0]]], dtype=np.float32)
    weights = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="physically reachable range"):
        find_chemical_potential(bands, weights, n_electrons=2.0, T=0.1)


def test_density_matrix_from_fock_rejects_unreachable_density():
    F = np.array([[[[0.0]]]], dtype=np.complex64)
    weights = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="physically reachable range"):
        density_matrix_from_fock(F, weights, n_electrons=2.0, T=0.1)


def test_selfenergy_fft_single_point_matches_definition():
    VR = np.array([[[[0.3]]]], dtype=np.complex64)
    P = np.array([[[[1.0 + 0.5j, 0.2], [0.2, 0.8 - 0.1j]]]], dtype=np.complex64)
    sigma = selfenergy_fft(VR, P)
    expected = -VR * P
    np.testing.assert_allclose(np.array(sigma), np.array(expected), rtol=1e-6, atol=1e-6)


def test_native_build_fock_with_hartree_matches_python_reference():
    rng = np.random.default_rng(4)
    nk, nb = 3, 3
    weights = np.ones((nk, nk), dtype=np.float64) / (nk * nk)
    raw_h = rng.normal(size=(nk, nk, nb, nb)) + 1j * rng.normal(size=(nk, nk, nb, nb))
    h = 0.5 * (raw_h + np.conj(np.swapaxes(raw_h, -1, -2)))
    raw_p = rng.normal(size=(nk, nk, nb, nb)) + 1j * rng.normal(size=(nk, nk, nb, nb))
    P = 0.5 * (raw_p + np.conj(np.swapaxes(raw_p, -1, -2)))
    raw_ref = rng.normal(size=(nk, nk, nb, nb)) + 1j * rng.normal(size=(nk, nk, nb, nb))
    refP = 0.1 * (raw_ref + np.conj(np.swapaxes(raw_ref, -1, -2)))
    HH = np.array(
        [[1.0, 0.2, -0.1], [0.2, 0.7, 0.05], [-0.1, 0.05, 0.9]],
        dtype=np.float64,
    )
    Vq = np.full((nk, nk, 1, 1), 0.15, dtype=np.float64)
    kernel = HartreeFockKernel(
        weights,
        h,
        Vq,
        T=0.1,
        include_hartree=True,
        include_exchange=True,
        reference_density=refP,
        hartree_matrix=HH,
    )

    sigma_ref, hartree_ref, fock_ref = build_fock(
        P,
        h=kernel.h,
        VR=kernel._VR_shifted,
        refP=kernel.refP,
        HH=kernel.HH,
        w2d=kernel.w2d,
        include_exchange=kernel.include_exchange,
        include_hartree=kernel.include_hartree,
        exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
        contact_g=kernel.contact_g,
        contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
    )
    energy_ref = hf_energy(
        P,
        h=kernel.h,
        Sigma=sigma_ref,
        H=hartree_ref,
        weights_b=kernel.weights_b,
        refP=kernel.refP,
    )

    sigma, hartree, fock, energy = _native.build_fock_apply(
        _kernel_args_for_native(kernel), np.ascontiguousarray(P), None
    )

    np.testing.assert_allclose(sigma, sigma_ref, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        hartree, np.broadcast_to(hartree_ref, hartree.shape), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(fock, fock_ref, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(energy, energy_ref, rtol=1e-11, atol=1e-11)


def test_hf_energy_with_reference_density_has_fock_derivative():
    weights = np.ones((1, 1), dtype=np.float64)
    h = np.array(
        [[[-0.3, 0.04 - 0.02j], [0.04 + 0.02j, 0.5]]],
        dtype=np.complex128,
    ).reshape(1, 1, 2, 2)
    P = np.array(
        [[[0.8, 0.07 + 0.03j], [0.07 - 0.03j, 0.2]]],
        dtype=np.complex128,
    ).reshape(1, 1, 2, 2)
    refP = np.array(
        [[[0.35, -0.02 + 0.01j], [-0.02 - 0.01j, 0.15]]],
        dtype=np.complex128,
    ).reshape(1, 1, 2, 2)
    dP = np.array(
        [[[0.11, 0.05 - 0.04j], [0.05 + 0.04j, -0.07]]],
        dtype=np.complex128,
    ).reshape(1, 1, 2, 2)
    HH = np.array([[1.2, 0.1], [0.1, 0.8]], dtype=np.float64)
    Vq = np.full((1, 1, 1, 1), 0.7, dtype=np.float64)
    kernel = HartreeFockKernel(
        weights,
        h,
        Vq,
        T=0.1,
        include_hartree=True,
        include_exchange=True,
        reference_density=refP,
        hartree_matrix=HH,
    )

    def energy_and_fock(P_eval):
        sigma, hartree, fock = build_fock(
            P_eval,
            h=kernel.h,
            VR=kernel._VR_shifted,
            refP=kernel.refP,
            HH=kernel.HH,
            w2d=kernel.w2d,
            include_exchange=kernel.include_exchange,
            include_hartree=kernel.include_hartree,
            exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
            contact_g=kernel.contact_g,
            contact_Oi=kernel.contact_Oi,
            contact_Oj=kernel.contact_Oj,
        )
        energy = hf_energy(
            P_eval,
            h=kernel.h,
            Sigma=sigma,
            H=hartree,
            weights_b=kernel.weights_b,
            refP=kernel.refP,
        )
        return float(energy), fock

    eps = 1e-6
    e_plus, _ = energy_and_fock(P + eps * dP)
    e_minus, _ = energy_and_fock(P - eps * dP)
    _, fock = energy_and_fock(P)
    finite_difference = (e_plus - e_minus) / (2.0 * eps)
    fock_derivative = float(
        np.sum(np.real(np.einsum("...ij,...ji->...", kernel.weights_b * fock, dP)))
    )

    np.testing.assert_allclose(finite_difference, fock_derivative, rtol=1e-7, atol=1e-8)


def test_native_build_fock_project_callback_after_gil_release():
    weights = np.ones((1, 1), dtype=np.float64)
    h = np.eye(2, dtype=np.complex128)[None, None, :, :]
    Vq = np.ones((1, 1, 1, 1), dtype=np.float64)
    kernel = HartreeFockKernel(weights, h, Vq, T=0.1)
    P = np.zeros_like(h)

    def project_to_zero(F):
        return np.zeros_like(F)

    _sigma, _hartree, fock, _energy = _native.build_fock_apply(
        _kernel_args_for_native(kernel), P, project_to_zero
    )

    np.testing.assert_allclose(fock, np.zeros_like(fock), atol=0.0)


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_find_chemical_potential_hits_target_multiband(method):
    rng = np.random.RandomState(42)
    nk1, nk2, nb = 4, 4, 6
    bands = rng.randn(nk1, nk2, nb).astype(np.float32)
    weights = np.ones((nk1, nk2), dtype=np.float32) / (nk1 * nk2)
    n_target = 3.0
    T = 0.05

    mu = find_chemical_potential(bands, weights, n_target, T, method=method)
    occ = fermidirac(bands - mu, T)
    total = float(np.sum(weights[..., None] * occ))
    assert abs(total - n_target) < 1e-4


@pytest.mark.parametrize("method", ["bisection", "newton"])
def test_find_chemical_potential_cold_limit(method):
    bands = np.array([[[0.0, 1.0, 2.0]]])
    weights = np.ones((1, 1))

    mu = find_chemical_potential(bands, weights, n_electrons=2.0, T=1e-5, method=method)
    occ = fermidirac(bands - mu, 1e-5)
    total = float(np.sum(weights[..., None] * occ))
    assert abs(total - 2.0) < 1e-4
