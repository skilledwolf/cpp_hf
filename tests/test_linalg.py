"""Tests for cpp_hf.linalg / selfenergy_fft block specs."""

import numpy as np

from cpp_hf.linalg import eigh
from cpp_hf.utils import selfenergy_fft


def test_block_eigh_falls_back_to_full_when_coupled():
    h = np.diag(np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)).astype(np.complex64)
    h[0, 2] = 1e-2 + 0.0j
    h[2, 0] = 1e-2 + 0.0j

    w_ref, _v_ref = np.linalg.eigh(h)
    w, v = eigh(h, block_sizes=(2, 2), check_offdiag=True, offdiag_atol=1e-6)

    np.testing.assert_allclose(np.array(w), np.array(w_ref), rtol=1e-6, atol=1e-6)
    h_rec = v @ np.diag(w) @ np.conj(np.swapaxes(v, -1, -2))
    np.testing.assert_allclose(np.array(h_rec), np.array(h), rtol=1e-6, atol=1e-6)


def test_block_eigh_forced_matches_full_when_exactly_block_diagonal():
    h0 = np.array(
        [
            [-1.0 + 0.0j, 0.2 + 0.1j],
            [0.2 - 0.1j, -0.3 + 0.0j],
        ],
        dtype=np.complex64,
    )
    h1 = np.array(
        [
            [0.4 + 0.0j, -0.15j],
            [0.15j, 1.1 + 0.0j],
        ],
        dtype=np.complex64,
    )
    h = np.zeros((4, 4), dtype=np.complex64)
    h[:2, :2] = h0
    h[2:, 2:] = h1

    w_ref, _v_ref = np.linalg.eigh(h)
    w, v = eigh(h, block_sizes=(2, 2), check_offdiag=False)

    np.testing.assert_allclose(np.array(w), np.array(w_ref), rtol=1e-6, atol=1e-6)
    h_rec = v @ np.diag(w) @ np.conj(np.swapaxes(v, -1, -2))
    np.testing.assert_allclose(np.array(h_rec), np.array(h), rtol=1e-6, atol=1e-6)


def test_selfenergy_fft_block_specs_matches_full_when_block_diagonal():
    rng = np.random.default_rng(0)
    nk = 4
    n0 = 2
    nb = 2 * n0

    P = np.zeros((nk, nk, nb, nb), dtype=np.complex64)
    for b in range(2):
        s = slice(b * n0, (b + 1) * n0)
        block = rng.normal(size=(nk, nk, n0, n0)) + 1j * rng.normal(size=(nk, nk, n0, n0))
        block = 0.5 * (block + np.conj(np.swapaxes(block, -1, -2)))
        P[..., s, s] = block.astype(np.complex64)

    VR = np.ones((nk, nk, 1, 1), dtype=np.complex64)

    sigma_full = selfenergy_fft(VR, P)
    sigma_block = selfenergy_fft(
        VR,
        P,
        block_specs=[{"block_sizes": [n0, n0]}],
        check_offdiag=True,
        offdiag_atol=1e-12,
    )
    np.testing.assert_allclose(np.array(sigma_block), np.array(sigma_full), rtol=1e-6, atol=1e-6)


def test_selfenergy_fft_hermitian_channel_packing_matches_full_for_scalar_real_kernel():
    rng = np.random.default_rng(123)
    nk = 8
    nb = 6

    raw = rng.normal(size=(nk, nk, nb, nb)) + 1j * rng.normal(size=(nk, nk, nb, nb))
    P = 0.5 * (raw + np.conj(np.swapaxes(raw, -1, -2)))
    P = P.astype(np.complex64)

    Vq = rng.normal(size=(nk, nk, 1, 1)).astype(np.float32)
    VR = np.fft.fftn(Vq.astype(np.complex64), axes=(0, 1))

    sigma_full = selfenergy_fft(VR, P)
    sigma_packed = selfenergy_fft(VR, P, hermitian_channel_packing=True)

    np.testing.assert_allclose(
        np.array(sigma_packed),
        np.array(sigma_full),
        rtol=1e-6, atol=1e-6,
    )


def test_selfenergy_fft_block_specs_works_with_hermitian_channel_packing():
    rng = np.random.default_rng(7)
    nk = 8
    block_sizes = (2, 2, 2)
    nb = sum(block_sizes)

    P = np.zeros((nk, nk, nb, nb), dtype=np.complex64)
    start = 0
    for size in block_sizes:
        stop = start + size
        raw = rng.normal(size=(nk, nk, size, size)) + 1j * rng.normal(size=(nk, nk, size, size))
        block = 0.5 * (raw + np.conj(np.swapaxes(raw, -1, -2)))
        P[..., start:stop, start:stop] = block.astype(np.complex64)
        start = stop

    Vq = rng.normal(size=(nk, nk, 1, 1)).astype(np.float32)
    VR = np.fft.fftn(Vq.astype(np.complex64), axes=(0, 1))

    sigma_full = selfenergy_fft(VR, P)
    sigma_block = selfenergy_fft(
        VR, P,
        block_specs=[{"block_sizes": list(block_sizes)}],
        check_offdiag=True,
        hermitian_channel_packing=True,
    )

    # The block + HCP path uses two different FFT routes than the plain
    # full path; in float32 the reduction orders differ, leaving residual
    # noise at ~5e-6.  The values agree analytically.
    np.testing.assert_allclose(
        np.array(sigma_block),
        np.array(sigma_full),
        rtol=5e-6, atol=5e-6,
    )


def test_selfenergy_fft_block_specs_falls_back_to_full_when_coupled():
    nk = 4
    n0 = 2
    nb = 2 * n0

    P = np.zeros((nk, nk, nb, nb), dtype=np.complex64)
    P[:, :, 0, 0] = 1.0 + 0.0j
    P[:, :, 2, 2] = 0.5 + 0.0j
    P[:, :, 0, 2] = 1e-1 + 0.0j
    P[:, :, 2, 0] = 1e-1 + 0.0j

    VR = np.ones((nk, nk, 1, 1), dtype=np.complex64)
    sigma_full = selfenergy_fft(VR, P)

    sigma_auto = selfenergy_fft(
        VR, P,
        block_specs=[{"block_sizes": [n0, n0]}],
        check_offdiag=True,
        offdiag_atol=1e-6,
    )
    np.testing.assert_allclose(np.array(sigma_auto), np.array(sigma_full), rtol=1e-6, atol=1e-6)

    sigma_forced = selfenergy_fft(
        VR, P,
        block_specs=[{"block_sizes": [n0, n0]}],
        check_offdiag=False,
    )
    diff = float(np.max(np.abs(sigma_forced - sigma_full)))
    assert diff > 1e-3
