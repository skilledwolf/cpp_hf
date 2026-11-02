import numpy as np

import jax.numpy as jnp

from hartreefock._jax.linalg import eigh
from hartreefock._jax.utils import selfenergy_fft


def test_block_eigh_falls_back_to_full_when_coupled():
    # 2x2 block structure, but add a large off-block coupling so the check fails.
    h = jnp.diag(jnp.array([-1.0, -0.5, 0.5, 1.0], dtype=jnp.float32)).astype(jnp.complex64)
    h = h.at[0, 2].set(1e-2 + 0.0j)
    h = h.at[2, 0].set(1e-2 + 0.0j)

    w_ref, _v_ref = jnp.linalg.eigh(h)
    w, v = eigh(h, block_sizes=(2, 2), check_offdiag=True, offdiag_atol=1e-6)

    np.testing.assert_allclose(np.array(w), np.array(w_ref), rtol=1e-6, atol=1e-6)
    h_rec = v @ jnp.diag(w) @ jnp.conj(jnp.swapaxes(v, -1, -2))
    np.testing.assert_allclose(np.array(h_rec), np.array(h), rtol=1e-6, atol=1e-6)


def test_block_eigh_forced_matches_full_when_exactly_block_diagonal():
    h0 = jnp.array(
        [
            [-1.0 + 0.0j, 0.2 + 0.1j],
            [0.2 - 0.1j, -0.3 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    h1 = jnp.array(
        [
            [0.4 + 0.0j, -0.15j],
            [0.15j, 1.1 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    h = jnp.zeros((4, 4), dtype=jnp.complex64)
    h = h.at[:2, :2].set(h0)
    h = h.at[2:, 2:].set(h1)

    w_ref, _v_ref = jnp.linalg.eigh(h)
    w, v = eigh(h, block_sizes=(2, 2), check_offdiag=False)

    np.testing.assert_allclose(np.array(w), np.array(w_ref), rtol=1e-6, atol=1e-6)
    h_rec = v @ jnp.diag(w) @ jnp.conj(jnp.swapaxes(v, -1, -2))
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

    VR = jnp.asarray(np.ones((nk, nk, 1, 1), dtype=np.complex64))
    Pj = jnp.asarray(P)

    sigma_full = selfenergy_fft(VR, Pj)
    sigma_block = selfenergy_fft(
        VR,
        Pj,
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
    Pj = jnp.asarray(P.astype(np.complex64))

    Vq = rng.normal(size=(nk, nk, 1, 1)).astype(np.float32)
    VR = jnp.fft.fftn(jnp.asarray(Vq, dtype=jnp.complex64), axes=(0, 1))

    sigma_full = selfenergy_fft(VR, Pj)
    sigma_packed = selfenergy_fft(
        VR,
        Pj,
        hermitian_channel_packing=True,
    )

    np.testing.assert_allclose(
        np.array(sigma_packed),
        np.array(sigma_full),
        rtol=1e-6,
        atol=1e-6,
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
    Pj = jnp.asarray(P)

    Vq = rng.normal(size=(nk, nk, 1, 1)).astype(np.float32)
    VR = jnp.fft.fftn(jnp.asarray(Vq, dtype=jnp.complex64), axes=(0, 1))

    sigma_full = selfenergy_fft(VR, Pj)
    sigma_block = selfenergy_fft(
        VR,
        Pj,
        block_specs=[{"block_sizes": list(block_sizes)}],
        check_offdiag=True,
        hermitian_channel_packing=True,
    )

    np.testing.assert_allclose(
        np.array(sigma_block),
        np.array(sigma_full),
        rtol=1e-6,
        atol=1e-6,
    )


def test_selfenergy_fft_block_specs_falls_back_to_full_when_coupled():
    nk = 4
    n0 = 2
    nb = 2 * n0

    P = jnp.zeros((nk, nk, nb, nb), dtype=jnp.complex64)
    P = P.at[:, :, 0, 0].set(1.0 + 0.0j)
    P = P.at[:, :, 2, 2].set(0.5 + 0.0j)
    # Add a sizable off-block coupling.
    P = P.at[:, :, 0, 2].set(1e-1 + 0.0j)
    P = P.at[:, :, 2, 0].set(1e-1 + 0.0j)

    VR = jnp.ones((nk, nk, 1, 1), dtype=jnp.complex64)
    sigma_full = selfenergy_fft(VR, P)

    # With a tight offdiag tolerance, this must fall back to the full result.
    sigma_auto = selfenergy_fft(
        VR,
        P,
        block_specs=[{"block_sizes": [n0, n0]}],
        check_offdiag=True,
        offdiag_atol=1e-6,
    )
    np.testing.assert_allclose(np.array(sigma_auto), np.array(sigma_full), rtol=1e-6, atol=1e-6)

    # If we force the block path, it should differ from the full answer.
    sigma_forced = selfenergy_fft(
        VR,
        P,
        block_specs=[{"block_sizes": [n0, n0]}],
        check_offdiag=False,
    )
    diff = float(jnp.max(jnp.abs(sigma_forced - sigma_full)))
    assert diff > 1e-3


