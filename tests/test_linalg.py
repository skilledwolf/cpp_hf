import jax.numpy as jnp
import numpy as np
import pytest

import hartreefock as hf
from hartreefock._jax.linalg import eigh
from hartreefock._jax.utils import selfenergy_fft

SECTORS = ((0, 2, 5), (1, 3, 4))


def _sector_hermitian(rng, shape, sectors):
    n = sum(len(s) for s in sectors)
    out = np.zeros(shape + (n, n), dtype=np.complex128)
    for idx in sectors:
        m = len(idx)
        raw = rng.normal(size=shape + (m, m)) + 1j * rng.normal(size=shape + (m, m))
        i = np.asarray(idx)
        out[..., i[:, None], i[None, :]] = raw + np.conj(np.swapaxes(raw, -1, -2))
    return out


def test_sector_eigh_matches_full():
    h = _sector_hermitian(np.random.default_rng(0), (3,), SECTORS)
    w, v = (np.asarray(x) for x in eigh(jnp.asarray(h), SECTORS))
    np.testing.assert_allclose(w, np.linalg.eigvalsh(h), atol=1e-10)
    np.testing.assert_allclose(v @ (w[..., None] * np.conj(np.swapaxes(v, -1, -2))), h, atol=1e-10)


@pytest.mark.parametrize("packing", [False, True])
def test_sector_selfenergy_matches_full(packing):
    rng = np.random.default_rng(7)
    P = jnp.asarray(_sector_hermitian(rng, (8, 8), SECTORS))
    VR = jnp.fft.fftn(jnp.asarray(rng.normal(size=(8, 8, 1, 1)), dtype=jnp.complex128), axes=(0, 1))
    full = selfenergy_fft(VR, P, hermitian_channel_packing=packing)
    sector = selfenergy_fft(VR, P, sectors=SECTORS, hermitian_channel_packing=packing)
    np.testing.assert_allclose(np.asarray(sector), np.asarray(full), atol=1e-10)


def test_selfenergy_fft_hermitian_channel_packing_matches_full_for_scalar_real_kernel():
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(8, 8, 6, 6)) + 1j * rng.normal(size=(8, 8, 6, 6))
    P = jnp.asarray((0.5 * (raw + np.conj(np.swapaxes(raw, -1, -2)))).astype(np.complex64))
    VR = jnp.fft.fftn(jnp.asarray(rng.normal(size=(8, 8, 1, 1)), dtype=jnp.complex64), axes=(0, 1))
    np.testing.assert_allclose(
        np.asarray(selfenergy_fft(VR, P, hermitian_channel_packing=True)),
        np.asarray(selfenergy_fft(VR, P)),
        rtol=1e-6,
        atol=1e-6,
    )


def _valley_kernel(sectors=None):
    """Two copies of a two-band model, sector (0, 1) and sector (2, 3)."""
    mesh = hf.Mesh.square(kmax=1.0, nk=4)
    k2 = np.sum(mesh.ks**2, axis=-1)
    h = np.zeros(mesh.shape + (4, 4), dtype=np.complex128)
    for a, b, shift in ((0, 1, 0.0), (2, 3, 0.3)):
        h[..., a, a] = k2 - 1.0 + shift
        h[..., b, b] = -k2 + 1.0
        h[..., a, b] = h[..., b, a] = 0.2
    return hf.HartreeFockKernel(mesh, h, lambda q: 2.0 / (q + 0.5), T=0.05, sectors=sectors)


@pytest.mark.parametrize("method", ["cg", "rdiis"])
def test_sector_solve_matches_full_solve(method):
    config = hf.SolverConfig(method=method, max_iter=300)
    n = 2.0 * float(_valley_kernel().weight_sum)
    full = hf.solve(_valley_kernel(), None, n, config=config)
    split = hf.solve(_valley_kernel(((0, 1), (2, 3))), None, n, config=config)
    assert full.converged and split.converged
    assert split.energy == pytest.approx(full.energy, abs=1e-6)
    np.testing.assert_allclose(split.density[..., :2, 2:], 0.0, atol=1e-12)


def test_kernel_rejects_sectors_that_h_couples():
    with pytest.raises(ValueError, match="h couples the sectors"):
        _valley_kernel(((0, 2), (1, 3)))
