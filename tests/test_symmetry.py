"""Symmetry projector tests (port of jax_hf.tests.test_symmetry)."""

from __future__ import annotations

import numpy as np

from cpp_hf.symmetry import make_project_fn


def _seeded_complex(rng, shape):
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return a.astype(np.complex64)


def test_make_project_fn_identity_when_no_args():
    proj = make_project_fn()
    A = np.ones((2, 2, 3, 3), dtype=np.complex64)
    out = proj(A)
    np.testing.assert_array_equal(np.array(A), np.array(out))


def test_unitary_group_averaging_produces_invariant_output():
    nb = 4
    eye = np.eye(nb, dtype=np.complex64)
    g = np.diag(np.array([1, 1, -1, -1], dtype=np.complex64))
    G = np.stack([eye, g], axis=0)

    proj = make_project_fn(unitary_group=G)

    rng = np.random.default_rng(42)
    A = _seeded_complex(rng, (3, 3, nb, nb))

    A_proj = proj(A)

    for i in range(G.shape[0]):
        gi = G[i]
        giH = np.conj(gi.T)
        rotated = (gi @ A_proj) @ giH
        np.testing.assert_allclose(
            np.array(rotated), np.array(A_proj), atol=1e-6, rtol=1e-6
        )


def test_unitary_group_averaging_is_hermitian_preserving():
    nb = 3
    eye = np.eye(nb, dtype=np.complex64)
    perm = np.zeros((nb, nb), dtype=np.complex64)
    perm[0, 1] = perm[1, 2] = perm[2, 0] = 1.0
    perm2 = perm @ perm
    G = np.stack([eye, perm, perm2], axis=0)

    proj = make_project_fn(unitary_group=G)

    rng = np.random.default_rng(99)
    A = _seeded_complex(rng, (2, 2, nb, nb))
    A_herm = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))

    result = proj(A_herm)
    np.testing.assert_allclose(
        np.array(result),
        np.array(np.conj(np.swapaxes(result, -1, -2))),
        atol=1e-6,
    )


def test_time_reversal_averaging_flip():
    nk1, nk2, nb = 4, 4, 2
    U_tr = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)

    proj = make_project_fn(
        time_reversal_U=U_tr,
        time_reversal_k_convention="flip",
    )

    rng = np.random.default_rng(55)
    A = _seeded_complex(rng, (nk1, nk2, nb, nb))

    A_proj = proj(A)

    UH = np.conj(U_tr.T)
    A_neg = np.flip(A_proj, axis=(0, 1))
    A_tr = U_tr @ np.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-6, rtol=1e-6
    )


def test_time_reversal_averaging_mod():
    nk1, nk2, nb = 5, 5, 2
    U_tr = np.eye(nb, dtype=np.complex64)

    proj = make_project_fn(
        time_reversal_U=U_tr,
        time_reversal_k_convention="mod",
    )

    rng = np.random.default_rng(77)
    A = _seeded_complex(rng, (nk1, nk2, nb, nb))

    A_proj = proj(A)

    UH = np.conj(U_tr.T)
    i_idx = (-np.arange(nk1, dtype=np.int32)) % nk1
    j_idx = (-np.arange(nk2, dtype=np.int32)) % nk2
    A_neg = A_proj[i_idx[:, None], j_idx[None, :], ...]
    A_tr = U_tr @ np.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-6, rtol=1e-6
    )


def test_combined_unitary_and_time_reversal():
    nk1, nk2, nb = 4, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_z = np.diag(np.array([1, -1], dtype=np.complex64))
    G = np.stack([eye, sigma_z], axis=0)
    U_tr = np.eye(nb, dtype=np.complex64)

    proj = make_project_fn(
        unitary_group=G,
        time_reversal_U=U_tr,
        time_reversal_k_convention="flip",
    )

    rng = np.random.default_rng(11)
    A = _seeded_complex(rng, (nk1, nk2, nb, nb))

    A_proj = proj(A)

    for i in range(G.shape[0]):
        gi = G[i]
        giH = np.conj(gi.T)
        rotated = (gi @ A_proj) @ giH
        np.testing.assert_allclose(
            np.array(rotated), np.array(A_proj), atol=1e-5, rtol=1e-5
        )

    UH = np.conj(U_tr.T)
    A_neg = np.flip(A_proj, axis=(0, 1))
    A_tr = U_tr @ np.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-5, rtol=1e-5
    )


def test_projection_is_idempotent():
    nb = 3
    eye = np.eye(nb, dtype=np.complex64)
    perm = np.zeros((nb, nb), dtype=np.complex64)
    perm[0, 1] = perm[1, 2] = perm[2, 0] = 1.0
    perm2 = perm @ perm
    G = np.stack([eye, perm, perm2], axis=0)

    proj = make_project_fn(unitary_group=G)

    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 2, nb, nb)).astype(np.float32).astype(np.complex64)

    once = proj(A)
    twice = proj(once)
    np.testing.assert_allclose(np.array(once), np.array(twice), atol=1e-6)


def test_spatial_group_produces_k_flip_invariant_output():
    nk1, nk2, nb = 4, 4, 2
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    S = sigma_x[None]

    proj = make_project_fn(spatial_group=S, spatial_k_convention="flip")

    rng = np.random.default_rng(33)
    A = rng.normal(size=(nk1, nk2, nb, nb)).astype(np.complex64)

    A_proj = proj(A)

    sxH = np.conj(sigma_x.T)
    A_neg = np.flip(A_proj, axis=(0, 1))
    A_rotated = sigma_x @ A_neg @ sxH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_rotated), atol=1e-6, rtol=1e-6
    )


def test_combined_same_k_and_flip_k_group():
    nk1, nk2, nb = 4, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_z = np.diag(np.array([1, -1], dtype=np.complex64))
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)

    G_same = np.stack([eye, sigma_z], axis=0)
    G_flip = sigma_x[None]

    proj = make_project_fn(
        unitary_group=G_same,
        spatial_group=G_flip,
        spatial_k_convention="flip",
    )

    rng = np.random.default_rng(44)
    A = rng.normal(size=(nk1, nk2, nb, nb)).astype(np.complex64)
    A_proj = proj(A)

    A_neg = np.flip(A, axis=(0, 1))
    expected = (
        A
        + sigma_z @ A @ np.conj(sigma_z.T)
        + sigma_x @ A_neg @ np.conj(sigma_x.T)
    ) / 3.0
    np.testing.assert_allclose(
        np.array(A_proj), np.array(expected), atol=1e-6, rtol=1e-6
    )


def test_kx_only_flip_is_idempotent():
    nk1, nk2, nb = 6, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    G_same = eye[None]
    G_flip = sigma_x[None]

    proj = make_project_fn(
        unitary_group=G_same,
        spatial_group=G_flip,
        spatial_k_convention="flip",
        spatial_k_flip_axes=(0,),
    )

    rng = np.random.default_rng(66)
    A = rng.normal(size=(nk1, nk2, nb, nb)).astype(np.complex64)

    once = proj(A)
    twice = proj(once)
    np.testing.assert_allclose(np.array(once), np.array(twice), atol=1e-5)
