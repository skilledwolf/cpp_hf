"""Symmetry projector tests (port of jax_hf.tests.test_symmetry)."""

from __future__ import annotations

import numpy as np
import pytest

from hartreefock.symmetry import make_project_fn


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


def _random_unitaries(rng, nb, n):
    mats = []
    for _ in range(n):
        Z = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        Q, _ = np.linalg.qr(Z)
        mats.append(Q)
    return np.stack(mats)


def _loop_average(A, G):
    return np.mean([g @ A @ np.conj(g.T) for g in G], axis=0)


def test_gemm_group_average_matches_the_conjugation_loop():
    """Below _GEMM_MAX_NB the unitary average is one precomputed GEMM on vec(A);
    it must reproduce the per-element conjugation loop to rounding, and the
    loop must still be selected above the threshold."""
    from hartreefock import symmetry as sym

    rng = np.random.default_rng(0)
    saved = sym._GEMM_MAX_NB
    try:
        # (threshold, nb): GEMM path, loop path above the threshold, loop path (default off)
        for threshold, nb in ((24, 6), (24, 26), (0, 6)):
            sym._GEMM_MAX_NB = threshold
            G = _random_unitaries(rng, nb, 4)
            A = rng.normal(size=(3, 2, nb, nb)) + 1j * rng.normal(size=(3, 2, nb, nb))
            out = make_project_fn(unitary_group=G)(A)
            ref = np.stack([[_loop_average(A[i, j], G) for j in range(2)] for i in range(3)])
            np.testing.assert_allclose(out, ref, atol=1e-13, rtol=0)
    finally:
        sym._GEMM_MAX_NB = saved
    M = sym._conj_sum_matrix(G[:1])
    assert M.shape == (nb * nb, nb * nb)


def test_gemm_combined_group_matches_loop_form():
    from hartreefock import symmetry as sym

    rng = np.random.default_rng(1)
    nb = 4
    G = _random_unitaries(rng, nb, 3)
    S = _random_unitaries(rng, nb, 2)
    A = rng.normal(size=(4, 4, nb, nb)) + 1j * rng.normal(size=(4, 4, nb, nb))
    fast = sym._avg_combined_group(
        np, A, G, S, "mod", (0, 1), M_same=sym._conj_sum_matrix(G), M_flip=sym._conj_sum_matrix(S),
    )
    slow = sym._avg_combined_group(np, A, G, S, "mod", (0, 1))
    np.testing.assert_allclose(fast, slow, atol=1e-13, rtol=0)


def test_projector_follows_the_density_dtype():
    """A complex128 group applied to a complex64 density returns complex64."""
    from hartreefock import symmetry as sym

    rng = np.random.default_rng(2)
    nb = 4
    G = _random_unitaries(rng, nb, 2)
    U = _random_unitaries(rng, nb, 1)[0]
    A = (rng.normal(size=(2, 2, nb, nb)) + 1j * rng.normal(size=(2, 2, nb, nb))).astype(np.complex64)
    for kwargs in (
        dict(unitary_group=G),
        dict(time_reversal_U=U, time_reversal_k_convention="mod"),
        dict(spatial_group=G),
    ):
        assert make_project_fn(**kwargs)(A).dtype == np.complex64, kwargs.keys()
    nb_big = max(sym._GEMM_MAX_NB, 4) + 1
    Gb = _random_unitaries(rng, nb_big, 2)
    Ab = (rng.normal(size=(1, 1, nb_big, nb_big)) + 0j).astype(np.complex64)
    assert make_project_fn(unitary_group=Gb)(Ab).dtype == np.complex64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("symmetry", ["unitary", "spatial", "time_reversal"])
def test_complex_symmetry_preserves_real_density_trace(dtype, symmetry):
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]])
    kwargs = {
        "unitary": {"unitary_group": np.array([np.eye(2), sigma_y])},
        "spatial": {"spatial_group": sigma_y[None]},
        "time_reversal": {"time_reversal_U": sigma_y},
    }[symmetry]
    project = make_project_fn(**kwargs)
    density = np.diag([0.7, 0.3]).astype(dtype)[None, None]
    projected = project(density)
    np.testing.assert_allclose(projected[0, 0], np.eye(2) / 2, atol=1e-7)
    np.testing.assert_allclose(project(projected), projected, atol=1e-7)
    assert projected.dtype == np.result_type(dtype, np.complex64)
