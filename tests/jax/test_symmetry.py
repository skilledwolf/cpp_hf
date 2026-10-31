from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from hartreefock.backends.jax.symmetry import make_project_fn


def test_make_project_fn_identity_when_no_args():
    proj = make_project_fn()
    A = jnp.ones((2, 2, 3, 3), dtype=jnp.complex64)
    out = proj(A)
    np.testing.assert_array_equal(np.array(A), np.array(out))


def test_unitary_group_averaging_produces_invariant_output():
    nb = 4
    eye = np.eye(nb, dtype=np.complex64)
    g = np.diag(np.array([1, 1, -1, -1], dtype=np.complex64))
    G = jnp.stack([jnp.asarray(eye), jnp.asarray(g)], axis=0)

    proj = make_project_fn(unitary_group=G)

    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, (3, 3, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(7), (3, 3, nb, nb))).astype(
        jnp.complex64
    )

    A_proj = proj(A)

    for i in range(G.shape[0]):
        gi = G[i]
        giH = jnp.conj(gi.T)
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
    G = jnp.stack([jnp.asarray(eye), jnp.asarray(perm), jnp.asarray(perm2)], axis=0)

    proj = make_project_fn(unitary_group=G)

    key = jax.random.PRNGKey(99)
    a = jax.random.normal(key, (2, 2, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(100), (2, 2, nb, nb))).astype(
        jnp.complex64
    )
    A_herm = 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))

    result = proj(A_herm)
    np.testing.assert_allclose(
        np.array(result),
        np.array(jnp.conj(jnp.swapaxes(result, -1, -2))),
        atol=1e-6,
    )


def test_time_reversal_averaging_flip():
    nk1, nk2, nb = 4, 4, 2
    U_tr = jnp.asarray(np.array([[0, -1j], [1j, 0]], dtype=np.complex64))

    proj = make_project_fn(
        time_reversal_U=U_tr,
        time_reversal_k_convention="flip",
    )

    key = jax.random.PRNGKey(55)
    a = jax.random.normal(key, (nk1, nk2, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(56), (nk1, nk2, nb, nb))).astype(
        jnp.complex64
    )

    A_proj = proj(A)

    UH = jnp.conj(U_tr.T)
    A_neg = jnp.flip(A_proj, axis=(0, 1))
    A_tr = U_tr @ jnp.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-6, rtol=1e-6
    )


def test_time_reversal_averaging_mod():
    nk1, nk2, nb = 5, 5, 2
    U_tr = jnp.eye(nb, dtype=jnp.complex64)

    proj = make_project_fn(
        time_reversal_U=U_tr,
        time_reversal_k_convention="mod",
    )

    key = jax.random.PRNGKey(77)
    a = jax.random.normal(key, (nk1, nk2, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(78), (nk1, nk2, nb, nb))).astype(
        jnp.complex64
    )

    A_proj = proj(A)

    UH = jnp.conj(U_tr.T)
    i_idx = (-jnp.arange(nk1, dtype=jnp.int32)) % nk1
    j_idx = (-jnp.arange(nk2, dtype=jnp.int32)) % nk2
    A_neg = A_proj[i_idx[:, None], j_idx[None, :], ...]
    A_tr = U_tr @ jnp.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-6, rtol=1e-6
    )


def test_combined_unitary_and_time_reversal():
    nk1, nk2, nb = 4, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_z = np.diag(np.array([1, -1], dtype=np.complex64))
    G = jnp.stack([jnp.asarray(eye), jnp.asarray(sigma_z)], axis=0)
    U_tr = jnp.eye(nb, dtype=jnp.complex64)

    proj = make_project_fn(
        unitary_group=G,
        time_reversal_U=U_tr,
        time_reversal_k_convention="flip",
    )

    key = jax.random.PRNGKey(11)
    a = jax.random.normal(key, (nk1, nk2, nb, nb))
    A = (a + 1j * jax.random.normal(jax.random.PRNGKey(12), (nk1, nk2, nb, nb))).astype(
        jnp.complex64
    )

    A_proj = proj(A)

    for i in range(G.shape[0]):
        gi = G[i]
        giH = jnp.conj(gi.T)
        rotated = (gi @ A_proj) @ giH
        np.testing.assert_allclose(
            np.array(rotated), np.array(A_proj), atol=1e-5, rtol=1e-5
        )

    UH = jnp.conj(U_tr.T)
    A_neg = jnp.flip(A_proj, axis=(0, 1))
    A_tr = U_tr @ jnp.conj(A_neg) @ UH
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_tr), atol=1e-5, rtol=1e-5
    )


def test_projection_is_idempotent():
    nb = 3
    eye = np.eye(nb, dtype=np.complex64)
    perm = np.zeros((nb, nb), dtype=np.complex64)
    perm[0, 1] = perm[1, 2] = perm[2, 0] = 1.0
    perm2 = perm @ perm
    G = jnp.stack([jnp.asarray(eye), jnp.asarray(perm), jnp.asarray(perm2)], axis=0)

    proj = make_project_fn(unitary_group=G)

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (2, 2, nb, nb), dtype=jnp.float32).astype(
        jnp.complex64
    )

    once = proj(A)
    twice = proj(once)
    np.testing.assert_allclose(np.array(once), np.array(twice), atol=1e-6)


def test_spatial_group_produces_k_flip_invariant_output():
    nk1, nk2, nb = 4, 4, 2
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    S = jnp.asarray(sigma_x)[None]

    proj = make_project_fn(spatial_group=S, spatial_k_convention="flip")

    key = jax.random.PRNGKey(33)
    A = jax.random.normal(key, (nk1, nk2, nb, nb)).astype(jnp.complex64)

    A_proj = proj(A)

    sxH = jnp.conj(sigma_x.T)
    A_neg = jnp.flip(A_proj, axis=(0, 1))
    A_rotated = jnp.asarray(sigma_x) @ A_neg @ jnp.asarray(sxH)
    np.testing.assert_allclose(
        np.array(A_proj), np.array(A_rotated), atol=1e-6, rtol=1e-6
    )


def test_combined_same_k_and_flip_k_group():
    nk1, nk2, nb = 4, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_z = np.diag(np.array([1, -1], dtype=np.complex64))
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)

    G_same = jnp.stack([jnp.asarray(eye), jnp.asarray(sigma_z)], axis=0)
    G_flip = jnp.asarray(sigma_x)[None]

    proj = make_project_fn(
        unitary_group=G_same,
        spatial_group=G_flip,
        spatial_k_convention="flip",
    )

    key = jax.random.PRNGKey(44)
    A = jax.random.normal(key, (nk1, nk2, nb, nb)).astype(jnp.complex64)
    A_proj = proj(A)

    A_neg = jnp.flip(A, axis=(0, 1))
    expected = (
        A
        + jnp.asarray(sigma_z) @ A @ jnp.conj(jnp.asarray(sigma_z).T)
        + jnp.asarray(sigma_x) @ A_neg @ jnp.conj(jnp.asarray(sigma_x).T)
    ) / 3.0
    np.testing.assert_allclose(
        np.array(A_proj), np.array(expected), atol=1e-6, rtol=1e-6
    )


def test_kx_only_flip_is_idempotent():
    nk1, nk2, nb = 6, 4, 2
    eye = np.eye(nb, dtype=np.complex64)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    G_same = jnp.asarray(eye)[None]
    G_flip = jnp.asarray(sigma_x)[None]

    proj = make_project_fn(
        unitary_group=G_same,
        spatial_group=G_flip,
        spatial_k_convention="flip",
        spatial_k_flip_axes=(0,),
    )

    key = jax.random.PRNGKey(66)
    A = jax.random.normal(key, (nk1, nk2, nb, nb)).astype(jnp.complex64)

    once = proj(A)
    twice = proj(once)
    np.testing.assert_allclose(np.array(once), np.array(twice), atol=1e-5)




# ----------------------------------------------------------------------
# Index-map (continuum-patch) spatial path
# ----------------------------------------------------------------------

def _make_c3_index_maps(nk1: int, nk2: int):
    """Build periodic-mod-nk C3 index maps for testing (no continuum patch).

    On the centred fractional grid ``f_i = i/nk - 0.5``, the rotation
    ``(f_x, f_y) -> (-f_y, f_x - f_y)`` maps integer indices via
    (-jj % nk, (ii - jj + nk_half) % nk).
    """
    if nk1 != nk2:
        raise ValueError("test helper assumes square grid")
    nk = nk1
    half = nk // 2
    ii, jj = np.meshgrid(np.arange(nk), np.arange(nk), indexing="ij")
    perm_i_R = (-jj) % nk
    perm_j_R = (ii - jj + half) % nk
    perm_i_R2 = (jj - ii + half) % nk
    perm_j_R2 = (-ii) % nk
    idx_i = np.stack([ii, perm_i_R, perm_i_R2], axis=0).astype(np.int32)
    idx_j = np.stack([jj, perm_j_R, perm_j_R2], axis=0).astype(np.int32)
    return idx_i, idx_j


def test_spatial_k_index_maps_path_is_idempotent_and_invariant():
    """Projector built from spatial_k_index_maps is a projection: P(P(A)) = P(A)."""
    nk = 6
    nb = 4
    # Diagonal cube-root-of-identity orbital unitary
    omega = np.exp(2j * np.pi / 3.0)
    U = np.diag([omega**0, omega**1, omega**2, omega**0]).astype(np.complex128)
    S = np.stack([np.eye(nb, dtype=np.complex128), U, U @ U], axis=0)
    idx_i, idx_j = _make_c3_index_maps(nk, nk)

    proj = make_project_fn(
        spatial_group=jnp.asarray(S),
        spatial_k_index_maps=(jnp.asarray(idx_i), jnp.asarray(idx_j)),
    )

    rng = np.random.default_rng(11)
    A = (rng.standard_normal((nk, nk, nb, nb))
         + 1j * rng.standard_normal((nk, nk, nb, nb))).astype(np.complex128)
    A_jax = jnp.asarray(A)

    once = proj(A_jax)
    twice = proj(once)
    # Default jax dtype is float32 unless jax_enable_x64; allow that tolerance.
    atol = 1e-5
    np.testing.assert_allclose(np.array(twice), np.array(once), atol=atol)

    # Symmetric output: applying any element of the spatial group to the
    # projected matrix (with the matching k-permutation) leaves it unchanged.
    once_np = np.asarray(once)
    for g in (1, 2):
        rotated = np.array([
            S[g] @ once_np[idx_i[g, i, j], idx_j[g, i, j]] @ np.conj(S[g].T)
            for i in range(nk) for j in range(nk)
        ]).reshape(nk, nk, nb, nb)
        np.testing.assert_allclose(rotated, once_np, atol=atol)


def test_spatial_k_index_maps_requires_spatial_group():
    import pytest

    nk = 4
    idx_i, idx_j = _make_c3_index_maps(nk, nk)
    with pytest.raises(ValueError, match="spatial_group"):
        make_project_fn(
            spatial_k_index_maps=(jnp.asarray(idx_i), jnp.asarray(idx_j)),
        )


def test_spatial_k_index_maps_dim_mismatch_raises():
    import pytest

    nk = 4
    nb = 2
    S = np.stack([np.eye(nb, dtype=np.complex128)] * 3, axis=0)
    # only 2 elements in idx, but spatial_group has 3
    idx_i = np.zeros((2, nk, nk), dtype=np.int32)
    idx_j = np.zeros((2, nk, nk), dtype=np.int32)
    with pytest.raises(ValueError, match="leading dim"):
        make_project_fn(
            spatial_group=jnp.asarray(S),
            spatial_k_index_maps=(jnp.asarray(idx_i), jnp.asarray(idx_j)),
        )


def test_time_reversal_k_index_map_path_with_invalid_mask():
    """TR with a per-k index map and a partial validity mask leaves invalid points alone."""
    nk = 4
    nb = 2
    U_tr = np.eye(nb, dtype=np.complex128)
    # Index map: standard mod-nk (-k) flip, but mask out the (0,0) corner
    ii, jj = np.meshgrid(np.arange(nk), np.arange(nk), indexing="ij")
    idx_i = (-ii) % nk
    idx_j = (-jj) % nk
    valid = np.ones((nk, nk), dtype=bool)
    valid[0, 0] = False  # leave (0,0) untouched

    proj = make_project_fn(
        time_reversal_U=jnp.asarray(U_tr),
        time_reversal_k_index_map=(
            jnp.asarray(idx_i), jnp.asarray(idx_j), jnp.asarray(valid),
        ),
    )

    rng = np.random.default_rng(3)
    A = (rng.standard_normal((nk, nk, nb, nb))
         + 1j * rng.standard_normal((nk, nk, nb, nb))).astype(np.complex128)

    out = np.asarray(proj(jnp.asarray(A)))
    # (0,0) was masked → output equals input there
    np.testing.assert_allclose(out[0, 0], A[0, 0], atol=1e-12)
    # (1,2) was averaged with conj(A[(-1) % nk, (-2) % nk]) = conj(A[3, 2])
    expected = 0.5 * (A[1, 2] + np.conj(A[3, 2]))
    np.testing.assert_allclose(out[1, 2], expected, atol=1e-12)


def _random_unitary_group(rng, nb, n):
    """``n`` random unitaries (not a group; the average is still a well-defined linear map)."""
    mats = []
    for _ in range(n):
        Z = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        Q, _ = np.linalg.qr(Z)
        mats.append(Q)
    return np.stack(mats)


def _loop_average(A, G):
    return np.mean([g @ A @ np.conj(g.T) for g in G], axis=0)


def test_gemm_group_average_matches_the_conjugation_loop(enable_x64):
    """Below _GEMM_MAX_NB the unitary average is one precomputed GEMM; it must
    reproduce the per-element conjugation loop to rounding, and the loop must
    still be selected above the threshold."""
    with enable_x64():
        from hartreefock.backends.jax import symmetry as sym

        rng = np.random.default_rng(0)
        for nb in (6, sym._GEMM_MAX_NB + 2):
            G = _random_unitary_group(rng, nb, 4)
            A = rng.normal(size=(3, 2, nb, nb)) + 1j * rng.normal(size=(3, 2, nb, nb))
            pf = make_project_fn(unitary_group=jnp.asarray(G))
            out = np.asarray(jax.jit(pf)(jnp.asarray(A)))
            ref = np.stack([[_loop_average(A[i, j], G) for j in range(2)] for i in range(3)])
            np.testing.assert_allclose(out, ref, atol=1e-13, rtol=0)
        M = sym._conj_sum_matrix(jnp.asarray(G[:1]))
        assert M.shape == (nb * nb, nb * nb)


def test_gemm_combined_group_matches_loop_form(enable_x64):
    with enable_x64():
        from hartreefock.backends.jax import symmetry as sym

        rng = np.random.default_rng(1)
        nb = 4
        G = _random_unitary_group(rng, nb, 3)
        S = _random_unitary_group(rng, nb, 2)
        A = rng.normal(size=(4, 4, nb, nb)) + 1j * rng.normal(size=(4, 4, nb, nb))
        Aj = jnp.asarray(A)
        fast = sym._avg_combined_group(
            Aj, jnp.asarray(G), jnp.asarray(S), "mod", (0, 1),
            M_same=sym._conj_sum_matrix(jnp.asarray(G)), M_flip=sym._conj_sum_matrix(jnp.asarray(S)),
        )
        slow = sym._avg_combined_group(Aj, jnp.asarray(G), jnp.asarray(S), "mod", (0, 1))
        np.testing.assert_allclose(np.asarray(fast), np.asarray(slow), atol=1e-13, rtol=0)


def test_projector_follows_the_density_dtype():
    """A complex128 group applied to a complex64 density must not promote it
    (with x64 enabled this used to raise a fori_loop carry-dtype error)."""
    from hartreefock.backends.jax import symmetry as sym

    rng = np.random.default_rng(2)
    nb = 4
    G = _random_unitary_group(rng, nb, 2).astype(np.complex128)
    U = _random_unitary_group(rng, nb, 1)[0].astype(np.complex128)
    A = (rng.normal(size=(2, 2, nb, nb)) + 1j * rng.normal(size=(2, 2, nb, nb))).astype(np.complex64)
    for kwargs in (
        dict(unitary_group=jnp.asarray(G)),
        dict(time_reversal_U=jnp.asarray(U), time_reversal_k_convention="mod"),
        dict(spatial_group=jnp.asarray(G)),
    ):
        pf = make_project_fn(**kwargs)
        out = jax.jit(pf)(jnp.asarray(A))
        assert out.dtype == jnp.complex64, (kwargs.keys(), out.dtype)
    # the loop form (above the GEMM threshold) has the same contract
    nb_big = sym._GEMM_MAX_NB + 1
    Gb = _random_unitary_group(rng, nb_big, 2).astype(np.complex128)
    Ab = (rng.normal(size=(1, 1, nb_big, nb_big)) + 0j).astype(np.complex64)
    assert jax.jit(make_project_fn(unitary_group=jnp.asarray(Gb)))(jnp.asarray(Ab)).dtype == jnp.complex64


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("gemm_threshold", [0, 24])
@pytest.mark.parametrize("kind", ["unitary", "spatial", "spatial_map", "time_reversal", "time_reversal_map"])
def test_complex_symmetry_promotes_real_density_without_losing_operator(
    dtype, gemm_threshold, kind, monkeypatch, enable_x64,
):
    with enable_x64(dtype == jnp.float64):
        import hartreefock.backends.jax.symmetry as module
        monkeypatch.setattr(module, "_GEMM_MAX_NB", gemm_threshold, raising=False)
        sigma_y = np.array([[0.0, -1j], [1j, 0.0]])
        unitary = (np.diag([1.0, -1.0]) + sigma_y) / np.sqrt(2.0)
        group = jnp.asarray([np.eye(2), unitary])
        index = jnp.zeros((1, 1), dtype=jnp.int32)
        kwargs = {
            "unitary": dict(unitary_group=group),
            "spatial": dict(spatial_group=group[1:]),
            "spatial_map": dict(spatial_group=group, spatial_k_index_maps=(
                jnp.stack([index, index]), jnp.stack([index, index]))),
            "time_reversal": dict(time_reversal_U=jnp.asarray(sigma_y)),
            "time_reversal_map": dict(time_reversal_U=jnp.asarray(sigma_y),
                time_reversal_k_index_map=(index, index, jnp.ones((1, 1), dtype=bool))),
        }[kind]
        density = jnp.diag(jnp.asarray([0.7, 0.3], dtype=dtype))[None, None]
        op = sigma_y if kind.startswith("time_reversal") else unitary
        expected = 0.5 * (np.asarray(density) + op @ np.asarray(density) @ op.conj().T)
        projected = jax.jit(make_project_fn(**kwargs))(density)
        np.testing.assert_allclose(projected, expected, atol=1e-7, rtol=0.0)
        assert projected.dtype == jnp.result_type(dtype, jnp.complex64)
