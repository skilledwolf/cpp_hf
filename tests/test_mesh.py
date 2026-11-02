"""The one momentum mesh: integer index arithmetic, Γ rules, transfers."""

import numpy as np
import pytest
from hartreefock import Mesh

_B_TRI = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]]) * 2 * np.pi


@pytest.mark.parametrize("nk", [4, 5, 6, 7])
@pytest.mark.parametrize("make", ["square", "bz"])
def test_gamma_rule_is_the_same_for_both_constructors(make, nk):
    build = (
        (lambda **kw: Mesh.square(1.0, nk, **kw))
        if make == "square"
        else (lambda **kw: Mesh.bz(_B_TRI, nk, **kw))
    )
    sym, gam = build(), build(gamma=True)
    # gamma=True always samples Γ at (nk//2, nk//2).
    assert gam.gamma_index == (nk // 2, nk // 2)
    np.testing.assert_allclose(gam.ks[nk // 2, nk // 2], 0.0, atol=1e-14)
    # gamma=False samples Γ exactly when nk is odd, and is closed under k -> -k.
    assert (sym.gamma_index is not None) == (nk % 2 == 1)
    IX, IY, valid = sym.minus_k_index()
    assert np.all(valid)
    np.testing.assert_allclose(sym.ks[IX, IY], -sym.ks, atol=1e-12)


def test_square_patch_matches_cell_centres_and_weight():
    mesh = Mesh.square(0.5, 4)
    np.testing.assert_allclose(mesh.kx, [-0.375, -0.125, 0.125, 0.375])
    assert mesh.weight == pytest.approx(0.25**2 / (2 * np.pi) ** 2)
    assert mesh.weights.sum() * (2 * np.pi) ** 2 == pytest.approx(mesh.area)
    assert not mesh.periodic


@pytest.mark.parametrize("nk", [4, 5])
@pytest.mark.parametrize("gamma", [False, True])
def test_transfer_grid_holds_every_difference_with_q0_at_the_centre(nk, gamma):
    mesh = Mesh.bz(_B_TRI, nk, gamma=gamma)
    q = mesh.transfer_grid
    c = mesh.center_index
    np.testing.assert_allclose(q[c, c], 0.0, atol=1e-14)
    ks = mesh.ks
    # k_i - k_j lands on transfer index (i - j + c) for every pair inside the grid.
    for i, j in [(0, 0), (nk - 1, 0), (1, 2), (2, nk - 1)]:
        d = (i - j + c, i - j + c)
        if 0 <= d[0] < nk:
            np.testing.assert_allclose(ks[i, i] - ks[j, j], q[d], atol=1e-12)


def test_periodic_shift_wraps_and_patch_shift_marks_the_edge():
    bz_mesh = Mesh.bz(_B_TRI, 5)
    IX, IY, valid = bz_mesh.shift_index((1, -2))
    assert np.all(valid)
    assert IX[4, 0] == 0 and IY[0, 0] == 3
    patch = Mesh.square(1.0, 5)
    _, _, valid = patch.shift_index((1, 0))
    assert not np.any(valid[-1]) and np.all(valid[:-1])


def test_oblique_cell_has_no_cartesian_axes_but_interpolates():
    mesh = Mesh.bz(_B_TRI, 6)
    with pytest.raises(ValueError, match="oblique"):
        _ = mesh.kx
    values = mesh.ks[..., 0] + 2 * mesh.ks[..., 1]  # linear in k
    pts = mesh.ks[1:4, 2]
    np.testing.assert_allclose(
        mesh.interpolator(values)(pts), pts[:, 0] + 2 * pts[:, 1], atol=1e-12
    )


def test_resample_keeps_the_gamma_rule():
    mesh = Mesh.square(1.0, 6, gamma=True)
    new, vals = mesh.resample(8, np.ones((6, 6)))
    assert new.gamma_index == (4, 4) and vals.shape == (8, 8)
    assert Mesh.square(1.0, 6).with_nk(8).gamma_index is None


def test_kmeshgrid_axis_order():
    kmax = 1.0
    nk = 3
    kmesh = Mesh.square(kmax, nk)
    ks = kmesh.ks

    assert ks.shape == (nk, nk, 2)
    # axis0 -> kx, axis1 -> ky
    np.testing.assert_allclose(ks[0, 0], [kmesh.kx[0], kmesh.ky[0]])
    np.testing.assert_allclose(ks[-1, 0], [kmesh.kx[-1], kmesh.ky[0]])
    np.testing.assert_allclose(ks[0, -1], [kmesh.kx[0], kmesh.ky[-1]])
    np.testing.assert_allclose(kmesh.bounds, [-kmax, kmax, -kmax, kmax])
    np.testing.assert_allclose(ks[-1, -1], [kmesh.kx[-1], kmesh.ky[-1]])


def test_resample_preserves_axis_order():
    kmesh = Mesh.square(1.0, 4)
    # values(kx, ky) = kx + 2 ky
    values = kmesh.kx[:, None] + 2.0 * kmesh.ky[None, :]
    kmesh_new, values_new = kmesh.resample(5, values)

    assert values_new.shape == (5, 5)
    # Corners preserve orientation
    np.testing.assert_allclose(values_new[0, 0], kmesh_new.kx[0] + 2 * kmesh_new.ky[0])
    np.testing.assert_allclose(
        values_new[-1, -1], kmesh_new.kx[-1] + 2 * kmesh_new.ky[-1]
    )


@pytest.mark.parametrize("nk", [2, 3, 5, 10, 101])
def test_square_mesh_constant_quadrature_matches_requested_domain(nk):
    kmax = 1.7
    mesh = Mesh.square(kmax, nk)

    expected_area = (2.0 * kmax) ** 2
    integrated_area = float(np.sum(mesh.weights)) * (2.0 * np.pi) ** 2

    assert integrated_area == pytest.approx(expected_area, rel=0.0, abs=1e-13)
    assert mesh.area == pytest.approx(expected_area, rel=0.0, abs=1e-13)
    np.testing.assert_allclose(mesh.bounds, [-kmax, kmax, -kmax, kmax])


@pytest.mark.parametrize("nk", [3, 4, 9, 10])
def test_square_mesh_uses_inversion_symmetric_cell_centers(nk):
    mesh = Mesh.square(2.0, nk)

    np.testing.assert_allclose(mesh.kx, -mesh.kx[::-1], atol=1e-15)
    np.testing.assert_allclose(mesh.ky, -mesh.ky[::-1], atol=1e-15)
    np.testing.assert_allclose(mesh.dk, [4.0 / nk, 4.0 / nk], atol=1e-15)
    assert mesh.kx[0] > -2.0
    assert mesh.kx[-1] < 2.0


def test_square_mesh_midpoint_quadrature_converges_for_smooth_function():
    # Integral of kx^2 + ky^2 over [-1, 1]^2 is 8/3.
    errors = []
    for nk in (4, 8, 16, 32):
        mesh = Mesh.square(1.0, nk)
        values = mesh.ks[..., 0] ** 2 + mesh.ks[..., 1] ** 2
        actual = float(np.sum(mesh.weights * values)) * (2.0 * np.pi) ** 2
        errors.append(abs(actual - 8.0 / 3.0))

    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 3e-3
