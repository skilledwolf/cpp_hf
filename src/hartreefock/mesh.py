"""One uniform 2D momentum mesh: an ``nk x nk`` grid on a parallelogram cell.

Every mesh point is an integer index pair ``(i, j)`` with fractional
coordinate ``((i, j) - nk // 2 + offset) / nk`` and Cartesian momentum
``k = cell @ frac``.  Two numbers fix the grid:

``offset``
    ``0.0`` puts Γ at index ``(nk // 2, nk // 2)``.  ``0.5`` shifts the grid
    by half a step.  :meth:`Mesh.square` and :meth:`Mesh.bz` choose it with
    ``gamma``: ``gamma=True`` always samples Γ; ``gamma=False`` (default)
    gives the grid that is closed under ``k -> -k`` (offset ``0.5`` for even
    ``nk``, ``0`` for odd), so Γ is sampled only for odd ``nk``.
``periodic``
    ``True`` for a Brillouin zone (the grid is a torus), ``False`` for a
    continuum patch.

Because every momentum is ``cell @ (index + offset) / nk``, differences
``k - k'`` are integer multiples of the steps: the transfer grid
(:attr:`Mesh.transfer_grid`) holds them with ``q = 0`` at index
``(nk // 2, nk // 2)`` for every ``nk`` and ``offset``, and ``-k`` / ``k + q``
lookups are integer operations (:meth:`Mesh.pair_index`,
:meth:`Mesh.shift_index`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _symmetric_offset(nk: int) -> float:
    return 0.5 if nk % 2 == 0 else 0.0


@dataclass(frozen=True, eq=False)
class Mesh:
    """Uniform ``nk x nk`` momentum mesh on the cell spanned by the columns of ``cell``.

    Build one with :meth:`square` (a continuum patch ``[-kmax, kmax]^2``) or
    :meth:`bz` (a Brillouin zone).  Arrays sampled on the mesh have leading
    axes ``(nk, nk)``: axis 0 runs along ``cell[:, 0]``, axis 1 along
    ``cell[:, 1]``.

    A k-integral is ``sum(weights * f)``, with
    ``weights = |det cell| / nk² / (2π)²``.  Interaction kernels are evaluated
    on the momentum transfers ``transfer_grid`` (``q = 0`` at index
    ``(nk//2, nk//2)``), which differ from the k points when ``k = 0`` is not
    on the mesh.

    Attributes
    ----------
    cell : (2, 2) array
        The spanning vectors, as columns.  For a Brillouin zone these are
        the reciprocal lattice vectors.
    nk : int
        Points per direction.
    offset : float
        Shift of the grid in units of the spacing (0 puts ``k = 0`` on it).
    periodic : bool
        Whether the cell is a Brillouin zone (wraps around).
    """

    cell: np.ndarray
    nk: int
    offset: float = 0.0
    periodic: bool = False

    def __post_init__(self) -> None:
        cell = np.array(self.cell, dtype=float)
        if cell.shape != (2, 2) or not np.all(np.isfinite(cell)):
            raise ValueError("cell must be a finite 2x2 matrix")
        if abs(np.linalg.det(cell)) <= 0.0:
            raise ValueError("cell must be non-singular")
        cell.setflags(write=False)
        object.__setattr__(self, "cell", cell)
        nk = int(self.nk)
        if nk < 1:
            raise ValueError("nk must be positive")
        object.__setattr__(self, "nk", nk)
        offset = float(self.offset)
        if offset not in (0.0, 0.5):
            raise ValueError("offset must be 0.0 or 0.5")
        object.__setattr__(self, "offset", offset)

    # -- constructors ------------------------------------------------------------
    @classmethod
    def square(cls, kmax: float, nk: int, *, gamma: bool = False) -> Mesh:
        """The continuum patch ``[-kmax, kmax]^2`` (not periodic).

        ``gamma=False`` gives cell-centred points, inversion symmetric; Γ is
        sampled for odd ``nk`` only.  ``gamma=True`` always samples Γ (for even
        ``nk`` the grid then runs from ``-kmax`` to ``kmax - dk``).
        """
        kmax = float(kmax)
        if not np.isfinite(kmax) or kmax <= 0:
            raise ValueError("kmax must be a positive finite number")
        offset = 0.0 if gamma else _symmetric_offset(int(nk))
        return cls(2.0 * kmax * np.eye(2), nk, offset, periodic=False)

    @classmethod
    def bz(cls, B, nk: int, *, gamma: bool = False) -> Mesh:
        """The Brillouin zone spanned by the reciprocal lattice vectors ``B`` (columns), periodic.

        ``gamma=False`` is the Monkhorst-Pack grid, closed under ``k -> -k``
        for every ``nk``.  ``gamma=True`` always samples Γ.
        """
        offset = 0.0 if gamma else _symmetric_offset(int(nk))
        return cls(np.asarray(B, dtype=float), nk, offset, periodic=True)

    def with_nk(self, nk: int) -> Mesh:
        """The same cell meshed with ``nk`` points per direction (same ``gamma`` rule)."""
        gamma = self.offset == 0.0 and (self.nk % 2 == 0)
        offset = 0.0 if gamma else _symmetric_offset(int(nk))
        return replace(self, nk=int(nk), offset=offset)

    # -- coordinates -------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """``(nk, nk)``, the leading axes of arrays on the mesh."""
        return (self.nk, self.nk)

    @property
    def center_index(self) -> int:
        """Index of the grid line through Γ (exactly Γ when ``offset == 0``)."""
        return self.nk // 2

    @property
    def gamma_index(self) -> tuple[int, int] | None:
        """Index of Γ, or ``None`` when the grid does not sample it."""
        c = self.center_index
        return (c, c) if self.offset == 0.0 else None

    @property
    def frac(self) -> np.ndarray:
        """1D fractional coordinate of the grid lines, ``(arange(nk) - nk//2 + offset) / nk``."""
        return (np.arange(self.nk) - self.center_index + self.offset) / self.nk

    @property
    def frac_grid(self) -> np.ndarray:
        """Fractional coordinates of the mesh points, shape ``(nk, nk, 2)``."""
        f = self.frac
        return np.stack(np.meshgrid(f, f, indexing="ij"), axis=-1)

    def to_cartesian(self, frac) -> np.ndarray:
        """Cartesian momenta of fractional coordinates ``(..., 2)``."""
        return np.einsum("ij,...j->...i", self.cell, np.asarray(frac, dtype=float))

    def to_fractional(self, ks) -> np.ndarray:
        """Fractional coordinates of Cartesian momenta ``(..., 2)``."""
        return np.einsum("ij,...j->...i", np.linalg.inv(self.cell), np.asarray(ks, dtype=float))

    @property
    def ks(self) -> np.ndarray:
        """Cartesian momenta of the mesh points, shape ``(nk, nk, 2)``."""
        return self.to_cartesian(self.frac_grid)

    @property
    def rectangular(self) -> bool:
        """Whether the cell vectors lie along the Cartesian axes."""
        return self.cell[0, 1] == 0.0 and self.cell[1, 0] == 0.0

    @property
    def kx(self) -> np.ndarray:
        """Cartesian ``kx`` of the grid lines along axis 0 (rectangular cells only)."""
        self._require_rectangular("kx")
        return self.cell[0, 0] * self.frac

    @property
    def ky(self) -> np.ndarray:
        """Cartesian ``ky`` of the grid lines along axis 1 (rectangular cells only)."""
        self._require_rectangular("ky")
        return self.cell[1, 1] * self.frac

    def _require_rectangular(self, name: str) -> None:
        if not self.rectangular:
            raise ValueError(
                f"Mesh.{name} is defined for rectangular cells only; this cell is oblique. "
                "Use mesh.frac (fractional axis) or mesh.ks (Cartesian points)."
            )

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """1D coordinates of the grid lines along the two cell vectors.

        Cartesian ``(kx, ky)`` for a rectangular cell, fractional ``(frac, frac)``
        for an oblique one (the only coordinates an oblique grid has per axis).
        """
        if self.rectangular:
            return self.kx, self.ky
        return self.frac, self.frac

    @property
    def transfer_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """1D coordinates of :attr:`transfer_grid`, ``(arange(nk) - nk // 2)`` steps;
        Cartesian for a rectangular cell, fractional for an oblique one."""
        n = np.arange(self.nk) - self.center_index
        if self.rectangular:
            return n * self.step[0, 0], n * self.step[1, 1]
        return n / self.nk, n / self.nk

    @property
    def kmax(self) -> float:
        """Half side of a square patch (``cell = 2 kmax I``)."""
        if not (self.rectangular and self.cell[0, 0] == self.cell[1, 1]):
            raise ValueError("Mesh.kmax is defined for square patches only.")
        return 0.5 * float(self.cell[0, 0])

    @property
    def step(self) -> np.ndarray:
        """The two grid step vectors (columns), ``cell / nk``."""
        return self.cell / self.nk

    @property
    def dk(self) -> np.ndarray:
        """Lengths of the two grid steps."""
        return np.linalg.norm(self.step, axis=0)

    @property
    def num_k(self) -> int:
        """Number of mesh points, ``nk * nk``."""
        return self.nk * self.nk

    @property
    def bounds(self) -> np.ndarray:
        """``[kx_min, kx_max, ky_min, ky_max]``: the patch for a square mesh, the points' extent otherwise."""
        if self.rectangular and not self.periodic:
            hx, hy = 0.5 * self.cell[0, 0], 0.5 * self.cell[1, 1]
            return np.array([-hx, hx, -hy, hy])
        ks = self.ks
        return np.array([ks[..., 0].min(), ks[..., 0].max(), ks[..., 1].min(), ks[..., 1].max()])

    # -- quadrature ----------------------------------------------------------------
    @property
    def area(self) -> float:
        """Area of the k-space cell, ``|det cell|``."""
        return abs(float(np.linalg.det(self.cell)))

    @property
    def weight(self) -> float:
        """Quadrature weight of one mesh point, ``|det cell| / nk^2 / (2 pi)^2``."""
        return self.area / self.nk**2 / (2 * np.pi) ** 2

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weight of each mesh point, shape ``(nk, nk)``."""
        return np.full(self.shape, self.weight)

    @property
    def weight_normalization(self) -> float:
        """Sum of the weights, ``area / (2 pi)^2``."""
        return self.weight * self.num_k

    # -- momentum transfers --------------------------------------------------------
    @property
    def transfer_grid(self) -> np.ndarray:
        """Momentum transfers ``q = k_i - k_c`` on the mesh's index grid, shape ``(nk, nk, 2)``.

        ``q`` at index ``(i, j)`` is ``step @ ((i, j) - nk // 2)``: integer
        multiples of the steps, ``q = 0`` at ``(nk // 2, nk // 2)`` for every
        ``nk`` and offset.  This is the grid an interaction ``V(q)`` is sampled
        on for the FFT exchange.  Do not sample ``V`` on ``|ks|``: a mesh
        without a ``k = 0`` point (``gamma=False`` with even ``nk``) sits half
        a step off these differences, which shifts ``V`` and breaks its
        ``q -> -q`` symmetry.
        """
        n = np.arange(self.nk) - self.center_index
        idx = np.stack(np.meshgrid(n, n, indexing="ij"), axis=-1).astype(float)
        return np.einsum("ij,...j->...i", self.step, idx)

    @property
    def transfer_distance_grid(self) -> np.ndarray:
        """``|q|`` on :attr:`transfer_grid`, shape ``(nk, nk)``."""
        q = self.transfer_grid
        return np.hypot(q[..., 0], q[..., 1])

    @property
    def extended_transfer_grid(self) -> np.ndarray:
        """Every difference ``k - k'`` of two mesh points, shape ``(2 nk - 1, 2 nk - 1, 2)``.

        Index ``(i, j)`` holds ``step @ ((i, j) - (nk - 1))``, so ``q = 0`` is
        at ``(nk - 1, nk - 1)`` and ``k_a - k_b`` at ``a - b + nk - 1``.
        """
        n = np.arange(2 * self.nk - 1, dtype=float) - (self.nk - 1)
        idx = np.stack(np.meshgrid(n, n, indexing="ij"), axis=-1)
        return np.einsum("ij,...j->...i", self.step, idx)

    def pair_index(self, dq=(0, 0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Index map ``k -> Q - k`` with ``Q = step @ dq``: ``(ix, iy, valid)``, each ``(nk, nk)``.

        The partner of point ``(i, j)`` is index ``dq + 2 c - 2 offset - (i, j)``
        with ``c = nk // 2``; ``dq = (0, 0)`` is ``k -> -k``.  Periodic meshes
        wrap it modulo ``nk``; patches mark points whose partner falls off
        the grid as invalid (and clip their index).
        """
        m = 2 * self.center_index - int(round(2 * self.offset))
        p0, p1 = (int(d) + m - np.arange(self.nk) for d in dq)
        if self.periodic:
            v0 = v1 = np.ones(self.nk, dtype=bool)
            p0, p1 = np.mod(p0, self.nk), np.mod(p1, self.nk)
        else:
            v0, v1 = (p0 >= 0) & (p0 < self.nk), (p1 >= 0) & (p1 < self.nk)
            p0, p1 = np.clip(p0, 0, self.nk - 1), np.clip(p1, 0, self.nk - 1)
        IX, IY = np.meshgrid(p0, p1, indexing="ij")
        return IX, IY, v0[:, None] & v1[None, :]

    def minus_k(self, values) -> np.ndarray:
        """``values`` at ``-k`` (leading axes ``(nk, nk)``); needs a grid closed under ``k -> -k``."""
        IX, IY, valid = self.pair_index()
        if not np.all(valid):
            raise ValueError("This mesh is not closed under k -> -k; use gamma=False.")
        return np.asarray(values)[IX, IY]

    def shift_index(self, dq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Index map ``k -> k + step @ dq`` for an integer shift ``dq``: ``(ix, iy, valid)``."""
        d0, d1 = (int(x) for x in dq)
        i = np.arange(self.nk)
        p0, p1 = i + d0, i + d1
        if self.periodic:
            v0 = v1 = np.ones(self.nk, dtype=bool)
            p0, p1 = np.mod(p0, self.nk), np.mod(p1, self.nk)
        else:
            v0, v1 = (p0 >= 0) & (p0 < self.nk), (p1 >= 0) & (p1 < self.nk)
            p0, p1 = np.clip(p0, 0, self.nk - 1), np.clip(p1, 0, self.nk - 1)
        IX, IY = np.meshgrid(p0, p1, indexing="ij")
        return IX, IY, v0[:, None] & v1[None, :]

    # -- interpolation -------------------------------------------------------------
    def interpolator(self, values, **kwargs):
        """Interpolator of ``values`` (leading axes ``(nk, nk)``) taking Cartesian points ``(..., 2)``.

        It interpolates in fractional coordinates, so it works on oblique
        cells.  Keyword arguments go to ``RegularGridInterpolator``.
        """
        frac = self.frac
        grid = RegularGridInterpolator((frac, frac), np.asarray(values), **kwargs)
        lo, hi, tol = frac[0], frac[-1], 1e-12

        def evaluate(points):
            f = self.to_fractional(points)
            # Points on the outer grid lines come back off by roundoff.
            f = np.where((f < lo) & (f > lo - tol), lo, f)
            f = np.where((f > hi) & (f < hi + tol), hi, f)
            return grid(f)

        return evaluate

    def resample(self, nk: int, values, *, method: str = "linear", **kwargs):
        """``(mesh, values)`` on the same cell with ``nk`` points per direction.

        Linear extrapolation covers the thin edge strip a finer mesh reaches;
        keyword arguments override the ``RegularGridInterpolator`` options.
        """
        new = self.with_nk(nk)
        values = np.asarray(values)
        kw = dict(method=method, bounds_error=False, fill_value=None)
        kw.update(kwargs)
        pts = new.ks
        if np.iscomplexobj(values):
            out = self.interpolator(values.real, **kw)(pts) + 1j * self.interpolator(
                values.imag, **kw
            )(pts)
        else:
            out = self.interpolator(values, **kw)(pts)
        return new, out

    def tile(self, op) -> np.ndarray:
        """A k-independent matrix ``op`` repeated at every mesh point, shape ``(nk, nk, n, n)``."""
        op = np.asarray(op)
        return np.broadcast_to(op, (*self.shape, *op.shape)).copy()


__all__ = ["Mesh"]
