"""Regression tests for defensive validation at the pybind11 boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import cpp_hf
from cpp_hf import _compat
from cpp_hf.solver import _kernel_args_for_native


pytestmark = pytest.mark.skipif(
    not _compat.HAVE_NATIVE,
    reason="cpp_hf native extension is not built",
)


def _small_kernel() -> cpp_hf.HartreeFockKernel:
    return cpp_hf.HartreeFockKernel(
        np.ones((1, 1), dtype=np.float64),
        np.zeros((1, 1, 2, 2), dtype=np.complex128),
        np.ones((1, 1, 1, 1), dtype=np.float64),
        T=0.2,
    )


def _native_scf(kernel_args, p0, *, diis_size: int = 6):
    return _compat._native.solve_scf(
        kernel_args,
        p0,
        1.0,
        2,
        1.0e-7,
        1.0e-6,
        0.5,
        0.0,
        None,
        [],
        "diis",
        diis_size,
        0,
        1.0,
        0.0,
        True,
        True,
    )


@pytest.mark.parametrize(
    "density, interaction",
    [
        (
            np.zeros((2, 2, 3, 2), dtype=np.complex128),
            np.ones((2, 2, 1, 1), dtype=np.complex128),
        ),
        (
            np.zeros((2, 2, 2, 2), dtype=np.complex128),
            np.ones((3, 2, 1, 1), dtype=np.complex128),
        ),
        (
            np.zeros((2, 2, 2, 2), dtype=np.complex128),
            np.ones((2, 2, 2, 1), dtype=np.complex128),
        ),
    ],
)
def test_native_fft_rejects_incompatible_shapes(density, interaction):
    with pytest.raises(ValueError):
        _compat._native.selfenergy_fft_full(
            interaction,
            density,
            apply_ifftshift=False,
            hermitian_channel_packing=False,
        )


def test_native_fock_rejects_wrong_density_shape():
    kernel = _small_kernel()
    args = _kernel_args_for_native(kernel)
    wrong = np.zeros((1, 1, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="P must have shape"):
        _compat._native.build_fock_apply(args, wrong, None)


def test_native_kernel_rejects_wrong_weight_shape():
    kernel = _small_kernel()
    args = _kernel_args_for_native(kernel)
    args["w2d"] = np.ones((1,), dtype=np.float64)
    with pytest.raises(ValueError, match="w2d must have shape"):
        _compat._native.build_fock_apply(args, np.zeros_like(kernel.h), None)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda args: args.__setitem__(
                "w2d", np.full((1, 1), np.nan, dtype=np.float64)
            ),
            "w2d must be finite",
        ),
        (
            lambda args: args.__setitem__("weight_sum", 2.0),
            r"weight_sum must equal sum\(w2d\)",
        ),
    ],
)
def test_native_kernel_rejects_invalid_weight_metadata(mutate, match):
    kernel = _small_kernel()
    args = _kernel_args_for_native(kernel)
    mutate(args)
    with pytest.raises(ValueError, match=match):
        _compat._native.build_fock_apply(
            args,
            np.zeros_like(kernel.h),
            None,
        )


def test_native_scf_rejects_wrong_initial_density_shape():
    kernel = _small_kernel()
    args = _kernel_args_for_native(kernel)
    wrong = np.zeros((1, 1, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="P0 must have shape"):
        _native_scf(args, wrong)


def test_native_scf_rejects_zero_diis_history_without_crashing():
    kernel = _small_kernel()
    args = _kernel_args_for_native(kernel)
    with pytest.raises(ValueError, match="diis_size must be positive"):
        _native_scf(args, np.zeros_like(kernel.h), diis_size=0)


def test_fft_plan_cache_can_be_cleared_during_concurrent_execution():
    rng = np.random.default_rng(7)
    density = np.ascontiguousarray(
        rng.normal(size=(5, 4, 3, 3))
        + 1j * rng.normal(size=(5, 4, 3, 3)),
        dtype=np.complex128,
    )
    interaction = np.ascontiguousarray(
        rng.normal(size=(5, 4, 1, 1))
        + 1j * rng.normal(size=(5, 4, 1, 1)),
        dtype=np.complex128,
    )
    expected = _compat._native.selfenergy_fft_full(
        interaction,
        density,
        apply_ifftshift=False,
        hermitian_channel_packing=False,
    )

    def execute_many():
        for _ in range(12):
            actual = _compat._native.selfenergy_fft_full(
                interaction,
                density,
                apply_ifftshift=False,
                hermitian_channel_packing=False,
            )
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def clear_many():
        for _ in range(48):
            _compat._native.clear_fft_plan_cache()

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(execute_many) for _ in range(4)]
        futures.append(pool.submit(clear_many))
        for future in futures:
            future.result()
