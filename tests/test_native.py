"""The extension loads, exports the four XLA FFI kernels and records its jaxlib."""

import jaxlib

from hartreefock_ffi import _native


def test_extension_exports_the_kernels_for_this_jaxlib():
    targets = dict(_native.xla_ffi_targets())
    assert {
        "hartreefock_eigh_c128",
        "hartreefock_eigvalsh_c128",
        "hartreefock_matmul_c128",
        "hartreefock_superlattice_fock",
    } <= set(targets)
    assert _native.xla_ffi_jaxlib_version() == jaxlib.__version__
