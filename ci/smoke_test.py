"""Installed-wheel smoke test used by cibuildwheel: the extension loads and exports its kernels."""

from __future__ import annotations

from hartreefock_ffi import _native

targets = dict(_native.xla_ffi_targets())
expected = {
    "hartreefock_eigh_c128",
    "hartreefock_eigvalsh_c128",
    "hartreefock_matmul_c128",
    "hartreefock_superlattice_fock",
}
assert expected <= set(targets), sorted(targets)
assert _native.xla_ffi_jaxlib_version()
print("wheel smoke test passed")
