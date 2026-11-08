"""scikit-build-core metadata provider: pin the wheel to the jaxlib it compiles against.

The FFI kernels are registered only under the jaxlib whose headers built
them (see ``CMakeLists.txt``), so the wheel declares ``jaxlib==<that version>``.
"""

from __future__ import annotations


def dynamic_metadata(field, settings=None, project=None):
    if field != "dependencies":
        raise RuntimeError(f"jaxlib_pin provides only 'dependencies', not {field!r}")
    import jaxlib

    return ["jax>=0.5.3", f"jaxlib=={jaxlib.__version__}"]


def get_requires_for_dynamic_metadata(settings=None):
    return ["jax>=0.5.3"]
