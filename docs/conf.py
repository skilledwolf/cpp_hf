"""Sphinx configuration for the hartreefock documentation."""
from pathlib import Path
import os
import re
import sys

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
sys.path[:0] = [str(DOCS), str(DOCS / "theme")]
# The executed pages draw with docs/theme/matplotlibrc and run single-threaded BLAS.
os.environ["MATPLOTLIBRC"] = str(DOCS / "theme")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

project = "hartreefock"
author = "Tobias Wolf"
copyright = "Tobias Wolf and contributors — MIT"
html_title = "hartreefock documentation"
extensions = [
    "myst_nb", "sphinx_design", "sphinx_copybutton",
    "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.doctest",
    "sphinxcontrib.bibtex", "autoapi.extension", "contimod_theme",
]
master_doc = "index"
exclude_patterns = ["README.md", "theme", "_build", "**/__pycache__", "**/.ipynb_checkpoints"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "linkify"]
myst_heading_anchors = 3
myst_linkify_fuzzy_links = False
# Every page with code cells is executed by the build: each printed number and
# figure comes from the code above it. A failing cell fails the build.
nb_execution_mode = "cache"
nb_execution_cache_path = str(DOCS / "_build" / ".jupyter_cache")
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_timeout = 600
nb_merge_streams = True
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

autoapi_dirs = [str(REPO / "src/hartreefock")]
autoapi_generate_api_docs = False
autoapi_options = ["members", "show-inheritance", "show-module-summary", "imported-members"]
autoapi_python_class_content = "class"
autodoc_typehints = "description"
napoleon_use_ivar = True
napoleon_google_docstring = False
templates_path = ["_templates"]
theme_repository = "https://github.com/skilledwolf/cpp_hf"
theme_sister_site = ("contimod", "https://skilledwolf.github.io/contimod/")
html_theme_options = {"logo": {"text": "hartreefock"}, "footer_end": ["contributing"]}
html_context = {
    "github_user": "skilledwolf", "github_repo": "cpp_hf",
    "github_version": "main", "doc_path": "docs",
}

# Every cross-reference must resolve. Third-party types are named but not
# linked, so the build needs no network.
nitpicky = True
napoleon_preprocess_types = False
EXTERNAL = ("numpy", "scipy", "xarray", "jax", "jaxlib", "pint", "matplotlib", "pathlib", "os",
            "collections", "typing", "types", "logging", "dataclasses", "enum", "functools",
            "numba", "shapely", "h5py", "joblib", "bztetra", "contimod", "abc", "concurrent",
            "io", "contextlib", "mpl_toolkits", "plotly", "omegaconf", "tqdm", "opt_einsum")
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray", "NDArray": "numpy.ndarray", "ArrayLike": "numpy.typing.ArrayLike",
    "Path": "pathlib.Path", "DataArray": "xarray.DataArray", "Dataset": "xarray.Dataset",
}
EXTERNAL_SHORT = frozenset(napoleon_type_aliases)
nitpick_ignore_regex = [
    ("py:.*", rf"({'|'.join(EXTERNAL)})(\..*)?"),
    ("py:.*", r"(np|jnp|xr|plt|mpl|hf|sp|ureg)\..*"),
    # Descriptive words that NumPy docstrings use as types.
    ("py:class", r"(array|arrays|function|complex64|complex128|float32|float64|int32|int64|bool_|"
                 r"pair|dicts|ints|floats|arraylike|n_G|N_s|coef|\d+)"),
    ("py:class", r"(Figure|Axes|Axes3D|Colorbar|Line2D|Normalize|Colormap|ScalarMappable|QuadMesh|"
                 r"PathCollection|Artist|GridSpec|Polygon|Path)"),
    ("py:class", r"(optional|array[-_]like|sequence|mapping|callable|scalar|path|file-like|"
                 r"default|shape|of|or|and|int|float|complex|bool|str|dict|list|tuple|None|"
                 r"Any|object|iterable|Iterable|Callable|Sequence|Mapping|Literal|Self)"),
]


DOCUMENTED = {"class", "exception", "function", "method", "property"}


def public_api_member(app, what, name, obj, skip, options):
    """Document what a module exports, and require a docstring on each of it."""
    if what == "apimodule":
        objects = app.env.autoapi_all_objects
        module_id = obj.id.rpartition(".")[0]
        exports = getattr(objects.get(module_id), "all", None)
        if exports is not None:
            skip = obj.name not in exports
        elif obj.imported:
            skip = True
        if obj.type in ("module", "package"):
            # Submodules have their own pages.
            skip = True
        # A name its package re-exports is documented on the package page only.
        package_exports = getattr(objects.get(module_id.rpartition(".")[0]), "all", None) or ()
        if not obj.imported and obj.name in package_exports:
            skip = True
    if obj.name.startswith("_") and obj.type != "method":
        skip = True
    if not skip and obj.type in DOCUMENTED:
        from sphinx.util import logging
        from _style import hits
        log = logging.getLogger(__name__)
        if not obj.docstring.strip() and not (obj.type == "method" and obj.name == "__init__"):
            log.warning("undocumented public name %s", obj.id, type="hartreefock", subtype="undocumented")
        for _, word in hits(obj.docstring):
            log.warning("banned word '%s' in the docstring of %s", word, obj.id,
                        type="hartreefock", subtype="style")
    return skip


def public_name(app, env, node, contnode):
    """Resolve a type named by its defining module to its documented home.

    Annotations name classes where they are defined, such as
    ``hartreefock.config.SolverConfig``; the reference may
    document them where the package exports them. Among documented objects
    with the same final name, take the one sharing the longest module prefix
    with the referring page.
    """
    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"]
    name = target.rpartition(".")[2]
    if target in EXTERNAL_SHORT:
        return contnode
    objects = env.get_domain("py").objects
    candidates = [full for full in objects if full.rpartition(".")[2] == name and full.startswith("hartreefock.")]
    if not candidates:
        return None
    context = (node.get("py:module") or "") + "." + target

    def shared(full):
        a, b = full.split("."), context.split(".")
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

    best = max(candidates, key=lambda full: (shared(full), -len(full)))
    from sphinx.util.nodes import make_refnode
    entry = objects[best]
    return make_refnode(app.builder, node["refdoc"], entry.docname, entry.node_id, contnode, best)


def clean_type(text):
    """Whether a type field is a plain expression of names, such as ``dict[str, int] | None``."""
    pieces = re.split(r"[\[\],|]", text)
    return all(re.fullmatch(r"\s*([A-Za-z_][\w.]*)?\s*", p) for p in pieces) and any(p.strip() for p in pieces)


def literal_descriptive_types(app, what, name, obj, options, lines):
    """Show a descriptive type field, such as ``(nk1, nk2) complex array``, as code.

    NumPy docstrings put shapes and dtypes in the type field. Sphinx would
    cross-reference each word of it; a type that is a plain expression of
    names, such as ``int | None`` or ``list[SolveResult]``, stays linked.
    """
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*:(?:type \S+|vartype \S+|rtype):\s*)(.+)$", line)
        if m and not clean_type(m.group(2)) and "`" not in m.group(2):
            lines[i] = f"{m.group(1)}``{m.group(2)}``"


def banned_words(app, docname, source):
    from sphinx.util import logging
    from _style import checked, hits
    if checked(docname):
        for number, word in hits(source[0]):
            logging.getLogger(__name__).warning(
                "banned word '%s' (see development/documentation.md)", word,
                location=(docname, number), type="hartreefock", subtype="style")


SECTIONS = ("Parameters\n", "Returns\n", "Attributes\n")


def primary_docstrings(app):
    """Warn when a name in ``primary_api.txt`` lacks a NumPy docstring section."""
    import importlib
    import inspect
    from pathlib import Path

    from sphinx.util import logging

    log = logging.getLogger(__name__)
    for line in (Path(app.confdir) / "primary_api.txt").read_text().splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        parts = name.split(".")
        obj = None
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except ImportError:
                continue
            try:
                for attr in parts[i:]:
                    obj = getattr(obj, attr)
            except AttributeError:
                obj = None
            break
        if obj is None:
            log.warning("primary_api.txt names %s, which does not exist", name,
                        type="hartreefock", subtype="primary")
            continue
        doc = inspect.getdoc(obj) or ""
        if not any(section in doc for section in SECTIONS):
            log.warning("%s needs a Parameters, Returns or Attributes section", name,
                        type="hartreefock", subtype="primary")


def setup(app):
    from _reference import generate
    app.connect("builder-inited", generate)
    app.connect("builder-inited", primary_docstrings)
    app.connect("source-read", banned_words)
    app.connect("autodoc-skip-member", public_api_member)
    app.connect("missing-reference", public_name)
    app.connect("autodoc-process-docstring", literal_descriptive_types, priority=600)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
