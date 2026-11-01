"""Generate the Python reference pages at the start of a build.

One AutoAPI page per public module: every package and module under
``src/hartreefock`` whose dotted path has no part starting with ``_``. The pages
under ``reference/python`` are written here and are not checked in.
"""
import ast
from pathlib import Path

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
PACKAGE = REPO / "src/hartreefock"
PYTHON_DIR = DOCS / "reference" / "python"
SKIP_PARTS = set()

# The reference's table of contents: (caption, top-level names under hartreefock).
SECTIONS = [
    ("Package", ("__init__",)),
    ("Modules", ("config", "kernel", "superlattice", "continuation", "deflation", "stability", "symmetry")),
]


def _exports(path):
    """The names in a module's ``__all__``, or ``None`` when it has none."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
            return set(ast.literal_eval(node.value))
    return None


def _defined(path):
    """Public names a module defines at top level."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return {n for n in names if not n.startswith("_")}


def public_modules():
    """Dotted names (relative to ``hartreefock``) of the documented modules.

    Every package and module whose path has no ``_``-prefixed part, except a
    module whose public names its package re-exports in full: those are
    documented on the package page.
    """
    names = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if any(p.startswith("_") or p in SKIP_PARTS for p in parts):
            continue
        if path.name != "__init__.py":
            package_exports = _exports(path.parent / "__init__.py") or set()
            mine = _exports(path)
            mine = mine if mine is not None else _defined(path)
            if parts[-1] not in package_exports and mine and mine <= package_exports:
                continue
        names.append(".".join(parts))
    return names


def generate(app):
    PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    modules = public_modules()
    pages = {}
    for name in modules:
        module = "hartreefock" + ("." + name if name else "")
        page = name.replace(".", "/") if name else "hartreefock"
        pages[page] = name
        path = PYTHON_DIR / (page + ".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, f"# `{module}`\n\n```{{eval-rst}}\n.. autoapimodule:: {module}\n   :members:\n```\n")
    for stale in PYTHON_DIR.rglob("*.md"):
        page = stale.relative_to(PYTHON_DIR).with_suffix("").as_posix()
        if page != "index" and page not in pages:
            stale.unlink()

    def top(page):
        name = pages[page]
        return name.split(".")[0] if name else "__init__"

    listed = {page for _, roots in SECTIONS for page in pages if top(page) in roots}
    assert listed == set(pages), sorted(set(pages) - listed)
    body = "".join(
        f"```{{toctree}}\n:caption: {title}\n:maxdepth: 1\n\n"
        + "\n".join(sorted(page for page in pages if top(page) in roots)) + "\n```\n\n"
        for title, roots in SECTIONS)
    write(PYTHON_DIR / "index.md",
          "# Python API\n\nOne page per public module, generated from the source. "
          "A module's page lists what its `__all__` exports, or every public name "
          "when it has no `__all__`.\n\n" + body)


def write(path, content):
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
