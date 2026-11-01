"""The look shared by the contimod and hartreefock documentation sites.

pydata-sphinx-theme with the layout, type and header behaviour of the ALPS
documentation theme and a teal accent. A site adds this directory to
``sys.path``, lists ``contimod_theme`` in ``extensions`` and sets its own
options:

- ``theme_repository``: the site's GitHub URL (the GitHub icon link);
- ``theme_sister_site``: ``(name, url)`` of the other site, or ``None``;
- ``html_theme_options``: anything else, such as ``{"logo": {"text": "contimod"}}``.

The same file is kept in ``contimod/docs/theme`` and ``cpp_hf/docs/theme``.
"""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"

THEME_OPTIONS = {
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "header_links_before_dropdown": 8,
    "show_toc_level": 2,
    "show_prev_next": True,
    "secondary_sidebar_items": ["page-toc"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "github-dark",
    "footer_start": ["copyright"],
    "footer_end": [],
}


def configure(app, config):
    config.html_theme = "pydata_sphinx_theme"
    config.html_static_path = [str(STATIC), *config.html_static_path]
    config.html_css_files = ["contimod.css", *config.html_css_files]
    config.html_js_files = ["contimod-header.js", *config.html_js_files]
    config.html_logo = str(STATIC / "logo.svg")
    config.html_favicon = str(STATIC / "logo.svg")
    config.html_show_sourcelink = False
    links = [{"name": "GitHub", "url": config.theme_repository, "icon": "fa-brands fa-github"}]
    if config.theme_sister_site:
        name, url = config.theme_sister_site
        links.append({"name": f"{name} documentation", "url": url, "icon": "fa-solid fa-book"})
    config.html_theme_options = {**THEME_OPTIONS, "icon_links": links, **config.html_theme_options}


def note_python_reference_sources(app, docname, source):
    """Rebuild generated API pages when the Python sources they read change."""
    if ".. autoapi" not in source[0]:
        return
    for _, filename in getattr(app.env, "autoapi_source_files", ()):
        path = Path(filename)
        app.env.note_dependency(path)
        # A new module need not modify any existing file; its package
        # directory changes, so register that dependency as well.
        app.env.note_dependency(path.parent)


def setup(app):
    app.add_config_value("theme_repository", "", "html")
    app.add_config_value("theme_sister_site", None, "html")
    app.connect("config-inited", configure)
    app.connect("source-read", note_python_reference_sources)
    return {"env_version": 1, "parallel_read_safe": True, "parallel_write_safe": True}
