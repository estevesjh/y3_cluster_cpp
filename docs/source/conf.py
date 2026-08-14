# Sphinx configuration for the DES Y3 cluster-cosmology documentation.
#
# Build locally with:
#   sphinx-build -W -b html docs/source docs/build/html
# (see docs/source/building_docs.md)

import datetime

project = "DES Y3 Cluster Cosmology"
author = "DES Y3 cluster working group"
copyright = f"{datetime.date.today().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_wagtail_theme",
]

# MyST-Markdown: $...$ / $$...$$ math, ::: fences, definition lists,
# GitHub-style heading anchors (needed by the included BUILDING.md TOC).
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 4

# Render ``{todo}`` boxes while the documentation is being filled in.
todo_include_todos = True

exclude_patterns = ["_build"]

# Theme: sphinx-wagtail-theme, matching the pandora-box template
# (https://pandora-box.readthedocs.io).
html_theme = "sphinx_wagtail_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "project_name": "DES Y3 Cluster Cosmology",
    # DES hexagon emblem on a white background (cropped from the official
    # des-logo-rev-lg.png, darkenergysurvey.org).
    "logo": "img/des-logo-white.png",
    "logo_alt": "Dark Energy Survey",
    "logo_height": 59,
    "logo_width": 59,
    # Base URL for the per-page "Edit on GitHub" button (the theme's
    # default points at its own repo, so set it explicitly).
    "github_url": "https://github.com/estevesjh/y3_cluster_cpp/blob/master/docs/source/",
    "footer_links": ",".join(
        [
            "Repository|https://github.com/estevesjh/y3_cluster_cpp",
            "Building the pipeline|installation.html",
        ]
    ),
}
