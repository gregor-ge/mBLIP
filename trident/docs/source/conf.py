# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "trident"
copyright = "2021, Fabian David Schmidt"
author = "Fabian David Schmidt"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    # "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
]

repo = "https://github.com/fdschmidt93/trident/"
main = f"{repo}tree/main/"

extlinks = {
    "repo": (f"{main}%s", None),
    "gh-pages": ("fdschmidt93.github.io/trident/", None),
}

# TODO(fdschmidt93): somewhat hacky to generate docs in minimal env with less warnings
autodoc_mock_imports = [
    "torch",
    "hydra",
    "numpy",
    "pytest",
    "omegaconf",
    "tmp",
    "rich",
    "lightning",
    "matplotlib",
    "seaborn",
    "dotenv",
    "sh",
    "wandb",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["configs.py"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def linkcode_resolve(domain, info):
    project_dir = Path.cwd().parent

    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=str(project_dir))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    return f"{main}{filename}"
