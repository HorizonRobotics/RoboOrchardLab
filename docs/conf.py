# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# -- Path setup --------------------------------------------------------------

import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__))))

import re
from collections import OrderedDict

from doc_gen import gen_index, patch_autoapi  # type: ignore
from pydantic import BaseModel
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

import robo_orchard_lab

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

with_comment = os.environ.get("DOC_WITH_COMMENT", "0") == "1"
# -- Project information -----------------------------------------------------

project = "RoboOrchardLab"
copyright = "2024-2025, Horizon Robotics Developers"
author = "Horizon Robotics Developers"

# # The short X.Y version
version = robo_orchard_lab.__version__
release = robo_orchard_lab.__version__

html_title = f"{project}"

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "6.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    # "recommonmark",
    "sphinx_markdown_tables",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_toolbox.more_autodoc.typevars",
    "sphinxcontrib.autodoc_pydantic",
    "autodocsumm",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    # "sphinx_toolbox.more_autodoc.generic_bases",
    # "myst_sphinx_gallery",
    "nbsphinx",
]

autodoc_mock_imports = []

html_show_sourcelink = True

autodoc_pydantic_settings_undoc_members = True
autodoc_pydantic_model_show_json = False
autodoc_pydantic_settings_show_json = False

all_typevars = True
typehints_fully_qualified = True
always_document_param_types = True
always_use_bars_union = True
typehints_defaults = "braces"
typehints_use_signature = True
typehints_use_signature_return = True
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"

# autodoc_class_signature = "separated"

autosummary_generate = True
autosummary_generate_overwrite = True
# default autodoc settings
autodoc_default_options = {
    "autosummary": True,
}

# autoapi configuration
autoapi_dirs = ["../robo_orchard_lab"]
autoapi_root = "autoapi"
autoapi_keep_files = True
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
autoapi_ignore = [
    "*migrations*",
    "*/setup.py",
    "*robo_orchard_lab/dataset/experimental",
]
autoapi_python_use_implicit_namespaces = True
autoapi_options = [
    "members",
    # "inherited-members",
    # Do not show inherited members because 3rdparty libs may not satisfy
    # the sphinx doc generation. This will bring in a lot of warnings
    # and error.
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]


def autoapi_skip_member(app):
    def _impl(app, what, name, obj, skip, options):
        if "robo_orchard_lab.ops.deformable_aggregation.setup" in name:
            skip = True
        elif "robo_orchard_lab.dataset.experimental" in name:
            skip = True
        return skip

    app.connect("autoapi-skip-member", _impl)


if with_comment:
    extensions.append("sphinx_comments")

gallery_dict = OrderedDict()
# accelerate building docs
if os.environ.get("ROBO_ORCHARD_NO_TUTORIALS", "0") != "1":
    gallery_dict["dataset_tutorial"] = [
        {"path": "tutorials/dataset_tutorial/"}
    ]
    gallery_dict["trainer_tutorial"] = [
        {"path": "tutorials/trainer_tutorial/"}
    ]
    gallery_dict["model_zoo_tutorial"] = [
        {"path": "tutorials/model_zoo_tutorial/"}
    ]

build_gallery_dict = OrderedDict()
examples_dirs = []
gallery_dirs = []

for key, value in gallery_dict.items():
    if key not in build_gallery_dict:
        build_gallery_dict[key] = []

    for v in value:
        gallery_type = v.get("gallery_type", "sphinx_gallery")
        gallery_path = v["path"]
        if gallery_type == "sphinx_gallery":
            build_gallery_dict[key].append("build/" + gallery_path)
            examples_dirs.append(gallery_path)
            gallery_dirs.append("build/" + gallery_path)
        else:
            build_gallery_dict[key].append(gallery_path)


# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None)|string %}

.. raw:: html

    <div class="admonition note">
      This page was generated from jupyter notebook. Click
      <a href="{{ env.docname.split('/')|last|e + '.ipynb' }}" class="reference download internal" download> here </a> to download.
    </div>

"""  # noqa: E501 D415 D205

nbsphinx_execute = "always"
nbsphinx_allow_errors = False

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": examples_dirs,
    # path where to save gallery generated examples
    "gallery_dirs": gallery_dirs,
    "filename_pattern": ".py",
    # "filename_pattern": r"^(?!.*ignore_exec\.py).*$",
    # "ignore_pattern": r"ignore_exec.py",
    # "example_extensions": {".py", ".rst"},
    # 'subsection_order': ExplicitOrder([
    # ]),
    "within_subsection_order": "FileNameSortKey",
    "plot_gallery": True,
    "reference_url": {},
    "matplotlib_animations": True,
    "download_all_examples": False,
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["**/nonb**.ipynb", "tutorials/**/*.rst"]


suppress_warnings = [
    "myst.header",
    # To suppress warnings that gallery ipynb files are not included in the TOC
    # This is caused by conflicting sphinx-gallery and nbsphinx extensions
    "toc.not_included",
    # Emitted if resolving references to objects in an imported module failed.
    "autoapi.python_import_resolution",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_context = {
    "current_version_id": os.getenv(
        "ROBO_ORCHARD_DOCS_CURRENT_VERSION_ID", "master"
    ),
    "versions_json_url": os.getenv(
        "ROBO_ORCHARD_DOCS_VERSIONS_JSON",
        "https://horizonrobotics.github.io/robot_lab/robo_orchard/lab/version.json",
    ),
}

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "version_switcher.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ],
}
html_css_files = [
    "css/version_switcher.css",
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#

source_parsers = {".md": CommonMarkParser}
# source_suffix = {
#     ".rst": "restructuredtext",
#     ".md": CommonMarkParser,
# }
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
# language = "zh_CN"
# html_search_language = "zh"


# html_extra_path = ["proto.html"]
# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'

pygments_style = "default"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "furo"
# # Guzzle theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the sidebar
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
    # "top_of_page_button": "edit",
    # "source_branch": "master",
    # "source_directory": "docs/",
    "light_logo": "logo/logo_light.png",
    "dark_logo": "logo/logo_dark.png",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/HorizonRobotics/robo_orchard_lab",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa
            "class": "",
        },
    ],
}
if os.environ.get("DOC_ANNOUNCEMENT", None) is not None:
    html_theme_options.update(
        dict(announcement=os.environ["DOC_ANNOUNCEMENT"])
    )

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "RoboOrchardLab"


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "RoboOrchardLab", "Documentation", [author], 1)]

# -- Extension configuration -------------------------------------------------

if with_comment:
    comments_config = {"hypothesis": True}

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", "_static/inv/py3/objects.inv"),
    "numpy": (
        "https://numpy.org/doc/stable/",
        "_static/inv/numpy/objects.inv",
    ),
    "matplotlib": (
        "http://matplotlib.sourceforge.net/",
        "_static/inv/matlab/objects.inv",
    ),
    "sortedcontainers": (
        "http://www.grantjenks.com/docs/sortedcontainers/",
        "_static/inv/sortedcontainers/objects.inv",
    ),
    "omni.isaac.lab": (
        "https://isaac-sim.github.io/IsaacLab/main/",
        "_static/inv/isaaclab/objects.inv",
    ),
    "pydantic": (
        "https://docs.pydantic.dev/",
        "_static/inv/pydantic/objects.inv",
    ),
    "torch": (
        "https://pytorch.org/docs/stable/",
        "_static/inv/torch/objects.inv",
    ),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


def is_specialized_pydantic_cls(cls: type):
    def matches_pattern(s):
        pattern = r"^.*\[.*\]$"
        return bool(re.match(pattern, s))

    # skip duplicated specialized pydantic classes e.g. SceneEntityCfg[+T].
    # if not, the autodoc will generate duplicated classes.
    return type(cls) is type(BaseModel) and matches_pattern(cls.__name__)


def get_template_cls_from_specialized_cls(cls: type):
    cls_module = sys.modules[cls.__module__]
    template_cls_name = cls.__name__.split("[")[0]
    return getattr(cls_module, template_cls_name)


def patch_autodoc(app):
    def skip_dup_process():
        def process(
            app,
            what,
            name: str,
            obj,
            skip: bool,
            options: dict[str, bool],
        ) -> bool:
            # skip duplicated specialized pydantic classes
            # e.g. SceneEntityCfg[+T].
            # if not, the autodoc will generate duplicated classes.

            is_pydantic_sp_cls = is_specialized_pydantic_cls(obj)
            if not skip and is_pydantic_sp_cls:
                return True
            return skip

        return process

    def process_base(app, name, obj, options, bases: list[type]):
        for i, base in enumerate(bases):
            # replace specialized pydantic classes with template classes
            if is_specialized_pydantic_cls(base):
                bases[i] = get_template_cls_from_specialized_cls(base)

    app.connect("autodoc-skip-member", skip_dup_process())
    app.connect("autodoc-process-bases", process_base)


def setup(app):
    app.add_js_file("google_analytics.js")
    app.add_css_file("css/custom.css")
    app.add_transform(AutoStructify)
    app.add_config_value("recommonmark_config", {}, True)
    patch_autodoc(app)
    autoapi_skip_member(app)
    # patch autoapi to support customize docstring
    patch_autoapi(app)

    gen_index(
        jinja_template_path="index.jinja", gallery_dirs_dict=build_gallery_dict
    )
    copy_files = [
        (
            os.path.join(CUR_DIR, "..", "README.md"),
            os.path.join(CUR_DIR, "readme.md"),
        ),
    ]
    for src, dst in copy_files:
        assert os.path.exists(src), f"File {src} does not exist"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Copy the file content
        with open(src, "rb") as f:
            content = f.read()
        with open(dst, "wb") as f:
            f.write(content)
