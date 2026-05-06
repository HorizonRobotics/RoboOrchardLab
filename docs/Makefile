# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
# Default to serial Sphinx builds because parallel mode has been observed to
# hang on this project. Opt in explicitly with SPHINXJOBS=auto or another
# positive job count when needed.
SPHINXJOBS    ?=
SPHINXOPTS    = -W
ifneq ($(strip $(SPHINXJOBS)),)
SPHINXOPTS    += -j $(SPHINXJOBS)
endif
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = RoboOrchardLab
SOURCEDIR     = .
BUILDDIR      = ../build/docs_build
# Debug API docs use an isolated source/output tree so they do not interfere
# with the normal full-doc build.
DEBUG_BUILDDIR = ../build/docs_debug_api
DEBUG_SOURCEDIR = build/debug_api
DEBUG_TUTORIAL_BUILDDIR = ../build/docs_debug_tutorial
DEBUG_TUTORIAL_SOURCEDIR = build/debug_tutorial
DEBUG_OVERVIEW_BUILDDIR = ../build/docs_debug_overview
DEBUG_OVERVIEW_SOURCEDIR = build/debug_overview
ROBO_ORCHARD_DOCS_CURRENT_VERSION_ID ?= master
API_ROOT      = autoapi
DEBUG_NO_TUTORIALS ?= 1

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@mkdir -p $(BUILDDIR)
	@rm -rf $(API_ROOT)
	@rm -rf tutorials/trainer_tutorial/workspace tutorials/model_zoo_tutorial/.workspace
	@PYTORCH_JIT=0 ROBO_ORCHARD_NO_SOURCE_FILE=1
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@cp -rf $(BUILDDIR)/html $(BUILDDIR)/$(ROBO_ORCHARD_DOCS_CURRENT_VERSION_ID)

clean:
	@rm -rf $(API_ROOT)
	@rm -rf $(BUILDDIR)/*
	@rm -rf build/

# Debug only the AutoAPI/autodoc/Sphinx output for a subset of modules.
# API_TARGETS accepts a file, directory, or comma-separated list relative to
# the repo root or to robo_orchard_lab/, for example:
#   make debug-api API_TARGETS="robo_orchard_lab/version.py"
#   make debug-api API_TARGETS="robo_orchard_lab/pipeline/inference"
debug-api:
	@if [ -z "$(API_TARGETS)" ]; then echo "API_TARGETS is required"; exit 1; fi
	@mkdir -p $(DEBUG_BUILDDIR)
	@rm -rf $(DEBUG_SOURCEDIR) $(DEBUG_BUILDDIR)/*
	@rm -rf tutorials/trainer_tutorial/workspace tutorials/model_zoo_tutorial/.workspace
	ROBO_ORCHARD_DOC_BUILD_MODE=debug-api ROBO_ORCHARD_API_TARGETS="$(API_TARGETS)" ROBO_ORCHARD_NO_TUTORIALS=$(DEBUG_NO_TUTORIALS) PYTORCH_JIT=0 ROBO_ORCHARD_NO_SOURCE_FILE=1 \
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(DEBUG_BUILDDIR)/html" $(SPHINXOPTS) -E $(O)

# Debug one or more tutorial gallery scripts while keeping generated output
# isolated from the normal docs build.
# This fast path intentionally skips AutoAPI generation to keep tutorial
# iteration cheap. Use debug-api when the change also needs API page coverage.
# TUTORIAL_TARGETS accepts a file, directory, or comma-separated list relative
# to the repo root, docs root, or docs/tutorials/, for example:
#   make debug-tutorial TUTORIAL_TARGETS="tutorials/model_zoo_tutorial/nonb-02_inference_api.py"
#   make debug-tutorial TUTORIAL_TARGETS="model_zoo_tutorial"
debug-tutorial:
	@if [ -z "$(TUTORIAL_TARGETS)" ]; then echo "TUTORIAL_TARGETS is required"; exit 1; fi
	@mkdir -p $(DEBUG_TUTORIAL_BUILDDIR)
	@rm -rf $(DEBUG_TUTORIAL_SOURCEDIR) $(DEBUG_TUTORIAL_BUILDDIR)/*
	@rm -rf tutorials/trainer_tutorial/workspace tutorials/model_zoo_tutorial/.workspace
	ROBO_ORCHARD_DOC_BUILD_MODE=debug-tutorial ROBO_ORCHARD_TUTORIAL_TARGETS="$(TUTORIAL_TARGETS)" ROBO_ORCHARD_NO_TUTORIALS=0 PYTORCH_JIT=0 ROBO_ORCHARD_NO_SOURCE_FILE=1 \
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(DEBUG_TUTORIAL_BUILDDIR)/html" $(SPHINXOPTS) -E $(O)

# Debug overview pages without AutoAPI, tutorials, generated docs, or the
# repository-level docs index. This is intended for fast architecture docs
# iteration under docs/overview/.
debug-overview:
	@mkdir -p $(DEBUG_OVERVIEW_BUILDDIR)
	@rm -rf $(DEBUG_OVERVIEW_SOURCEDIR) $(DEBUG_OVERVIEW_BUILDDIR)/*
	ROBO_ORCHARD_DOC_BUILD_MODE=debug-overview ROBO_ORCHARD_NO_TUTORIALS=1 PYTORCH_JIT=0 ROBO_ORCHARD_NO_SOURCE_FILE=1 \
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(DEBUG_OVERVIEW_BUILDDIR)/html" $(SPHINXOPTS) -E $(O)

.PHONY: help Makefile debug-api debug-tutorial debug-overview
