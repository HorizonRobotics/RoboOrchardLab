# Contribution Guide

## AI-assisted contribution

- This project is compatible with AI-assisted development, and contributors are encouraged to use GitHub Copilot, Codex, or other coding agents.
- The repository includes AI agent instructions in `AGENTS.md` and `.agents/instructions/`; use them as the source of truth when working with AI tools.
- AI-assisted changes should still stay focused, reuse existing patterns, and be reviewed and validated before commit.

## GitHub pull request workflow

- Community contributions can be submitted through GitHub pull requests.
- The project team does not use GitHub pull requests as the final merge path for the main code repository. Internally, code is merged through GitLab merge requests instead.
- For community pull requests, the project team will convert the submitted GitHub pull request into an internal GitLab merge request before merging it into the main repository.
- This internal conversion is used to preserve the repository's one-way synchronization model and keep the authoritative merge flow on the GitLab side.
- Your original commit history will be kept intact during this process.
- Review comments and follow-up changes may still be discussed on GitHub, but the final integration into the main repository is completed through the corresponding internal GitLab merge request.

## Install by editable mode

```bash
make install-editable
```

## Install development requirements

```bash
make dev-env
```

## Lint

```bash
make check-lint
```

## Auto format

```bash
make auto-format
```

## Build docs

```bash
export ROBO_ORCHARD_DOCS_DATA_ROOT=/path/to/local/robo_orchard_docs_assets

ln -sfn "${ROBO_ORCHARD_DOCS_DATA_ROOT}/dataset_tutorial/data1" docs/tutorials/dataset_tutorial/data1
ln -sfn "${ROBO_ORCHARD_DOCS_DATA_ROOT}/dataset_tutorial/data2" docs/tutorials/dataset_tutorial/data2
export HF_LEROBOT_HOME="${ROBO_ORCHARD_DOCS_DATA_ROOT}/lerobot"

make doc
```

The docs build executes runnable tutorials through Sphinx Gallery. Some
dataset tutorials require local sample datasets under
`docs/tutorials/dataset_tutorial/data1` and `data2`, plus a local
`HF_LEROBOT_HOME` for LeRobot-related examples. Use any local directory that
matches this layout; the example above intentionally uses a generic placeholder
path instead of an internal workspace address.

The default docs Makefile now runs Sphinx serially because parallel builds have
been observed to hang on this project. If your environment is stable and you
want to try parallelism, pass `SPHINXJOBS=auto` or another positive job count
explicitly.

## Debug a subset of API docs

Use the debug target when you only need to inspect the combined
`autoapi + autodoc + sphinx` output for a small module set.

```bash
# Single file
make doc-debug-api API_TARGETS="robo_orchard_lab/version.py"

# Multiple files
make doc-debug-api API_TARGETS="robo_orchard_lab/version.py,robo_orchard_lab/pipeline/inference/mixin.py"

# Directory
make doc-debug-api API_TARGETS="robo_orchard_lab/pipeline/inference"
```

- `API_TARGETS` accepts a Python file, a package directory, or a comma-separated list.
- Paths may be relative to the repository root or to `robo_orchard_lab/`.
- The debug build skips tutorials by default and writes HTML to `build/docs_debug_api/html`.
- Use `make doc` when you need the full documentation build.

## Debug a subset of tutorials

Use the tutorial debug target when you only need to inspect the Sphinx Gallery
output for one or more tutorial scripts.

```bash
# Single tutorial file
make doc-debug-tutorial TUTORIAL_TARGETS="tutorials/model_zoo_tutorial/nonb-02_inference_api.py"

# Tutorial directory
make doc-debug-tutorial TUTORIAL_TARGETS="tutorials/model_zoo_tutorial"
```

- `TUTORIAL_TARGETS` accepts a tutorial Python file, a tutorial directory, or a comma-separated list.
- Paths may be relative to the repository root, to `docs/`, or to `docs/tutorials/`.
- The debug build writes HTML to `build/docs_debug_tutorial/html`.
- The tutorial fast path intentionally skips AutoAPI generation; use `make doc-debug-api` when the same change also needs API page validation.

## Preview docs

```bash
cd build/docs_build/html
python3 -m http.server {PORT}
# open browser: http://localhost:{PORT}
```

## Run test

```bash
make test
```

## Acknowledgement

Our work is inspired by many existing deep learning algorithm frameworks, such as [OpenMM Lab](https://github.com/openmm), [Hugging face transformers](https://github.com/huggingface/transformers) [Hugging face diffusers](https://github.com/huggingface/diffusers) etc. We would like to thank all the contributors of all the open-source projects that we use in RoboOrchard.
