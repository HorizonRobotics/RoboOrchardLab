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
make doc
```

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
