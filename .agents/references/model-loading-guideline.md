# Model Loading Guideline

Use this reference for stable model-loading, model-reference, and
`hf://`-compatible resource-handling guidance in this repository.

## Applicability

Use this guideline when designing, implementing, reviewing, or testing:

- `TorchModelRef`, `HFPretrainedModelRef`, `TorchModelLoadConfig`, or
  related config-facing model-loading surfaces
- pipeline, policy, or model configs that need to describe how a model is
  built or where its weights come from
- `hf://`-compatible resource handling or shared Hugging Face path helpers
- compatibility migrations away from ad hoc package-local model-loading
  wrappers

## Shared Model-Loading Surfaces

- Prefer shared model-reference objects such as `TorchModelRef` and
  `HFPretrainedModelRef` for new caller-facing or config-facing
  model-loading surfaces instead of repeating ad hoc `class_type + path +
  load_weights` wrappers in each package.
- Keep model structure and load source explicit. When a config needs both,
  represent the build config and the load-from behavior as first-class fields
  instead of hiding them in one-off booleans or helper-specific path rules.
- Keep reconstruction semantics stable when a config or ref is rebound to a
  runtime model. Do not normalize away the load-source portion of a ref if
  callers still rely on that config to rebuild the same weighted artifact.
- Narrow model-reference field types to the smallest surface the caller
  actually needs. Do not expose a broader union than necessary when a field
  is torch-only or Hugging Face-only.

## Compatibility Wrappers

- When an old package-local wrapper or import path must remain supported,
  keep it as a deprecated compatibility surface and move repository-owned
  callers to the shared model-reference path.
- Do not add new model-loading features to deprecated compatibility wrappers.
- When a compatibility wrapper remains supported, test both the canonical
  shared model-reference path and the deprecated path directly.

## `hf://`-Compatible Path Handling

- Centralize `hf://` normalization and download behavior in a shared helper
  instead of re-implementing the same translation logic in model, pipeline,
  and utility modules.
- Keep call sites consuming resolved local paths or resolved Hugging Face
  identifiers rather than duplicating repo-type branching around the helper.
- If a path helper accepts both local paths and `hf://` sources, make the
  return contract explicit so callers know when they receive an absolute local
  path versus a model identifier.
- If a path helper accepts local paths, `hf://` sources, and Hugging Face
  model identifiers, document every real branch the helper can return,
  including pass-through behavior for non-existing local-looking strings
  when that behavior is intentional.
- For Hugging Face model refs, prefer the target model's own `config_class`
  loader when available before falling back to `AutoConfig`, so custom
  `PreTrainedModel` classes do not require global `AutoConfig` registration
  on config-loading branches.

## Branch-Local API Contracts

- If the load-with-weights path and build-from-config path accept different
  kwargs, expose separate fields or validate the active branch explicitly.
  Do not route one shared kwargs bag into incompatible downstream APIs and
  rely on Hugging Face or torch internals to reject bad combinations.
- When a compatibility alias remains for a pre-split kwargs field, route it to
  the active branch deterministically and document that behavior near the ref
  contract.
- If that alias routing depends on another field that Pydantic coerces
  (for example string booleans), normalize against the coerced semantics
  rather than branching on the raw pre-validation value.

## Validation Expectations

- For model-loading migrations, add focused validation for the canonical
  shared model-reference path.
- If a deprecated wrapper or import path remains supported, validate it
  directly instead of assuming the canonical-path tests cover it.
- When changing shared `hf://` handling, cover at least one local-path case
  and one `hf://` case so the helper contract stays explicit.
- Keep at least one focused test for any documented non-existing
  local-looking path branch so callers can rely on the contract rather than
  inference from implementation.
