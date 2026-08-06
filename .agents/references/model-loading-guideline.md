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

## Transformers Compatibility

- Keep Transformers version-specific compatibility behavior in
  `robo_orchard_lab.utils.transformers_compat` instead of spreading
  runtime-version branches through model implementations.
- Do not import private Transformers helpers directly from model code. If an
  older supported Transformers version provides a helper whose behavior must
  be preserved, delegate to it inside the repository-owned compatibility
  module and keep the fallback there.
- Normalize Hugging Face `dtype` / `torch_dtype` load and build kwargs
  through the shared compatibility helper so model refs and model-specific
  loaders follow the same runtime rule.
- For Transformers compatibility migrations, validate behavior at the
  downstream model or module boundary. For tensor-shape or attention-mask
  changes, inspect the actual tensors received by the underlying Hugging Face
  module instead of relying only on wrapper-level outputs.

## DeepSpeed ZeRO-3 Model Weight Loading

- Detect whether ZeRO-3 manages a constructed model from the parameter state
  that affects loading, such as DeepSpeed's `ds_id` parameter protocol. Do not
  route ordinary model loading from Accelerate or Transformers process-global
  flags alone; those flags can be active before every model in the process has
  the same parameter representation.
- Load ordinary exported model weights into ZeRO-3 parameters through a
  repository-owned, layerwise gather path. Big Model Inference helpers such as
  `load_checkpoint_and_dispatch` own meta-device and `device_map` dispatch;
  they are not a ZeRO-3 runtime parameter loader and must not be used as one.
- Keep DeepSpeed optional at the package boundary. Detection and module import
  must work without importing DeepSpeed; import it lazily only after a model
  with ZeRO-3 managed parameters actually enters the specialized load path.
- All ranks must traverse the same module and gather sequence. For models that
  mix ZeRO-3 parameters with ordinary parameters or buffers, copy the local
  layer state on every rank and use `modifier_rank=0` to synchronize and
  repartition the gathered ZeRO parameters. The rank-zero copy gate is
  relative to each parameter's ZeRO data-parallel process group, not the
  process-global rank.
- The loader assumes that every rank has already constructed the same model
  topology and that the selected artifact is the intended generation. A
  caller must pass one unambiguous single-file or indexed layout; mixed
  layouts, duplicate shard keys, missing index shards, and unreferenced
  sibling weight files are rejected rather than guessed.
- Preserve the model-weight formats produced by the shared save surfaces:
  support a single safetensors file and an indexed safetensors shard directory.
  The public ``state_dict()`` key set is the ZeRO-3 checkpoint contract, so
  intentional public omissions remain omitted, and every public key must map
  directly to a persistent parameter or buffer owner. Artifact tensors must
  have the exact logical target shape (``ds_shape`` for a partitioned
  parameter) before any rank enters a gather or copy. The loader restores only
  persistent aliases represented by the same ``Parameter``
  or buffer object in the target model. Non-persistent buffers are not
  state-dict keys and are never synthesized. It does not invoke a custom
  ``load_state_dict`` method, support registered load-state hooks, interpret
  safetensors metadata, or consume repository alias sidecars. `strict=True`
  reports missing and unexpected keys against that public key set;
  `strict=False` returns them without inventing values from untrusted metadata.
  Ordinary meta-device parameters and buffers are rejected; meta-device
  materialization remains the responsibility of the dedicated dispatch path.
- This loading MR does not define a writer or publication transaction. It does
  not promise crash-safe replacement, staging cleanup, Accelerate offloaded
  writer support, sidecar generation, or recovery of aliases represented only
  by distinct objects/shared storage.
- Reject `device_map` for ZeRO-3 model-weight loading. Let DeepSpeed own final
  parameter placement and do not call `model.to(...)` after the gathered load.
  A submitter should not disable `zero3_init_flag` merely to make a generic
  loader see full parameter shapes; fix the model-weight loading boundary
  instead.
- Validate the import-without-DeepSpeed path, single-file tied weights,
  indexed shards, strict missing/unexpected keys, automatic loader routing,
  and placement rejection with focused tests. Use a real multi-rank DeepSpeed
  job as the integration gate for collective ordering and repartitioning.

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
