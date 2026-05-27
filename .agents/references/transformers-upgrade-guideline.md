# Transformers Upgrade Guideline

Use this reference for stable repository-local guidance when upgrading Hugging
Face Transformers or fixing post-upgrade regressions in `robo_orchard_lab`.

## Applicability

Use this guideline when a task changes the Transformers version or touches
repository-owned code that depends on:

- `robo_orchard_lab.utils.transformers_compat`
- `TorchModelRef`, `HFPretrainedModelRef`, model-loading helpers, or other
  repository-owned Hugging Face compatibility boundaries
- `PreTrainedModel` or `GenerationMixin` wrappers
- tokenizer special tokens, chat templates, or multimodal placeholder tokens
- generation-time cache classes, `GenerationConfig`, or `generate()` kwargs

For shared model-loading contracts and `hf://`-specific rules, also read
`.agents/references/model-loading-guideline.md`.

## Upgrade Workflow

1. Pin the exact current and target Transformers versions before editing.
2. Read the target-version migration page, release notes, and generation or
   cache docs when the changed seam touches generation, cache, or tokenizer
   behavior.
3. Inventory repository-owned upgrade seams before patching code:
   - `robo_orchard_lab.utils.transformers_compat`
   - model refs or loaders that normalize Hugging Face kwargs
   - wrappers around `PreTrainedModel` or `GenerationMixin`
   - tokenizer-loading helpers, chat-template setup, or multimodal prompt
     assembly
   - direct imports from non-public Transformers symbols
4. Fix compatibility at repository-owned boundaries before patching one-off
   callers, scripts, or project-local entrypoints.
5. Add focused regressions at the same seam where the break was found.

## Repository-Local Rules

- Keep Transformers version-specific compatibility behavior in
  `robo_orchard_lab.utils.transformers_compat` instead of spreading runtime
  version branches across model implementations.
- When a model-loading or wrapper seam depends on Hugging Face behavior, prefer
  boundary fixes in shared loaders, model refs, or compatibility helpers over
  one-off patches in downstream callers.
- Do not assume internal generation signals such as `cache_position`, cache
  subclasses, or wrapper callbacks remain stable across upgrades.
- When a local wrapper overrides upstream generation or attention semantics,
  re-check every coupled behavior such as mask preparation, cache setup,
  prefill/decode staging, and validation instead of patching one symptom.
- Keep version-specific facts such as removed symbols or renamed kwargs in
  version-scoped notes rather than baking them into durable guidance.

## Required Validation Surfaces

- `tests/test_robo_orchard_lab/utils/test_transformers_compat.py` for shared
  compatibility-helper behavior and runtime-version normalization
- `tests/test_robo_orchard_lab/models/test_model_ref.py` when the upgrade
  touches shared model refs, loaders, or Hugging Face config loading
- `tests/test_robo_orchard_lab/utils/test_state.py` when the upgrade affects
  `PreTrainedModel`, tokenizer, or processor save/load ownership
- at least one focused downstream model or module test under
  `tests/test_robo_orchard_lab/models/` when the change touches generation,
  cache, tokenizer, or wrapper behavior
- at least one real runtime path such as evaluation, export/load, or a short
  project-specific smoke test when the changed seam affects runtime integration
