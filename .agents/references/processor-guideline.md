# Processor Guideline

Use this reference for stable processor, pre-process, post-process, and
compose-contract guidance in this repository.

## Applicability

Use this guideline when designing, implementing, reviewing, or testing:

- `ModelIOProcessor` and compatibility adapters
- `EnvelopeIOProcessor` and `PipelineEnvelope`
- composed processor chains such as `ComposedIOProcessor` and
  `ComposedEnvelopeIOProcessor`
- `pre_process(...)` / `post_process(...)` contracts and
  `processor_context` semantics
- `InferencePipeline` model-forward hook seams and compatibility behavior

## Processor Families

- `ModelIOProcessor` is the legacy-compatible family.
- `EnvelopeIOProcessor` is the preferred family for new processor work when
  side-channel context must cross pre-process and post-process boundaries.
- Keep compatibility surfaces explicit. Do not silently redefine one family
  to follow the other's contract.

## Processor Package Surface

- Treat `robo_orchard_lab.processing.io_processor.__init__` as a curated
  convenience surface, not a mirror of every processor submodule.
- Keep the common envelope-family runtime types and compose helpers at the
  package root.
- For repository-owned code, import legacy `ModelIOProcessor` family types
  from `robo_orchard_lab.processing.io_processor.base`, legacy compose types
  from `robo_orchard_lab.processing.io_processor.compose`, and envelope
  adapters/resolvers from `robo_orchard_lab.processing.io_processor.envelope`.
- If historical package-root imports must remain available for compatibility,
  keep them as deprecated re-exports rather than expanding `__all__`.

## InferencePipeline Hook Seams

- `InferencePipeline._model_forward_with_envelope(...)` is the canonical
  subclass override seam for inference runtimes that need direct access to
  envelope `processor_context`.
- The standard `InferencePipeline` runtime dispatches through
  `_model_forward_with_envelope(...)`.
- `_model_forward_with_envelope(...)` is the only supported runtime hook seam.
- The standard runtime does not dispatch through other private helper names,
  even if a subclass still defines them.
- If setup detects a downstream class that still provides
  `_model_forward_with_processor(...)` while relying on the base
  `_model_forward_with_envelope(...)`, it emits a warning because runtime
  will ignore the old private hook name.
- `_model_forward_with_processor(...)` is not a supported compatibility
  entrypoint. Detection of that legacy name exists only to surface migration
  warnings for downstream subclasses that still define it.
- Do not document or test direct-call compatibility for
  `_model_forward_with_processor(...)`; migrate callers to
  `_model_forward_with_envelope(...)`.
- Keep a single forward/post-process path. Do not introduce alias hook names
  or a second independent compatibility branch around the canonical seam.
- Direct composed `post_process(...)` calls may still reach child processors
  with `processor_context=None`; degrade gracefully there instead of raising.
- Pass data explicitly on the canonical path instead of reviving a second
  independent forward/post-process branch just to preserve the old hook name.

## Trainer Boundary

- `SimpleStepProcessor` owns per-batch orchestration only. It does not own
  `accelerator.prepare(...)`.
- Keep `accelerator.prepare(...)` at trainer/runtime setup boundaries such as
  `HookBasedTrainer` instead of adding a second prepare path inside processor
  or adapter runtime code.
- If a processor-related runtime boundary changes, make that support contract
  explicit in docs and tests rather than preserving old behavior through a
  hidden second prepare path.

## Envelope Contract

- `PipelineEnvelope` is a structural contract, not a single-sample-only one.
- `PipelineEnvelope.model_input` may describe either one sample or an already
  batched payload, depending on the owner runtime.
- `PipelineEnvelope.processor_context` aligns with the same runtime unit as
  `model_input`; it is side-channel passthrough data and not model input.
- Standard `InferencePipeline` pre-process boundaries still start from
  single-sample envelopes before optional model-input collation.
- `EnvelopeIOProcessor.post_process(..., model_input=...)` receives the final
  model-facing input.
- `InferencePipeline` collates only `model_input`.
- `InferencePipeline` does not collate `processor_context` itself. Paths
  that bypass model-input collation keep the original object. Paths that do
  collate model input pass a list of per-sample contexts, even if only one
  sample was collated, so the processor owns any batch-specific aggregation
  logic.
- `ComposedEnvelopeIOProcessor` uses `ProcessorContextStack` to distinguish
  one composed path's per-child context stack from collated per-sample lists.
- In `ComposedEnvelopeIOProcessor`, child `post_process(...)` calls receive
  the same final `model_input`, replayed in reverse order.
- In `ComposedEnvelopeIOProcessor`, child `processor_context` values are
  replayed in reverse order and are passed through exactly as returned by the
  matching `pre_process(...)` path.
- Exception: deprecated direct-call compatibility paths may call composed
  `post_process(...)` with `processor_context=None`. On that path, replay
  `None` to each child instead of raising.
- `ComposedEnvelopeIOProcessor` does not copy or deepcopy
  `processor_context`.
- If a processor needs isolation from later mutation, it must return a fresh
  context object from `pre_process(...)` instead of relying on the framework
  to snapshot it.
- If a processor intentionally relies on shared-reference behavior, document
  that choice near the implementation or test that owns it.

## Testing Expectations

- When adding a new `EnvelopeIOProcessor` or any processor or adapter that
  reads or writes `processor_context`, add tests for the contract it relies
  on.
- At minimum, tests should make clear whether the processor returns a fresh
  context object or intentionally shares references.
- For composed envelope behavior, test the `model_input` semantics and the
  `processor_context` replay semantics that the processor depends on.
- If identity matters, assert identity directly, for example with `is`, not
  only value equality.
