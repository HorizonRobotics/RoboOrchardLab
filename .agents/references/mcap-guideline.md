# MCAP Export Guideline

Use this reference when changing experimental MCAP export code, including
`Dict2Mcap`, `Dataset2Mcap`, `FoxgloveMcapWriter`, `FoxgloveEncoder`,
`StampedMessage` topic maps, converter contracts, or Pydantic JSON topics.

## Topic Map Boundaries

- Treat `StampedMessage.data` as the original business message plus timing,
  not as a pre-serialized SDK payload.
- Keep converter outputs as final-topic maps:
  `dict[str, list[StampedMessage[Any]]]`. This lets callers merge, append, or
  post-process by topic before the writer flattens the map.
- `Dict2Mcap` should stay an input-order topic-map writer. Do not add
  cross-topic `log_time` sorting, watermark merge, or batch-level scheduling
  policy to it by default.
- After converter expansion, ordering and watermark decisions must be based on
  final emitted MCAP messages, not only on wrapper source
  `StampedMessage.log_time`.
- When multiple source domains emit TF frames into one MCAP, namespace derived
  frame ids by their source domain instead of reusing raw runtime names.
  Processor-input images, rollout observations, and action sidecars should not
  publish identical `child_frame_id` values unless they describe the same
  physical frame from the same source. Treat duplicate child frames from
  different topics as a visualization correctness bug because later TF edges
  can overwrite earlier ones.

## TF And FK Export Boundaries

- When exporting FK transforms from joint-state datatypes, derive link frame
  ids from the written joint-state object instead of hardcoding robot-specific
  terminal link names or gripper frame names.
- Keep observed joint-state FK and commanded action FK in distinct frame-id
  domains when both are present in one TF graph, so visualization cannot
  overwrite one source with the other.
- For debug or viewer MCAP exports, prefer a root-to-link star graph when the
  goal is to inspect per-link poses. Do not reproduce the full URDF
  parent-child tree unless the exported contract explicitly requires full
  robot TF topology.
- Use the robot or kinematic-chain root frame as the fixed parent when camera,
  joint, and action transforms share that root coordinate system. Avoid
  hardcoded `"world"` parents when the source model owns a concrete root frame
  id.
- Do not export gripper FK only because gripper state values exist. Gripper
  transforms belong in the graph only when the source joint-state contract
  explicitly includes gripper links in the exported arm FK surface.

## Converter Ownership

- Converter instances may be created from source topic, sample data, and
  caller-provided kwargs, then reused for the same source topic stream.
- A converter should consume the complete source `StampedMessage`, so source
  `log_time` and `pub_time` remain available without adding a parallel
  `log_time` parameter.
- If converted messages preserve a relative local time axis, align each output
  topic to the source message log-time anchor before writing.
- Source-topic converters own datatype expansion, such as expanding
  `BatchCameraData` into final image, calibration, and TF topics. Writer-side
  converters own only last-mile payload conversion after final topics exist.
- Keep default converter knowledge out of the writer module. The writer should
  resolve converters, but heavy datatype and ORM imports belong in a default
  converter module.

## Writer And Encoder Boundaries

- `FoxgloveEncoder` owns channel creation, schema lookup, and topic-kind
  consistency. It should preserve original message objects inside
  `StampedMessage`.
- `FoxgloveMcapWriter` owns SDK-boundary serialization immediately before
  `Channel.log(...)`. Protobuf messages serialize with `SerializeToString()`;
  Pydantic models dump to JSON-compatible payloads at this boundary.
- Do not move SDK-required serialization into topic-map converters or generic
  encoder state unless that layer becomes the final writer boundary.
- Reused opened writers own cross-call final-topic state. `Dict2Mcap` should
  not duplicate a cross-call schema cache for final output topics.
- Final writer converters are keyed by final MCAP topic after source-topic
  conversion, not by source topic, schema name, or payload class.
- Apply final writer converters after `Dict2Mcap` source converters and
  before `FoxgloveEncoder` channel/schema resolution, so the channel kind is
  created from the payload that will actually be written.
- `FoxgloveEncoder` writer converters must be stateless and emit exactly one
  message per input message. The final writer boundary keeps topic, timestamp,
  and channel resolution one-to-one.
- `Dict2Mcap(writer_converters=...)` may install final-topic converters only
  when it opens the writer itself. If the caller passes an already-open
  `McapWriter`, construct that writer with `converters=...` instead.
- Image compression such as `RawImage` to `CompressedImage` belongs at this
  final writer boundary. Keep image topic names format-neutral, for example
  `.../image`, instead of encoding raw or compressed storage format in the
  topic name.

## JSON And Pydantic Topics

- Pydantic v2 `BaseModel` support is a final writer capability, not a
  `Dict2Mcap` default converter.
- When a caller already has a Pydantic `BaseModel` for a JSON topic, pass the
  model object through `StampedMessage.data` instead of pre-dumping it to a
  `dict`. The typed model preserves schema generation and topic-kind
  consistency until the writer boundary; an early `model_dump(...)` degrades
  the message to schemaless JSON and can hide missing fields or shape drift.
- Create Pydantic JSON channels from the model class serialization schema and
  reject `RootModel` or any model whose top-level serialization JSON schema is
  not an object.
- Keep Pydantic schema-generation options and payload-dump options aligned.
  Do not rely on mismatched Pydantic defaults when aliases or serialization
  aliases are involved.
- `FoxgloveEncoder` should maintain a topic-kind registry, not a JSON shape
  registry: exact protobuf class, exact Pydantic model class, and schemaless
  JSON are distinct topic kinds. Plain `dict` and `list` messages remain one
  schemaless JSON kind and may mix on the same topic.
- For Pydantic models consumed by Foxglove or other MCAP readers, avoid field
  types that serialize to unsupported or ambiguous JSON schema shapes. In
  particular, do not use `Any` for model fields that become MCAP topic schema;
  Pydantic may emit `anyOf` schemas that Foxglove rejects.
- When adding or changing a Pydantic model used as an MCAP topic payload, cover
  it with the shared schema-compatibility helper and at least one writer or
  encoder test that proves the schema can be registered for the intended
  topic.
- Do not add a local JSON schema fingerprint or shape validator unless a
  future task explicitly changes JSON compatibility ownership.

## Dataset2Mcap Boundary

- `Dataset2Mcap` owns episode orchestration: metadata lookup, frame range,
  timestamp validation, metadata topic-map assembly, and batch encoder use.
- `Dataset2Mcap` may delegate prepared final `StampedMessage` topic maps to
  `Dict2Mcap`.
- Dataset metadata ORM-to-MCAP payload conversion belongs to the default
  `ToMcapMessageFactory`. Keep `Robot`, `Task`, `Instruction`, and `Episode`
  storage records as source messages in `Dataset2Mcap`, and let the default
  converters produce Pydantic JSON payloads.
- Episode metadata export should remain a faithful storage metadata JSON
  export. For detailed RODataset metadata rules, use
  `.agents/references/rodataset-metadata-guideline.md`.

## Tests

- Cover both encoder boundaries and writer boundaries when adding a new
  writer-capable message family: the encoder should preserve the original
  message object, and the writer should produce the bytes or JSON payload
  accepted by the SDK.
- For Pydantic JSON topics, cover schema creation, writer payload dump,
  alias-policy consistency, RootModel rejection, and topic-kind conflicts with
  schemaless JSON.
