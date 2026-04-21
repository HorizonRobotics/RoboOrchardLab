# DictTransform Scaffold

Use this scaffold together with `.agents/references/dict-transform-guideline.md`
when adding a new repository-owned `DictTransform` or focused tests for one.

In the real source file, add the repository license header and adjust imports
to match the target module.

Default preference:

- implement new repository-owned row transforms as `DictTransform`
- let `transform(...)` return a stable structured dataclass or `BaseModel`
  when the output schema is known
- fall back to a plain `dict` only for genuinely dynamic output schemas

## Transform Class Scaffold

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from pydantic import Field
from robo_orchard_core.utils.config import ClassType

from robo_orchard_lab.transforms import (
    DictTransform,
    DictTransformConfig,
)


@dataclass
class ExampleTransformOutput:
    value: int


class ExampleTransform(DictTransform[ExampleTransformOutput]):
    cfg: "ExampleTransformConfig"

    def __init__(self, cfg: "ExampleTransformConfig") -> None:
        self.cfg = cfg

    def transform(
        self,
        value: int,
        optional_bias: int = 0,
    ) -> ExampleTransformOutput:
        return ExampleTransformOutput(
            value=value + self.cfg.delta + optional_bias
        )


class ExampleTransformConfig(DictTransformConfig[ExampleTransform]):
    class_type: ClassType[ExampleTransform] = ExampleTransform

    delta: int = 0
    input_columns: dict[str, str] | Sequence[str] = Field(
        default_factory=dict
    )
```

## Pydantic Output Variant

Use this when the transform output has a stable schema and benefits from
validation, field aliases, or other `BaseModel` features:

```python
from pydantic import BaseModel


class ExampleModelOutput(BaseModel):
    value: int


class ExampleModelTransform(DictTransform[ExampleModelOutput]):
    cfg: "ExampleModelTransformConfig"

    def __init__(self, cfg: "ExampleModelTransformConfig") -> None:
        self.cfg = cfg

    def transform(self, value: int) -> ExampleModelOutput:
        return ExampleModelOutput(value=value + self.cfg.delta)
```

## Dict Return Variant

If `transform(...)` returns a plain `dict`, keep `apply(...)[0]` as that same
dict-shaped semantic output and define `output_columns` explicitly when
`check_return_columns` needs a stable schema:

```python
class ExampleDictTransform(DictTransform[dict[str, int]]):
    cfg: "ExampleDictTransformConfig"

    def __init__(self, cfg: "ExampleDictTransformConfig") -> None:
        self.cfg = cfg

    @property
    def output_columns(self) -> list[str]:
        return ["value"]

    def transform(self, value: int) -> dict:
        return {"value": value + self.cfg.delta}


class ExampleDictTransformConfig(
    DictTransformConfig[ExampleDictTransform]
):
    class_type: ClassType[ExampleDictTransform] = ExampleDictTransform

    delta: int = 0
```

## Focused Test Scaffold

```python
def test_example_transform_apply_and_call():
    cfg = ExampleTransformConfig(delta=2)
    transform = cfg()

    structured_result, final_row = transform.apply({"value": 5})

    assert structured_result == ExampleTransformOutput(value=7)
    assert final_row == {"value": 7}
    assert transform({"value": 5}) == final_row


def test_example_transform_output_mapping_boundary():
    cfg = ExampleDictTransformConfig(
        delta=2,
        output_column_mapping={"value": "renamed_value"},
        check_return_columns=True,
    )
    transform = cfg()

    structured_result, final_row = transform.apply({"value": 5})

    assert structured_result == {"value": 7}
    assert final_row == {"value": 5, "renamed_value": 7}
```

## Identity Test Scaffold

Use this when the semantic output carries tensors, arrays, lists, or other
mutable payloads whose identity should survive row materialization:

Define the transform/config analogously to the main scaffold, but let the
structured output carry a `torch.Tensor` or other mutable payload by
reference.

```python
def test_example_transform_preserves_tensor_identity():
    cfg = ExampleTensorTransformConfig()
    transform = cfg()
    tensor = torch.tensor([1, 2, 3])

    structured_result, final_row = transform.apply({"value": tensor})

    assert structured_result.value is tensor
    assert final_row["value"] is tensor
```

## Pipeline Composition Scaffold

```python
def test_pipeline_apply_returns_last_child_structured_output():
    transform = ExampleDictTransformConfig(
        delta=2
    )() + ExampleTransformConfig(delta=3)()

    structured_result, final_row = transform.apply({"value": 5})

    assert structured_result == ExampleTransformOutput(value=10)
    assert final_row == {"value": 10}
    assert transform({"value": 5}) == final_row
```

When pipeline metadata order matters, assert the exact
`mapped_input_columns` / `mapped_output_columns` list order instead of only
wrapping them in `set(...)`.

When pipeline/container reference behavior matters, assert that the new
pipeline container does not mutate the old container while still sharing the
expected child transform or child config references.

When mutable runtime state matters, add a focused test that makes the
difference between runtime composition and config composition explicit.

## Consumer Rule Of Thumb

- Need a consumer that accepts either a leaf transform or a pipeline:
  type it against `DictRowTransform`.
- Need a config boundary that accepts either a leaf config or a pipeline
  config: type it against `DictRowTransformConfig`.
- Need semantic structured data: read `apply(...)[0]`
- Need the final row dict: read `apply(...)[1]` or use `__call__(...)`
- Need to skip row-aware logic on purpose: call `transform(...)` directly and
  document why

`DictRowTransform` and `DictRowTransformConfig` are weak interfaces. They do
not imply support for leaf-only APIs such as `transform(...)`,
`input_columns`, `output_columns`, or leaf-only outer config fields.
