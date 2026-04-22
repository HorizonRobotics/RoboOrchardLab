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

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
from robo_orchard_core.utils.config import ClassType

from robo_orchard_lab.models.mixin import ModelMixin, TorchModuleCfg
from robo_orchard_lab.pipeline.inference import (
    InferencePipeline,
    InferencePipelineCfg,
)
from robo_orchard_lab.policy.base import (
    InferencePipelinePolicy,
    InferencePipelinePolicyCfg,
    PolicyConfig,
    PolicyMixin,
)


class PolicyWithoutTo(PolicyMixin):
    cfg: "PolicyWithoutToCfg"

    def __init__(self, cfg: "PolicyWithoutToCfg") -> None:
        self.cfg = cfg

    def reset(self, **kwargs) -> None:
        del kwargs

    def act(self, obs):
        return obs


class PolicyWithoutToCfg(PolicyConfig[PolicyWithoutTo]):
    class_type: ClassType[PolicyWithoutTo] = PolicyWithoutTo


class DummyModel(ModelMixin):
    cfg: "DummyModelCfg"

    def __init__(self, cfg: "DummyModelCfg"):
        super().__init__(cfg)
        self.linear = nn.Linear(1, 1)

    def forward(self, batch: dict[str, torch.Tensor]):
        return {"output_data": self.linear(batch["input_data"])}


class DummyModelCfg(TorchModuleCfg[DummyModel]):
    class_type: ClassType[DummyModel] = DummyModel


class DummyPipeline(
    InferencePipeline[dict[str, torch.Tensor], dict[str, torch.Tensor]]
):
    last_reset_kwargs: dict[str, Any]

    def __init__(
        self,
        cfg: "DummyPipelineCfg",
        model: DummyModel | None = None,
    ) -> None:
        super().__init__(cfg=cfg, model=model)
        self.last_reset_kwargs = {}

    def reset(self, **kwargs) -> None:
        self.last_reset_kwargs = dict(kwargs)


class DummyPipelineCfg(InferencePipelineCfg[DummyPipeline]):
    class_type: ClassType[DummyPipeline] = DummyPipeline
    model_cfg: DummyModelCfg = DummyModelCfg()


class DefaultResetPipeline(
    InferencePipeline[dict[str, torch.Tensor], dict[str, torch.Tensor]]
):
    pass


class DefaultResetPipelineCfg(InferencePipelineCfg[DefaultResetPipeline]):
    class_type: ClassType[DefaultResetPipeline] = DefaultResetPipeline
    model_cfg: DummyModelCfg = DummyModelCfg()


def test_policy_mixin_to_requires_override():
    policy = PolicyWithoutTo(PolicyWithoutToCfg())

    try:
        policy.to("cpu")
    except NotImplementedError as exc:
        assert "must be implemented" in str(exc)
    else:
        raise AssertionError("PolicyMixin.to() should require an override.")


def test_inference_pipeline_policy_reset_forwards_kwargs():
    policy = InferencePipelinePolicy(
        InferencePipelinePolicyCfg(pipeline_cfg=DummyPipelineCfg())
    )

    policy.reset(episode_id=1)

    assert policy.pipeline.last_reset_kwargs == {"episode_id": 1}


def test_inference_pipeline_policy_reset_accepts_default_pipeline_hook():
    policy = InferencePipelinePolicy(
        InferencePipelinePolicyCfg(pipeline_cfg=DefaultResetPipelineCfg())
    )

    policy.reset(episode_id=1)

    output = policy.act({"input_data": torch.ones(1, 1)})
    assert "output_data" in output
    assert output["output_data"].shape == (1, 1)


def test_inference_pipeline_policy_to_delegates_to_pipeline(monkeypatch):
    policy = InferencePipelinePolicy(
        InferencePipelinePolicyCfg(pipeline_cfg=DummyPipelineCfg())
    )
    called: dict[str, object] = {}

    def fake_to(device: torch.device | str):
        called["device"] = device

    monkeypatch.setattr(policy.pipeline, "to", fake_to)

    policy.to("cuda:1")

    assert called["device"] == "cuda:1"


def test_inference_pipeline_policy_accepts_identical_pipeline_cfg_without_eq(
    monkeypatch,
):
    pipeline_cfg = DummyPipelineCfg()
    pipeline = DummyPipeline(cfg=pipeline_cfg)

    def _raise_on_eq(self, other):
        del self, other
        raise AssertionError("__eq__ should not be called for same cfg")

    monkeypatch.setattr(DummyPipelineCfg, "__eq__", _raise_on_eq)

    policy = InferencePipelinePolicy(
        InferencePipelinePolicyCfg(),
        pipeline=pipeline,
    )

    assert policy.pipeline is pipeline


def test_inference_pipeline_policy_state_save_load_preserves_pipeline(
    tmp_path,
):
    policy = InferencePipelinePolicy(
        InferencePipelinePolicyCfg(pipeline_cfg=DummyPipelineCfg())
    )
    policy.reset(episode_id=1)
    save_path = tmp_path / "policy"

    policy.save(str(save_path))
    recovered = InferencePipelinePolicy.load(str(save_path))

    assert isinstance(recovered, InferencePipelinePolicy)
    assert isinstance(recovered.pipeline, DummyPipeline)
    assert recovered.pipeline.last_reset_kwargs == {"episode_id": 1}

    output = recovered.act({"input_data": torch.ones(1, 1)})
    assert "output_data" in output
    assert output["output_data"].shape == (1, 1)
