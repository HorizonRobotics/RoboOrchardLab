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

from types import SimpleNamespace

import pytest
import torch

from robo_orchard_lab.models.holobrain.structure import (
    HoloBrain_Qwen2_5_VL,
    TextTemplate,
)
from robo_orchard_lab.models.holobrain.structure_qwen3_5 import (
    HoloBrain_Qwen3_5_VL,
)


def _build_uninitialized_holobrain():
    model = HoloBrain_Qwen2_5_VL.__new__(HoloBrain_Qwen2_5_VL)
    model.vlm = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=99,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
    )
    return model


class TestTextTemplateReferenceImages:
    """Tests for reference image prompt assembly."""

    def test_adds_reference_image_tokens_and_image_first_flag(self):
        template = TextTemplate(image_first=False)
        data = {
            "imgs": torch.zeros(2, 2, 3, 4, 4),
            "reference_imgs": [
                torch.zeros(1, 3, 4, 4),
                None,
            ],
            "text": ["pick", "place"],
        }

        result = template(data)

        image_token = template.image_token
        assert result["image_first"] is False
        assert result["instruction"] == ["pick", "place"]
        assert result["text"][0].count(image_token) == 3
        assert "Reference images:" in result["text"][0]
        assert result["text"][1].count(image_token) == 2
        assert "Reference images:" not in result["text"][1]


class TestHoloBrainReferenceImageMasks:
    """Tests for reference image ordering and patch masks."""

    def test_get_image_list_keeps_processor_order_for_reference_first(self):
        model = _build_uninitialized_holobrain()
        imgs = torch.arange(2 * 2 * 1 * 1 * 1).reshape(2, 2, 1, 1, 1)
        reference_imgs = [
            torch.full((1, 1, 1, 1), 10),
            None,
        ]
        inputs = {
            "imgs": imgs,
            "reference_imgs": reference_imgs,
            "image_first": False,
        }

        image_list, image_is_main = model._get_image_list(inputs)

        assert image_is_main == [False, True, True, True, True]
        assert [int(x.item()) for x in image_list] == [10, 0, 1, 2, 3]

    def test_build_main_img_mask_expands_per_image_patch_counts(self):
        model = _build_uninitialized_holobrain()
        image_grid_thw = torch.tensor(
            [
                [1, 4, 4],
                [1, 2, 4],
                [1, 4, 2],
                [1, 4, 4],
            ]
        )

        main_img_mask = model._build_main_img_mask(
            image_grid_thw,
            [True, False, False, True],
        )

        assert main_img_mask.tolist() == (
            [True] * 4 + [False] * 2 + [False] * 2 + [True] * 4
        )

    def test_build_main_img_mask_rejects_metadata_count_mismatch(self):
        model = _build_uninitialized_holobrain()

        with pytest.raises(ValueError, match="Image metadata count"):
            model._build_main_img_mask(
                torch.tensor([[1, 4, 4], [1, 4, 4]]),
                [True],
            )

    def test_vlm_outputs_handler_scatters_main_img_mask_to_image_tokens(self):
        model = _build_uninitialized_holobrain()
        model.with_cot = False
        model.qwen_patch_size = 2
        model.vlm_processor = SimpleNamespace(
            tokenizer=SimpleNamespace(pad_token_id=0)
        )
        model.foward_feat_mapping = lambda hidden_states: hidden_states
        hidden_states = torch.arange(6 * 4, dtype=torch.float32).reshape(
            1, 6, 4
        )
        vlm_inputs = {"input_ids": torch.tensor([[0, 99, 7, 99, 99, 8]])}
        inputs = {"imgs": torch.zeros(1, 2, 3, 2, 2)}

        feature_maps, text_dict = model._vlm_outputs_handler(
            {"hidden_states": hidden_states},
            vlm_inputs,
            inputs,
            main_img_mask=torch.tensor([True, False, True]),
        )

        expected_img_feature = torch.stack(
            [hidden_states[0, 1], hidden_states[0, 4]]
        ).reshape(1, 2, 4, 1, 1)
        assert torch.equal(feature_maps[0], expected_img_feature)
        assert text_dict["text_token_mask"].tolist() == [
            [False, True, True, True]
        ]


class TestHoloBrainQwen35:
    """Tests for Qwen3.5 VLM input adaptation."""

    def test_forward_vlm_builds_mm_token_type_ids_when_missing(self):
        model = HoloBrain_Qwen3_5_VL.__new__(HoloBrain_Qwen3_5_VL)
        captured = {}

        class FakeVLMModel:
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return {"hidden_states": ()}

        model.vlm = SimpleNamespace(
            config=SimpleNamespace(image_token_id=99, video_token_id=98),
            model=FakeVLMModel(),
        )

        model._forward_vlm(
            input_ids=torch.tensor([[1, 99, 2, 98, 3]]),
            attention_mask=torch.ones(1, 5),
            pixel_values=torch.zeros(1, 3),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )

        assert captured["mm_token_type_ids"].tolist() == [[0, 1, 0, 2, 0]]
        assert captured["output_hidden_states"] is True
        assert captured["return_dict"] is True
        assert captured["use_cache"] is False
