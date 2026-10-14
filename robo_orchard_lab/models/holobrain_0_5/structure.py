# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

import os

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoProcessor,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)

from robo_orchard_lab.models.holobrain_0_5.resources import (
    export_pretrained_directory,
    link_pretrained_directory,
    pretrained_export_path,
    pretrained_source,
)
from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
    TorchModuleCfgType_co,
)
from robo_orchard_lab.utils.build import DelayInitDictType, build

__all__ = [
    "HoloBrain_Qwen3_5_VL",
    "HoloBrain_Qwen3_5_VLConfig",
    "TextTemplate",
]


class TextTemplate(nn.Module):
    """Format instructions and camera placeholders for action prediction."""

    def __init__(self):
        super().__init__()
        self.template = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nYou are a robot. {instruction}\n"
            "{image_token}<|im_end|>\n<|im_start|>assistant\n"
        )

    def forward(self, data):
        batch_size, num_cams = data["imgs"].shape[:2]
        instructions = data.get("text", [""] * batch_size)
        if isinstance(instructions, str) or len(instructions) != batch_size:
            raise ValueError("text must contain one instruction per sample")
        image_token = " ".join(
            ["<|vision_start|><|image_pad|><|vision_end|>"] * num_cams
        )
        data["instruction"] = instructions
        data["text"] = [
            self.template.format(instruction=x or "", image_token=image_token)
            for x in instructions
        ]
        return data


class HoloBrain_Qwen3_5_VL(ModelMixin):  # noqa: N801
    """HoloBrain 0.5 action model with a truncated Qwen3.5 backbone.

    Training returns action and optional depth losses; evaluation returns
    per-sample trajectories in [mode, time, joint, 8] layout. Each state holds
    a joint angle or gripper value, xyz and a scalar-first quaternion.
    """

    cfg: "HoloBrain_Qwen3_5_VLConfig"

    def __init__(self, cfg: "HoloBrain_Qwen3_5_VLConfig"):
        super().__init__(cfg)
        self._init_holobrain_modules()

        vlm_pretrain = self._get_vlm_pretrain()
        self._vlm_resource_source = vlm_pretrain
        self.vlm = self._build_vlm(vlm_pretrain)

        language_model = self._get_vlm_language_model()
        vision_model = self._get_vlm_vision_model()
        self._configure_vlm_freezing(language_model, vision_model)

        origin_num_layers = len(language_model.layers)
        self._select_vlm_layers(language_model)
        num_hidden_states = self._get_vlm_num_hidden_states(language_model)
        self._configure_vlm_after_layer_selection(language_model)

        self.vlm_processor = AutoProcessor.from_pretrained(
            vlm_pretrain, use_fast=True, local_files_only=True
        )
        self.vlm_processor.tokenizer.padding_side = "left"
        self.action_feature_layer_indices = tuple(range(num_hidden_states))
        if self.decoder is not None or self.spatial_enhancer is not None:
            self._init_feat_mapping(language_model)
            self._init_action_feature_weight(
                origin_num_layers, num_hidden_states
            )
        self.qwen_patch_size = self._get_qwen_patch_size()

    def _init_holobrain_modules(self):
        self.decoder = build(self.cfg.decoder)
        self.spatial_enhancer = build(self.cfg.spatial_enhancer)
        self.data_preprocessor = build(self.cfg.data_preprocessor)
        self.backbone_3d = build(self.cfg.backbone_3d)
        self.neck_3d = build(self.cfg.neck_3d)
        self.input_2d = self.cfg.input_2d
        self.input_3d = self.cfg.input_3d
        self.use_state_dict_with_vlm = self.cfg.use_state_dict_with_vlm
        if not self.use_state_dict_with_vlm:
            assert self.cfg.freeze_vlm, (
                "The VLM's state_dict must be saved when it is not frozen."
            )

    def _build_vlm(self, vlm_pretrain):
        source = pretrained_source(vlm_pretrain)
        if not os.path.isdir(source):
            raise FileNotFoundError(
                f"Expected a local Qwen3.5 directory: {source}"
            )
        if self.cfg.load_vlm_checkpoint:
            return Qwen3_5ForConditionalGeneration.from_pretrained(
                source,
                dtype=torch.bfloat16,
                attn_implementation=self.cfg.attn_implementation,
                local_files_only=True,
            )
        config = Qwen3_5Config.from_pretrained(source, local_files_only=True)
        return Qwen3_5ForConditionalGeneration._from_config(
            config,
            dtype=torch.bfloat16,
            attn_implementation=self.cfg.attn_implementation,
        )

    def _configure_vlm_after_layer_selection(self, language_model):
        if not self.cfg.freeze_vlm:
            language_model.norm.requires_grad_(False)
            language_model.layers[-1].requires_grad_(False)

    def forward(self, inputs):
        """Return losses or trajectories for a collated action batch."""
        if self.data_preprocessor is not None:
            inputs = self.data_preprocessor(
                inputs, next(self.parameters()).device
            )
        context = self._encode_vlm(inputs)
        outputs, _, text_dict, loss_depth = self._forward_decoder(
            inputs, context
        )
        if self.training:
            losses = self.decoder.loss(outputs, inputs, text_dict=text_dict)
            if loss_depth is not None:
                losses["loss_depth"] = loss_depth
            return losses
        return self.decoder.post_process(outputs, inputs, text_dict=text_dict)

    def _encode_vlm(self, inputs):
        images = inputs["imgs"].flatten(0, 1)
        vlm_inputs = self.vlm_processor(
            text=inputs["text"],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(next(self.parameters()).device)
        return {
            "outputs": self._forward_vlm(**vlm_inputs),
            "vlm_inputs": vlm_inputs,
        }

    def _get_vlm_pretrain(self) -> str:
        return pretrained_source(self.cfg.vlm_pretrain)

    def _get_vlm_language_model(self):
        if hasattr(self.vlm, "language_model"):
            return self.vlm.language_model
        return self.vlm.model.language_model

    def _get_vlm_vision_model(self):
        if hasattr(self.vlm, "visual"):
            return self.vlm.visual
        return self.vlm.model.visual

    def _configure_vlm_freezing(self, language_model, vision_model):
        if self.cfg.freeze_vlm:
            self.vlm.eval()
            self.vlm.requires_grad_(False)
        else:
            if self.cfg.lm_with_cp:
                language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            if self.cfg.freeze_vision:
                vision_model.eval()
                vision_model.requires_grad_(False)
            elif self.cfg.vision_with_cp:
                vision_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

    def _select_vlm_layers(self, language_model):
        if (
            self.cfg.num_vlm_layers is not None
            and self.cfg.num_vlm_layers >= 0
        ):
            language_model.layers = language_model.layers[
                : self.cfg.num_vlm_layers
            ]

    def _get_vlm_num_hidden_states(self, language_model) -> int:
        return len(language_model.layers) + 1

    def _get_feat_mapping_input_dim(self, language_model) -> int:
        return language_model.config.hidden_size

    def _get_decoder_embed_dims(self) -> int:
        if self.cfg.embed_dims is not None:
            return self.cfg.embed_dims
        elif self.decoder is not None:
            return self.decoder.embed_dims

        raise ValueError("`embed_dims` must be configured without a decoder.")

    def _init_feat_mapping(self, language_model):
        embed_dims = self._get_decoder_embed_dims()
        self.feat_mapping = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self._get_feat_mapping_input_dim(language_model),
                    embed_dims,
                    bias=True,
                    dtype=torch.bfloat16,
                )
                for _ in self.action_feature_layer_indices
            ]
        )

    def _init_action_feature_weight(
        self, origin_num_layers: int, num_hidden_states: int
    ):
        temperature = 3
        highlighted_layer = origin_num_layers // 2  # the highlighted layer
        weight = torch.cat(
            [
                torch.linspace(0.1, 1, highlighted_layer + 1),
                torch.linspace(1, 0.1, origin_num_layers - highlighted_layer),
            ]
        )[:num_hidden_states]
        weight = weight[list(self.action_feature_layer_indices)]
        weight = weight.to(dtype=torch.bfloat16) * temperature
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def _get_qwen_patch_size(self) -> int:
        vision_config = self.vlm.config.vision_config
        patch_size = vision_config.patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[-1]
        return int(patch_size) * int(vision_config.spatial_merge_size)

    def save_model(
        self, directory: str, model_prefix: str = "model", **kwargs
    ):
        if kwargs.get("accelerator") is not None:
            model_prefix = "model"
        super().save_model(directory, model_prefix, **kwargs)
        cfg = self.cfg.model_copy(deep=True)
        source = self._vlm_resource_source
        if cfg.save_model_with_vlm or os.path.isabs(source):
            cfg.vlm_pretrain = pretrained_export_path(source, model_prefix)
        if cfg.save_model_with_vlm:
            with export_pretrained_directory(
                directory, cfg.vlm_pretrain
            ) as target:
                self.vlm.save_pretrained(target)
                self.vlm_processor.save_pretrained(target)
        elif os.path.isabs(source):
            link_pretrained_directory(directory, cfg.vlm_pretrain, source)
        with open(
            os.path.join(directory, f"{model_prefix}.config.json"), "w"
        ) as f:
            f.write(cfg.model_dump_json(indent=4))

    def extract_feature_3d(self, inputs):
        input_3d = inputs.get(self.input_3d)
        if self.backbone_3d is not None and input_3d is not None:
            dtype = next(self.backbone_3d.parameters()).dtype
            input_3d = input_3d.to(dtype=dtype)
            if "depth" in self.input_3d and input_3d.dim() == 5:
                bs, num_cams = input_3d.shape[:2]
                input_3d = input_3d.flatten(end_dim=1)
            elif "depth" in self.input_3d:
                num_cams = 1
            feature_3d = self.backbone_3d(input_3d)
            if self.neck_3d is not None:
                feature_3d = self.neck_3d(feature_3d)
            if "depth" in self.input_3d:
                feature_3d = [
                    x.unflatten(0, (bs, num_cams)) for x in feature_3d
                ]
        else:
            feature_3d = None
        return feature_3d

    def foward_feat_mapping(self, x):
        if isinstance(x, (list, tuple)):
            x = [x[index] for index in self.action_feature_layer_indices]
            x = torch.stack(x, dim=0)
        weight = torch.stack(
            [layer.weight for layer in self.feat_mapping], dim=0
        )
        weight = weight[:, None]
        x = x @ weight.mT

        bias = torch.stack([layer.bias for layer in self.feat_mapping], dim=0)
        bias = bias[:, None, None]
        x = x + bias
        x = x.permute(1, 2, 3, 0)
        x = x @ torch.nn.functional.softmax(self.weight, dim=0)
        return x

    def _forward_decoder(self, inputs, vlm_context):
        (
            feature_maps,
            feature_3d,
            text_dict,
            depth_prob,
            loss_depth,
        ) = self._extract_decoder_features(inputs, vlm_context)
        model_outs = self.decoder(
            feature_maps=feature_maps,
            feature_3d=feature_3d,
            text_dict=text_dict,
            inputs=inputs,
            depth_prob=depth_prob,
        )
        return model_outs, feature_maps, text_dict, loss_depth

    def _forward_vlm(self, output_hidden_states: bool = True, **vlm_inputs):
        input_ids = vlm_inputs.get("input_ids")
        if (
            input_ids is not None
            and "mm_token_type_ids" not in vlm_inputs
            and (
                vlm_inputs.get("image_grid_thw") is not None
                or vlm_inputs.get("video_grid_thw") is not None
            )
        ):
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.vlm.config.image_token_id] = 1
            video_token_id = getattr(self.vlm.config, "video_token_id", None)
            if video_token_id is not None:
                mm_token_type_ids[input_ids == video_token_id] = 2
            vlm_inputs["mm_token_type_ids"] = mm_token_type_ids

        return self.vlm.model(
            **vlm_inputs,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(*args, destination, prefix, keep_vars)
        if self.use_state_dict_with_vlm:
            return state_dict
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("vlm.")]
        for k in keys_to_remove:
            del state_dict[k]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if self.use_state_dict_with_vlm:
            return super().load_state_dict(state_dict, strict=strict, **kwargs)
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("vlm.")
        }
        incompatible_keys = super().load_state_dict(
            filtered_state_dict, strict=False, **kwargs
        )
        missing_keys = []
        for key in incompatible_keys.missing_keys:
            if not key.startswith("vlm."):
                missing_keys.append(key)
        incompatible_keys = type(incompatible_keys)(
            missing_keys, incompatible_keys.unexpected_keys
        )
        if strict:
            assert (
                len(incompatible_keys.missing_keys) == 0
                and len(incompatible_keys.unexpected_keys) == 0
            ), (
                "Unexpected key(s) in state_dict: {}. Missing key(s) in state_dict: {}.".format(  # noqa: E501
                    ", ".join(
                        f'"{k}"' for k in incompatible_keys.unexpected_keys
                    ),
                    ", ".join(
                        f'"{k}"' for k in incompatible_keys.missing_keys
                    ),
                )
            )
        return incompatible_keys

    def _vlm_outputs_handler(
        self,
        vlm_outputs,
        vlm_inputs,
        inputs,
    ):
        vlm_input_ids = vlm_inputs["input_ids"]
        batch_size, num_cams, _, h, w = inputs["imgs"].shape
        vlm_outputs = vlm_outputs["hidden_states"]

        vlm_outputs = checkpoint(
            self.foward_feat_mapping, vlm_outputs, use_reentrant=False
        )
        # vlm_outputs = self.foward_feat_mapping(vlm_outputs)
        vlm_outputs = vlm_outputs.to(torch.float32)

        img_token_mask = self.vlm.config.image_token_id == vlm_input_ids
        img_feature_mask = img_token_mask
        h_, w_ = h // self.qwen_patch_size, w // self.qwen_patch_size

        img_feature = vlm_outputs[img_feature_mask].unflatten(
            0, (batch_size, -1)
        )
        feature_maps = [
            img_feature.reshape(batch_size, num_cams, h_, w_, -1).permute(
                0, 1, 4, 2, 3
            )
        ]

        non_main_feature_mask = ~img_feature_mask
        text_feature = vlm_outputs[non_main_feature_mask]
        text_feature = text_feature.unflatten(0, (batch_size, -1))
        not_pad_mask = (
            vlm_input_ids != self.vlm_processor.tokenizer.pad_token_id
        )
        text_feature_mask = non_main_feature_mask & not_pad_mask
        compressed_text_mask = text_feature_mask[non_main_feature_mask]
        compressed_text_mask = compressed_text_mask.unflatten(
            0, (batch_size, -1)
        )

        text_dict = dict(
            embedded=text_feature, text_token_mask=compressed_text_mask
        )
        return feature_maps, text_dict

    def _extract_decoder_features(self, inputs, vlm_context):
        vlm_outputs = vlm_context["outputs"]
        vlm_inputs = vlm_context["vlm_inputs"]
        feature_maps, text_dict = self._vlm_outputs_handler(
            vlm_outputs,
            vlm_inputs,
            inputs,
        )

        feature_3d = self.extract_feature_3d(inputs)

        if self.spatial_enhancer is not None:
            feature_maps, depth_prob, loss_depth = self.spatial_enhancer(
                feature_maps=feature_maps,
                feature_3d=feature_3d,
                text_dict=text_dict,
                inputs=inputs,
            )
        else:
            depth_prob = loss_depth = None
        return feature_maps, feature_3d, text_dict, depth_prob, loss_depth


MODULE_TYPE = TorchModuleCfgType_co | DelayInitDictType


class HoloBrain_Qwen3_5_VLConfig(TorchModuleCfg[HoloBrain_Qwen3_5_VL]):  # noqa: N801
    """Architecture and local pretrained-resource configuration for 0.5."""

    class_type: ClassType_co[HoloBrain_Qwen3_5_VL] = HoloBrain_Qwen3_5_VL
    vlm_pretrain: str
    decoder: MODULE_TYPE
    spatial_enhancer: MODULE_TYPE | None = None
    data_preprocessor: MODULE_TYPE | None = None
    backbone_3d: MODULE_TYPE | None = None
    neck_3d: MODULE_TYPE | None = None
    embed_dims: int = 384
    input_2d: str = "imgs"
    input_3d: str = "depths"
    freeze_vlm: bool = False
    freeze_vision: bool = True
    lm_with_cp: bool = True
    vision_with_cp: bool = False
    use_state_dict_with_vlm: bool = True
    load_vlm_checkpoint: bool = True
    save_model_with_vlm: bool = False
    num_vlm_layers: int = 4
    attn_implementation: str = "flash_attention_2"
