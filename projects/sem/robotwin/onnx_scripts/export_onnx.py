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

import argparse
import logging
import os
from functools import partial

import numpy as np
import torch
from misc import (
    DECODER_INPUT_NAMES,
    DECODER_OUTPUT_NAMES,
    ENCODER_INPUT_NAMES,
    ENCODER_OUTPUT_NAMES,
    get_pixels,
    unsqueeze_batch,
)
from onnx_model import OnnxSEMModel, OnnxSEMModelConfig
from torch import nn
from utils import load_config

from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.models.sem_modules.layers import AdaRMSNorm, rotate_half
from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__file__)


class DeployDataPreprocessor:
    def __init__(
        self,
        model,
        num_joint=14,
        pred_steps=64,
        do_unsqueeze_batch=True,
        get_noisy_action=True,
    ):
        self.preprocessor = model.data_preprocessor.eval()
        strides = model.cfg.data_preprocessor["batch_transforms"][0]["stride"]
        strides = tuple(strides[i] for i in model.decoder.feature_level)
        self.strides = strides
        self.pixels = None
        self.num_joint = num_joint
        self.pred_steps = pred_steps
        self.do_unsqueeze_batch = do_unsqueeze_batch
        self.get_noisy_action = get_noisy_action

    def __call__(self, data):
        data["projection_mat_inv"] = torch.linalg.inv(data["projection_mat"])
        if self.do_unsqueeze_batch:
            data = unsqueeze_batch(data)
        data = self.preprocessor(data)
        if self.pixels is None:
            self.pixels = get_pixels(data, self.strides)
        data["pixels"] = self.pixels.clone()
        if self.get_noisy_action:
            noise = torch.randn((self.pred_steps, self.num_joint)).to(
                data["imgs"]
            )
            noisy_action = data["kinematics"][0].joint_state_to_robot_state(
                noise, data.get("embodiedment_mat")[0]
            )
            data["noisy_action"] = noisy_action.unsqueeze(dim=0)
        return data


def rms_norm(layer, x):
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    if layer.eps is not None:
        rms = rms + layer.eps
    rms = rms.sqrt()
    x = x / rms
    if layer.weight is not None:
        x = x * layer.weight
    return x


def ada_rms_norm(layer, x, c):
    x = rms_norm(layer, x)
    return layer.apply_ada_func(x, c)


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        model.cuda().eval().requires_grad_(False)
        self.backbone = model.backbone
        self.neck = model.neck
        self.backbone_3d = model.backbone_3d
        self.neck_3d = model.neck_3d
        self.spatial_enhancer = model.spatial_enhancer
        self.robot_encoder = model.decoder.robot_encoder
        self.feature_level = model.decoder.feature_level
        for layer in self.robot_encoder.layers:
            if isinstance(layer, torch.nn.RMSNorm):
                layer.forward = partial(rms_norm, layer)

    def extract_feat(self, imgs, depths=None):
        if imgs.dim() == 5:
            bs, num_cams = imgs.shape[:2]
            imgs = imgs.flatten(end_dim=1)
        else:
            bs = imgs.shape[0]
            num_cams = 1

        feature_maps = self.backbone(imgs)
        if self.neck is not None:
            feature_maps = self.neck(feature_maps)
        feature_maps = [x.unflatten(0, (bs, num_cams)) for x in feature_maps]
        feature_maps = [feature_maps[i] for i in self.feature_level]
        feature_maps = self.flat_feature_maps(feature_maps)

        if self.backbone_3d is not None:
            assert depths.shape[1] == num_cams
            depths = depths.flatten(end_dim=1)
            feature_3d = self.backbone_3d(depths)
            if self.neck_3d is not None:
                feature_3d = self.neck_3d(feature_3d)
            feature_3d = [x.unflatten(0, (bs, num_cams)) for x in feature_3d]
            feature_3d = [feature_3d[i] for i in self.feature_level]
            feature_3d = self.flat_feature_maps(feature_3d)
        else:
            feature_3d = None
        return feature_maps, feature_3d

    def forward(
        self,
        imgs,
        depths,
        pixels,
        projection_mat_inv,
        hist_robot_state,
        joint_scale_shift,
        joint_relative_pos,
    ):
        feature_maps, feature_3d = self.extract_feat(imgs, depths)

        feature_maps = self.forward_spatial_enhancer(
            feature_maps, feature_3d, pixels, projection_mat_inv
        )

        hist_robot_state = torch.cat(
            [
                (
                    hist_robot_state[..., :1]
                    - joint_scale_shift[:, None, :, 1:2]
                )
                / joint_scale_shift[:, None, :, :1],
                hist_robot_state[..., 1:],
            ],
            dim=-1,
        )
        robot_feature = self.robot_encoder(
            hist_robot_state, joint_relative_pos
        )
        return feature_maps, robot_feature

    def flat_feature_maps(self, feature_maps):
        feature_maps = torch.cat(
            [x.flatten(start_dim=-2).transpose(-1, -2) for x in feature_maps],
            dim=-2,
        )
        return feature_maps

    def forward_spatial_enhancer(
        self, feature_2d, feature_3d, pixels, projection_mat_inv
    ):
        pts = self.get_pts(pixels, projection_mat_inv)

        se = self.spatial_enhancer
        if se.with_feature_3d:
            depth_prob_feat = se.pts_prob_pre_fc(feature_2d)
            depth_prob_feat = torch.cat([depth_prob_feat, feature_3d], dim=-1)
            depth_prob = se.pts_prob_fc(depth_prob_feat).softmax(dim=-1)
            feature_fused = [feature_2d, feature_3d]
        else:
            depth_prob = se.pts_prob_fc(feature_2d).softmax(dim=-1)
            feature_fused = [feature_2d]

        pts_feature = se.pts_fc(pts)
        pts_feature = (depth_prob.unsqueeze(dim=-1) * pts_feature).sum(dim=-2)
        feature_fused.append(pts_feature)
        feature_fused = torch.cat(feature_fused, dim=-1)
        feature_fused = se.fusion_fc(feature_fused) + feature_2d
        feature_fused = se.fusion_norm(feature_fused)
        return feature_fused

    def get_pts(self, pixels, projection_mat_inv):
        depths = torch.linspace(
            self.spatial_enhancer.min_depth,
            self.spatial_enhancer.max_depth,
            self.spatial_enhancer.num_depth,
        ).to(pixels)  # D
        depths = depths[None, :, None]  # 1 D 1
        pts = pixels * depths  # N 1 2 x 1 D 1 = N D 2
        depths = depths.tile(pixels.shape[0], 1, 1)  # N D 1
        pts = torch.cat([pts, depths, torch.ones_like(depths)], dim=-1)  # ND4
        pts = (
            pts
            @ projection_mat_inv.permute(0, 1, 3, 2).unsqueeze(dim=2)[..., :3]
        )  # ND4 x bc144
        return pts


class Decoder(nn.Module):
    def __init__(self, model, hist_steps=1, num_joint=14):
        super().__init__()
        model.cuda().eval().requires_grad_(False)
        self.decoder = model.decoder
        for op, layer in zip(
            self.decoder.operation_order, self.decoder.layers, strict=False
        ):
            if isinstance(layer, AdaRMSNorm):
                layer.forward = partial(ada_rms_norm, layer)
            elif isinstance(layer, torch.nn.RMSNorm):
                layer.forward = partial(rms_norm, layer)
            elif op == "temp_cross_attn":
                num_pred_chunk = (
                    self.decoder.pred_steps // self.decoder.chunk_size
                )
                num_hist_chunk = (
                    hist_steps // self.decoder.robot_encoder.chunk_size
                )
                query_pos = (
                    torch.arange(num_pred_chunk)[None].tile(num_joint, 1)
                    + num_hist_chunk
                )
                key_pos = torch.arange(num_pred_chunk + num_hist_chunk)[
                    None
                ].tile(num_joint, 1)
                layer.cos_query = layer.position_encoder.cos[
                    query_pos
                ].unsqueeze(1)
                layer.sin_query = layer.position_encoder.sin[
                    query_pos
                ].unsqueeze(1)
                layer.cos_key = layer.position_encoder.cos[key_pos].unsqueeze(
                    1
                )
                layer.sin_key = layer.position_encoder.sin[key_pos].unsqueeze(
                    1
                )

                def apply_position_encode(layer, q, query_pos, k, key_pos):
                    q = (q * layer.cos_query.to(q)) + (
                        rotate_half(q) * layer.sin_query.to(q)
                    )
                    k = (k * layer.cos_key.to(k)) + (
                        rotate_half(k) * layer.sin_key.to(k)
                    )
                    return q, k

                layer.apply_position_encode = partial(
                    apply_position_encode, layer
                )

        for layers in self.decoder.head.act_and_norm:
            for layer in layers:
                if isinstance(layer, torch.nn.RMSNorm):
                    layer.forward = partial(rms_norm, layer)

    def forward(
        self,
        noisy_action,
        image_feature,
        robot_feature,
        timestep,
        joint_relative_pos,
    ):
        image_feature = image_feature.flatten(1, 2)
        pred = self.decoder.forward_layers(
            noisy_action,
            image_feature,
            robot_feature=robot_feature,
            timesteps=timestep,
            joint_relative_pos=joint_relative_pos,
        )
        return pred


def export_onnx(config, model, num_joint, output_path, validate=True):
    model = ModelMixin.load_model(model).eval().cuda().requires_grad_(False)

    config = load_config(config)
    base_config = config.config
    if base_config["multi_task"]:
        raise NotImplementedError(
            "Only single-task model export is currently supported"
        )
    build_dataset = config.build_dataset
    dataset = build_dataset(base_config)[0]

    data_processor = DeployDataPreprocessor(
        model,
        num_joint=num_joint,
        pred_steps=base_config["pred_steps"],
        do_unsqueeze_batch=True,
    )
    data = dataset[0]
    data = data_processor(data)

    encoder = Encoder(model)
    decoder = Decoder(model, base_config["hist_steps"], num_joint)
    image_feature, robot_feature = encoder(
        *[data[key] for key in ENCODER_INPUT_NAMES]
    )

    os.makedirs(output_path, exist_ok=True)
    opset_version = 19
    encoder_onnx = os.path.join(output_path, "encoder.onnx")
    decoder_onnx = os.path.join(output_path, "decoder.onnx")

    # export encoder onnx
    torch.onnx.export(
        encoder,
        tuple(data[key] for key in ENCODER_INPUT_NAMES),
        encoder_onnx,
        input_names=ENCODER_INPUT_NAMES,
        output_names=ENCODER_OUTPUT_NAMES,
        opset_version=opset_version,
    )

    # export decoder onnx
    timestep = torch.tensor([999]).to(data["imgs"])
    torch.onnx.export(
        decoder,
        (
            data["noisy_action"],
            image_feature,
            robot_feature,
            timestep,
            data["joint_relative_pos"],
        ),
        decoder_onnx,
        input_names=DECODER_INPUT_NAMES,
        output_names=DECODER_OUTPUT_NAMES,
        opset_version=opset_version,
    )

    # export OnnxSEMModel config json
    onnx_model = OnnxSEMModel(
        OnnxSEMModelConfig(
            data_preprocessor=model.cfg.data_preprocessor,
            onnx_path=output_path,
            test_noise_scheduler=model.cfg.decoder["test_noise_scheduler"],
            num_inference_timesteps=model.cfg.decoder[
                "num_inference_timesteps"
            ],
            strides=data_processor.strides,
        )
    )
    onnx_model.save_model(output_path)

    if validate:
        image_feature_onnx, robot_feature_onnx = (
            onnx_model.encoder_session.run(
                ENCODER_OUTPUT_NAMES,
                {key: data[key].cpu().numpy() for key in ENCODER_INPUT_NAMES},
            )
        )

        decoder_pred = decoder(
            data["noisy_action"],
            image_feature,
            robot_feature,
            timestep,
            data["joint_relative_pos"],
        )
        decoder_pred_onnx = onnx_model.decoder_session.run(
            DECODER_OUTPUT_NAMES,
            {
                "noisy_action": data["noisy_action"].cpu().numpy(),
                "image_feature": image_feature_onnx,
                "robot_feature": robot_feature_onnx,
                "timestep": timestep.cpu().numpy(),
                "joint_relative_pos": data["joint_relative_pos"].cpu().numpy(),
            },
        )[0]

        diff_image_feature = np.abs(
            image_feature_onnx - image_feature.cpu().numpy()
        )
        diff_robot_feature = np.abs(
            robot_feature_onnx - robot_feature.cpu().numpy()
        )
        diff_decoder_pred = np.abs(
            decoder_pred_onnx - decoder_pred.cpu().numpy()
        )
        logger.info(
            "Diff of image_feature: "
            f"max_diff[{diff_image_feature.max()}] "
            f"mean[{diff_image_feature.mean()}]"
        )
        logger.info(
            "Diff of robot_feature: "
            f"max_diff[{diff_robot_feature.max()}] "
            f"mean[{diff_robot_feature.mean()}]"
        )
        logger.info(
            "Diff of decoder_pred: "
            f"max_diff[{diff_decoder_pred.max()}] "
            f"mean[{diff_decoder_pred.mean()}]"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_path", type=str, default="./onnx_files")
    parser.add_argument("--num_joint", type=int, default=14)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    export_onnx(
        args.config,
        args.model,
        args.num_joint,
        args.output_path,
        args.validate,
    )
