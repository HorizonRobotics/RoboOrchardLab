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


from projects.holobrain_0_5.configs.dataset_factory import (
    build_processors,
    build_training_dataset,
    build_validation_dataset,
    collate_action_batch,
)

config = dict(
    hist_steps=1,
    pred_steps=64,
    chunk_size=4,
    embed_dims=384,
    num_vlm_layers=4,
    decoder_layers=10,
    dst_wh=(352, 256),
    with_depth=True,
    with_depth_loss=True,
    min_depth=0.01,
    max_depth=1.2,
    num_depth=128,
    batch_size=16,
    max_step=100000,
    lr=1e-4,
    warmup_step=500,
    lr_decay_milestones=[0.9],
    step_log_freq=100,
    save_step_freq=5000,
    total_limit=3,
    resume_from=None,
    num_workers=8,
    vlm_pretrain="Qwen3.5-2B",
    checkpoint="hf://model/HorizonRobotics/HoloBrain_v0.5/model.safetensors",
    freeze_vlm=False,
    dataset_specs="projects.holobrain_0_5.configs.dataset_specs",
    deploy_specs="projects.holobrain_0_5.configs.deploy_specs",
    training_datasets=["robotwin2_0"],
    validation_datasets=[],
)
__all__ = [
    "config",
    "collate_fn",
    "build_model",
    "build_optimizer",
    "build_training_dataset",
    "build_validation_dataset",
    "build_processors",
]
collate_fn = collate_action_batch


def build_model(config):
    """Build the HoloBrain 0.5 action model with local Qwen3.5 resources."""
    import math

    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )
    from torch import nn

    from robo_orchard_lab.models.bip3d.spatial_enhancer import (
        BatchDepthProbGTGenerator,
        DepthFusionSpatialEnhancer,
    )
    from robo_orchard_lab.models.holobrain_0_5.action_decoder import (
        HoloBrainActionDecoder,
        HoloBrainDecoderBaseConfig,
        HoloBrainDecoderTransformerConfig,
        HoloBrainTrainingConfig,
    )
    from robo_orchard_lab.models.holobrain_0_5.layers import (
        ActionFFN,
        AdaRMSNorm,
        JointGraphAttention,
        MultiModalAttention,
        RotaryAttention,
        TemporalJointGraphAttention,
        UpsampleHead,
    )
    from robo_orchard_lab.models.holobrain_0_5.loss import HoloBrainActionLoss
    from robo_orchard_lab.models.holobrain_0_5.robot_state_encoder import (
        HoloBrainEncoderBaseConfig,
        HoloBrainEncoderTransformerConfig,
        HoloBrainRobotStateEncoder,
    )
    from robo_orchard_lab.models.holobrain_0_5.structure import (
        HoloBrain_Qwen3_5_VL,
        HoloBrain_Qwen3_5_VLConfig,
        TextTemplate,
    )
    from robo_orchard_lab.models.layers.data_preprocessors import (
        BaseDataPreprocessor,
    )
    from robo_orchard_lab.models.modules.swin_transformer import (
        SwinTransformer,
    )

    patch_size = 32
    model_class = HoloBrain_Qwen3_5_VL
    model_config = HoloBrain_Qwen3_5_VLConfig
    embed_dims = config["embed_dims"]
    decoder_norm = nn.RMSNorm
    num_chunk = config["pred_steps"] // config["chunk_size"]
    state_dims = 8
    head = dict(
        type=UpsampleHead,
        upsample_sizes=[num_chunk * 2, config["pred_steps"]],
        input_dim=embed_dims,
        dims=[128, 64],
        norm=dict(type=decoder_norm, normalized_shape=embed_dims),
        act=dict(type=nn.SiLU, inplace=True),
        norm_act_idx=[0, 1],
        num_output_layers=2,
        out_dim=state_dims,
    )
    decoder_layers = config.get("decoder_layers", 6)
    attn_output_proj_init_cfg = dict(
        output_proj_init="normal",
        output_proj_std=0.02 / math.sqrt(2 * decoder_layers),
    )
    num_heads = config.get("num_heads", 8)
    decoder_operation_order = [
        "t_norm",
        "multi_modal_attn",
        "gate_msa",
        "norm",
        "scale_shift",
        "ffn",
        "gate_mlp",
    ] * decoder_layers
    model = model_class(
        cfg=model_config(
            embed_dims=embed_dims,
            vlm_pretrain=config["vlm_pretrain"],
            attn_implementation=config.get(
                "attn_implementation", "flash_attention_2"
            ),
            num_vlm_layers=config.get("num_vlm_layers"),
            freeze_vlm=config.get("freeze_vlm", True),
            use_state_dict_with_vlm=not config.get("freeze_vlm", True),
            data_preprocessor=dict(
                type=BaseDataPreprocessor,
                channel_flip=True,
                unsqueeze_depth_channel=True,
                batch_transforms=[
                    dict(
                        type=BatchDepthProbGTGenerator,
                        min_depth=config["min_depth"],
                        max_depth=config["max_depth"],
                        num_depth=config["num_depth"],
                        origin_stride=2,
                        valid_threshold=0.5,
                        stride=(patch_size,),
                    ),
                    dict(type=TextTemplate),
                ],
            ),
            backbone_3d=dict(
                type=SwinTransformer,
                in_channels=1,
                embed_dims=32,
                depths=[2, 6, 2],
                num_heads=[2, 4, 8],
                window_size=8,
                patch_size=patch_size // 4,
                strides=[patch_size // 4, 2, 2],
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                out_indices=(2,),
                with_cp=True,
                convert_weights=False,
            )
            if config.get("with_depth")
            else None,
            spatial_enhancer=dict(
                type=DepthFusionSpatialEnhancer,
                embed_dims=embed_dims,
                feature_3d_dim=128,
                num_depth_layers=2,
                min_depth=config["min_depth"],
                max_depth=config["max_depth"],
                num_depth=config["num_depth"],
                with_feature_3d=config.get("with_depth"),
                loss_depth_weight=config.get("loss_depth_weight", 1.0)
                if config.get("with_depth_loss")
                else -1,
            ),
            decoder=dict(
                type=HoloBrainActionDecoder,
                head=head,
                transformer_cfg=HoloBrainDecoderTransformerConfig(
                    img_cross_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        max_position_embeddings=32,
                        **attn_output_proj_init_cfg,
                    ),
                    norm_layer=dict(
                        type=decoder_norm, normalized_shape=embed_dims
                    ),
                    ffn=dict(
                        type=ActionFFN,
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims * 8,
                        act_cfg=dict(type=nn.SiLU, inplace=True),
                    ),
                    text_cross_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        max_position_embeddings=512,
                        **attn_output_proj_init_cfg,
                    ),
                    timestep_norm_layer=dict(
                        type=AdaRMSNorm,
                        normalized_shape=embed_dims,
                        condition_dims=256,
                        zero=True,
                        ada_init="identity",
                        gate_bias=1.0,
                    ),
                    multi_modal_attn=dict(
                        type=MultiModalAttention,
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        img_cross_attn=dict(
                            type=RotaryAttention,
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            max_position_embeddings=32,
                            **attn_output_proj_init_cfg,
                        ),
                        text_cross_attn=dict(
                            type=RotaryAttention,
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            max_position_embeddings=512,
                            **attn_output_proj_init_cfg,
                        ),
                        temp_joint_attn=dict(
                            type=TemporalJointGraphAttention,
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            max_position_embeddings=32,
                            **attn_output_proj_init_cfg,
                        ),
                    ),
                    operation_order=decoder_operation_order,
                ),
                base_cfg=HoloBrainDecoderBaseConfig(
                    chunk_size=config.get("chunk_size", 8),
                    use_joint_mask=True,
                    noise_type="local_joint",
                    pred_scaled_joint=False,
                    prediction_type="relative_joint_relative_pose",
                    pred_steps=config["pred_steps"],
                    state_dims=state_dims,
                    embed_dims=embed_dims,
                    training_noise_scheduler=dict(
                        type=DDPMScheduler,
                        num_train_timesteps=1000,
                        beta_schedule="squaredcos_cap_v2",
                        prediction_type="sample",
                        clip_sample=False,
                    ),
                    test_noise_scheduler=dict(
                        type=DPMSolverMultistepScheduler,
                        num_train_timesteps=1000,
                        beta_schedule="squaredcos_cap_v2",
                        prediction_type="sample",
                    ),
                    num_inference_timesteps=10,
                    feature_level=[0],
                    act_cfg=dict(type=nn.SiLU, inplace=True),
                ),
                training_cfg=HoloBrainTrainingConfig(
                    temporal_attn_drop=0.05,
                    num_parallel_training_sample=4,
                    teacher_forcing_rate=0.02,
                    loss=dict(
                        type=HoloBrainActionLoss,
                        timestep_loss_weight=1000,
                        parallel_loss_weight=0.1,
                        smooth_l1_beta=0.04,
                        loss_mode="smooth_l1",
                    ),
                ),
                robot_encoder=dict(
                    type=HoloBrainRobotStateEncoder,
                    transformer_cfg=HoloBrainEncoderTransformerConfig(
                        joint_self_attn=dict(
                            type=JointGraphAttention,
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            **attn_output_proj_init_cfg,
                        ),
                        norm_layer=dict(
                            type=decoder_norm, normalized_shape=embed_dims
                        ),
                        ffn=dict(
                            type=ActionFFN,
                            embed_dims=embed_dims,
                            feedforward_channels=2048,
                            act_cfg=dict(type=nn.SiLU, inplace=True),
                        ),
                        operation_order=[
                            "norm",
                            "joint_self_attn",
                            None,
                            None,
                            "norm",
                            "ffn",
                        ]
                        * 4
                        + ["norm"],
                    ),
                    base_cfg=HoloBrainEncoderBaseConfig(
                        embed_dims=embed_dims,
                        chunk_size=min(8, config["hist_steps"]),
                        act_cfg=dict(type=nn.SiLU, inplace=True),
                        state_dims=state_dims,
                    ),
                ),
            ),
        )
    )
    return model


def build_optimizer(config, model):
    """Build AdamW parameter groups and the per-update LR schedule."""
    import torch
    from torch import optim

    base_lr = config["lr"]
    max_step = config["max_step"]
    warmup_step = config.get("warmup_step", 500)

    vlm_params = []
    bit16_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "vlm." in name:
            if p.requires_grad:
                vlm_params.append(p)
        elif p.dtype == torch.float16 or p.dtype == torch.bfloat16:
            bit16_params.append(p)
        else:
            other_params.append(p)
    optim_params = [
        {"params": bit16_params},
        {"params": other_params},
    ]
    if len(vlm_params) > 0:
        optim_params.append(
            {"params": vlm_params, "lr": base_lr * 0.1},
        )
    optimizer = optim.AdamW(
        optim_params,
        lr=base_lr,
        weight_decay=config.get("weight_decay", 0.0005),
    )

    lr_decay_milestones = config.get("lr_decay_milestones", None)
    lr_scheduler = optim.lr_scheduler.ChainedScheduler(
        [
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, total_iters=warmup_step
            ),
        ]
        + (
            []
            if lr_decay_milestones is None
            else [
                optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        int(max_step * x) for x in lr_decay_milestones
                    ],
                    gamma=0.1,
                ),
            ]
        )
    )
    return optimizer, lr_scheduler
