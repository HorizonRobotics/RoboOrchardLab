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


config = dict(
    hist_steps=1,
    pred_steps=64,
    chunk_size=4,
    embed_dims=256,
    with_depth=True,
    with_depth_loss=True,
    min_depth=0.01,
    max_depth=1.2,
    num_depth=128,
    batch_size=16,
    max_step=int(1e5),
    step_log_freq=50,
    save_step_freq=5000,
    num_workers=8,
    lr=1e-4,
    training_with_subtask=False,
    with_cot=False,
    training_datasets=[
        "robotwin1_0",
        "robotwin2_0",
        "robotwin2_0_ur5_wsg",
        "robotwin2_0_arx_x5a",
        "robotwin2_0_franka_panda",
        # "robotwin2_0_piper",
        "challenge",
        "challenge_finetune",
        "challenge_self_collect",
        "horizon_beijing",
        "horizon_shanghai",
        "agilex",
        # "rh20t",
        "agibot_alpha",
        "agibot_beta",
    ],
    deploy_datasets=[
        "horizon_beijing",
        "horizon_shanghai",
        "robotwin2_0",
    ],
    vlm_pretrain="./ckpt/Qwen2.5-VL-3B-Instruct",
)


# v2.0
# config.update(
#     num_vlm_layers=0,
#     checkpoint="http://pfs-svcspawner.bcloud-bj-zone1.hobot.cc/user/homespace/xuewu.lin/plat_gpu/2025-08-25/16-58/sem_alldata_layers0_resume2-20250825-153143.959699-COPY/output/checkpoints/checkpoint_15/model.safetensors",  # pretrain  # noqa: E501
# )


def build_model(config):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )
    from torch import nn

    from robo_orchard_lab.models.bip3d.spatial_enhancer import (
        BatchDepthProbGTGenerator,
        DepthFusionSpatialEnhancer,
    )
    from robo_orchard_lab.models.layers.data_preprocessors import (
        BaseDataPreprocessor,
    )
    from robo_orchard_lab.models.layers.transformer_layers import (
        FFN,
    )
    from robo_orchard_lab.models.modules.swin_transformer import (
        SwinTransformer,
    )
    from robo_orchard_lab.models.sem_modules import (
        AdaRMSNorm,
        JointGraphAttention,
        RotaryAttention,
        SEM_Qwen2_5_VL,
        SEM_Qwen2_5_VLConfig,
        SEMActionDecoder,
        SEMRobotStateEncoder,
        TemporalJointGraphAttention,
        TextTemplate,
        UpsampleHead,
    )

    embed_dims = config["embed_dims"]
    decoder_norm = nn.RMSNorm

    num_chunk = config["pred_steps"] // config["chunk_size"]
    state_dims = 8  # [joint_angle, x, y, z, qw, qx, qy, qz]
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

    decoder_operation_order = [
        "t_norm",
        # "joint_self_attn",
        # "gate_msa",
        # "norm",
        # "temp_cross_attn",
        "temp_joint_attn",
        "gate_msa",
        "norm",
        "img_cross_attn",
        "norm",
        "text_cross_attn",
        "norm",
        "scale_shift",
        "ffn",
        "gate_mlp",
    ] * config.get("decoder_layers", 6)

    model = SEM_Qwen2_5_VL(
        cfg=SEM_Qwen2_5_VLConfig(
            use_state_dict_with_vlm=False,
            with_cot=config["with_cot"],
            vlm_pretrain=config["vlm_pretrain"],
            num_vlm_layers=config.get("num_vlm_layers"),
            data_preprocessor=dict(
                type=BaseDataPreprocessor,
                # input image should in BGR convention, it will be converted to RGB here  # noqa: E501
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
                        stride=(28,),
                    ),
                    dict(
                        type=TextTemplate,
                        with_subtask=config["training_with_subtask"],
                    ),
                ],
            ),
            backbone_3d=(
                dict(
                    type=SwinTransformer,
                    in_channels=1,
                    embed_dims=32,
                    depths=[2, 6, 2],
                    num_heads=[2, 4, 8],
                    window_size=8,
                    patch_size=7,
                    strides=[7, 2, 2],
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
                else None
            ),
            spatial_enhancer=dict(
                type=DepthFusionSpatialEnhancer,
                embed_dims=embed_dims,
                feature_3d_dim=128,
                num_depth_layers=2,
                min_depth=config["min_depth"],
                max_depth=config["max_depth"],
                num_depth=config["num_depth"],
                with_feature_3d=config.get("with_depth"),
                loss_depth_weight=(
                    config.get("loss_depth_weight", 1.0)
                    if config.get("with_depth_loss")
                    else -1
                ),
            ),
            decoder=dict(
                type=SEMActionDecoder,
                embed_dims=embed_dims,
                head=head,
                img_cross_attn=dict(
                    type=RotaryAttention,
                    embed_dims=embed_dims,
                    num_heads=8,
                    max_position_embeddings=32,
                ),
                temp_joint_attn=dict(
                    type=TemporalJointGraphAttention,
                    embed_dims=embed_dims,
                    num_heads=8,
                    max_position_embeddings=32,
                ),
                norm_layer=dict(
                    type=decoder_norm,
                    normalized_shape=embed_dims,
                ),
                ffn=dict(
                    type=FFN,
                    embed_dims=embed_dims,
                    feedforward_channels=2048,
                    act_cfg=dict(type=nn.SiLU, inplace=True),
                ),
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
                joint_self_attn=dict(
                    type=JointGraphAttention,
                    embed_dims=embed_dims,
                    num_heads=8,
                ),
                temp_cross_attn=dict(
                    type=RotaryAttention,
                    embed_dims=embed_dims,
                    num_heads=8,
                    max_position_embeddings=32,
                ),
                text_cross_attn=dict(
                    type=RotaryAttention,
                    embed_dims=embed_dims,
                    num_heads=8,
                    max_position_embeddings=512,
                ),
                pred_steps=config["pred_steps"],
                timestep_norm_layer=dict(
                    type=AdaRMSNorm,
                    normalized_shape=embed_dims,
                    condition_dims=256,
                    zero=True,
                ),
                operation_order=decoder_operation_order,
                feature_level=[0],
                act_cfg=dict(type=nn.SiLU, inplace=True),
                robot_encoder=dict(
                    type=SEMRobotStateEncoder,
                    embed_dims=embed_dims,
                    chunk_size=min(8, config["hist_steps"]),
                    joint_self_attn=dict(
                        type=JointGraphAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                    ),
                    norm_layer=dict(
                        type=decoder_norm,
                        normalized_shape=embed_dims,
                    ),
                    ffn=dict(
                        type=FFN,
                        embed_dims=embed_dims,
                        feedforward_channels=2048,
                        act_cfg=dict(type=nn.SiLU, inplace=True),
                    ),
                    temp_self_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=32,
                    ),
                    act_cfg=dict(type=nn.SiLU, inplace=True),
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
                    state_dims=state_dims,
                ),
                state_dims=state_dims,
            ),
        )
    )
    return model


def build_training_dataset(config, lazy_init=False):
    from config_agibot_dataset import build_datasets as build_agibot_datasets
    from config_agilex_dataset import build_datasets as build_agilex_datasets
    from config_rh20t_dataset import build_datasets as build_rh20t_datasets
    from config_robotwin_dataset import (
        build_datasets as build_robotwin_datasets,
    )

    from robo_orchard_lab.dataset.dataset_wrapper import ConcatDatasetWithFlag

    datasets = []
    datasets.extend(
        build_robotwin_datasets(
            config,
            config["training_datasets"],
            mode="training",
            lazy_init=lazy_init,
        )
    )
    datasets.extend(
        build_agilex_datasets(
            config,
            config["training_datasets"],
            mode="training",
            lazy_init=lazy_init,
        )
    )
    datasets.extend(
        build_rh20t_datasets(
            config,
            config["training_datasets"],
            mode="training",
            lazy_init=lazy_init,
        )
    )
    datasets.extend(
        build_agibot_datasets(
            config,
            config["training_datasets"],
            mode="training",
            lazy_init=lazy_init,
        )
    )
    dataset = ConcatDatasetWithFlag(datasets=datasets)
    return dataset


def build_optimizer(config, model):
    import torch
    from torch import optim

    base_lr = config["lr"]
    max_step = config["max_step"]

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
    optimizer = optim.AdamW(
        [
            # {"params": vlm_params, "lr": base_lr * 0.1},
            {"params": bit16_params},
            {"params": other_params},
        ],
        lr=base_lr,
        weight_decay=config.get("weight_decay", 0.0005),
    )
    lr_scheduler = optim.lr_scheduler.ChainedScheduler(
        [
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, total_iters=500
            ),
            optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(max_step * 0.9)],
                gamma=0.1,
            ),
        ]
    )
    return optimizer, lr_scheduler


def build_processors(config):
    from config_agilex_dataset import (
        build_processors as build_agilex_processors,
    )
    from config_robotwin_dataset import (
        build_processors as build_robotwin_processors,
    )

    processors = build_agilex_processors(config, config["deploy_datasets"])
    processors.update(
        build_robotwin_processors(config, config["deploy_datasets"])
    )
    return processors
