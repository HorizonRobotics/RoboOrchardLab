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
    chunk_size=8,
    embed_dims=256,
    with_depth=True,
    with_depth_loss=True,
    min_depth=0.01,
    max_depth=1.2,
    num_depth=128,
    batch_size=8,
    max_step=int(1e5),
    step_log_freq=25,
    save_step_freq=4000,
    num_workers=8,
    lr=1e-4,
    checkpoint="./ckpt/groundingdino_swint_ogc_mmdet-822d7e9d-rename.pth",
    bert_checkpoint="./ckpt/bert-base-uncased",
    data_path="./data/lmdb",
    urdf="./urdf/arx5/arx5_description_isaac.urdf",
    multi_task=False,
    task_names=["place_empty_cup"],
)


# For horizon-released real-world data
# https://huggingface.co/datasets/HorizonRobotics/Real-World-Dataset
# config.update(
#     urdf="./urdf/piper_description_dualarm.urdf",
#     img_channel_flip=True,
#     scale_shift=[
#         [1.478021398, 0.10237011399999996],
#         [1.453678296, 1.4043815520000003],
#         [1.553963852, -1.5014923],
#         [1.86969153, -0.0010728060000000372],
#         [1.3381379620000002, -0.012585846000000012],
#         [3.086157592, -0.06803160000000008],
#         [0.03857, 0.036329999999999994],
#         [1.478021398, 0.10237011399999996],
#         [1.453678296, 1.4043815520000003],
#         [1.553963852, -1.5014923],
#         [1.86969153, -0.0010728060000000372],
#         [1.3381379620000002, -0.012585846000000012],
#         [3.086157592, -0.06803160000000008],
#         [0.03857, 0.036329999999999994],
#     ],
#     kinematics_config=dict(
#         left_arm_joint_id=list(range(6)),
#         right_arm_joint_id=list(range(8, 14)),
#         left_arm_link_keys=[
#             "left_link1",
#             "left_link2",
#             "left_link3",
#             "left_link4",
#             "left_link5",
#             "left_link6",
#         ],
#         right_arm_link_keys=[
#             "right_link1",
#             "right_link2",
#             "right_link3",
#             "right_link4",
#             "right_link5",
#             "right_link6",
#         ],
#         left_finger_keys=["left_link7"],
#         right_finger_keys=["right_link7"],
#     ),
#     T_base2world=[
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1.],
#     ],
# )


def build_model(config):
    import numpy as np
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )
    from torch import nn

    from robo_orchard_lab.models.bip3d.bert import BertModel
    from robo_orchard_lab.models.bip3d.feature_enhancer import (
        TextImageDeformable2DEnhancer,
    )
    from robo_orchard_lab.models.bip3d.spatial_enhancer import (
        BatchDepthProbGTGenerator,
        DepthFusionSpatialEnhancer,
    )
    from robo_orchard_lab.models.bip3d.structure import BIP3D, BIP3DConfig
    from robo_orchard_lab.models.layers.data_preprocessors import (
        BaseDataPreprocessor,
    )
    from robo_orchard_lab.models.layers.transformer_layers import FFN
    from robo_orchard_lab.models.modules.channel_mapper import ChannelMapper
    from robo_orchard_lab.models.modules.resnet import ResNet
    from robo_orchard_lab.models.modules.swin_transformer import (
        SwinTransformer,
    )
    from robo_orchard_lab.models.sem_modules import (
        AdaRMSNorm,
        JointGraphAttention,
        RotaryAttention,
        SEMActionDecoder,
        SEMRobotStateEncoder,
        UpsampleHead,
    )

    embed_dims = config["embed_dims"]
    decoder_norm = nn.RMSNorm

    multi_task = config["multi_task"]

    state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    state_dims = len(state_loss_weights)
    ee_state_loss_weights = np.array([[1, 2, 2, 2, 0.2, 0.2, 0.2, 0.2]])
    joint_state_loss_weights = np.array([state_loss_weights] * 6)
    state_loss_weights = np.concatenate(
        [
            joint_state_loss_weights,
            ee_state_loss_weights,
            joint_state_loss_weights,
            ee_state_loss_weights,
        ],
        axis=0,
    )
    state_loss_weights = state_loss_weights.tolist()

    num_chunk = config["pred_steps"] // config["chunk_size"]
    upsample_sizes = []
    head_dims = []
    dim = embed_dims
    while num_chunk < config["pred_steps"]:
        num_chunk *= 2
        dim //= 2
        upsample_sizes.append(min(num_chunk, config["pred_steps"]))
        if num_chunk >= config["pred_steps"]:
            head_dims.append(state_dims)
        else:
            head_dims.append(dim)

    if multi_task:
        num_feature_levels = 4
        depth_gt_stride = (8, 16, 32, 64)
    else:
        num_feature_levels = 3
        depth_gt_stride = (8, 16, 32)

    model = BIP3D(
        cfg=BIP3DConfig(
            data_preprocessor=dict(
                type=BaseDataPreprocessor,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                channel_flip=False,
                unsqueeze_depth_channel=True,
                batch_transforms=[
                    dict(
                        type=BatchDepthProbGTGenerator,
                        min_depth=config["min_depth"],
                        max_depth=config["max_depth"],
                        num_depth=config["num_depth"],
                        origin_stride=2,
                        valid_threshold=0.5,
                        stride=depth_gt_stride,
                    ),
                ],
            ),
            embed_dims=embed_dims,
            backbone=dict(
                type=SwinTransformer,
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                out_indices=(1, 2, 3),
                with_cp=True,
                convert_weights=False,
            ),
            neck=dict(
                type=ChannelMapper,
                in_channels=[192, 384, 768],
                kernel_size=1,
                out_channels=embed_dims,
                act_cfg=None,
                bias=True,
                norm_cfg=dict(type=nn.GroupNorm, num_groups=32),
                num_outs=num_feature_levels,
            ),
            backbone_3d=(
                dict(
                    type=ResNet,
                    depth=34,
                    in_channels=1,
                    base_channels=4,
                    num_stages=4,
                    out_indices=(1, 2, 3),
                    bn_eval=True,
                    with_cp=True,
                    style="pytorch",
                )
                if config.get("with_depth")
                else None
            ),
            neck_3d=(
                dict(
                    type=ChannelMapper,
                    in_channels=[8, 16, 32],
                    kernel_size=1,
                    out_channels=32,
                    act_cfg=None,
                    bias=True,
                    norm_cfg=dict(type=nn.GroupNorm, num_groups=4),
                    num_outs=num_feature_levels,
                )
                if config.get("with_depth")
                else None
            ),
            text_encoder=(
                dict(
                    type=BertModel,
                    special_tokens_list=["[CLS]", "[SEP]"],
                    name=config["bert_checkpoint"],
                    pad_to_max=False,
                    use_sub_sentence_represent=True,
                    add_pooling_layer=False,
                    max_tokens=768,
                    use_checkpoint=True,
                    return_tokenized=True,
                )
                if multi_task
                else None
            ),
            feature_enhancer=(
                dict(
                    type=TextImageDeformable2DEnhancer,
                    embed_dims=embed_dims,
                    num_layers=6,
                    text_img_attn_block=dict(
                        v_dim=embed_dims,
                        l_dim=embed_dims,
                        embed_dim=1024,
                        num_heads=4,
                        init_values=1e-4,
                    ),
                    img_attn_block=dict(
                        self_attn_cfg=dict(
                            embed_dims=embed_dims,
                            num_levels=num_feature_levels,
                            im2col_step=1,
                        ),
                        ffn_cfg=dict(
                            embed_dims=embed_dims,
                            feedforward_channels=2048,
                            ffn_drop=0.0,
                        ),
                    ),
                    text_attn_block=dict(
                        self_attn_cfg=dict(
                            num_heads=4,
                            embed_dims=embed_dims,
                        ),
                        ffn_cfg=dict(
                            embed_dims=embed_dims,
                            feedforward_channels=1024,
                            ffn_drop=0.0,
                        ),
                    ),
                    num_feature_levels=4,
                    positional_encoding=dict(
                        num_feats=embed_dims // 2,
                        normalize=True,
                        offset=0.0,
                        temperature=20,
                    ),
                )
                if multi_task
                else None
            ),
            spatial_enhancer=dict(
                type=DepthFusionSpatialEnhancer,
                embed_dims=embed_dims,
                feature_3d_dim=32,
                num_depth_layers=2,
                min_depth=config.get("min_depth", 0.25),
                max_depth=config.get("max_depth", 10),
                num_depth=config.get("num_depth", 64),
                with_feature_3d=config.get("with_depth"),
                loss_depth_weight=1.0 if config.get("with_depth_loss") else -1,
            ),
            decoder=dict(
                type=SEMActionDecoder,
                img_cross_attn=dict(
                    type=RotaryAttention,
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
                head=dict(
                    type=UpsampleHead,
                    upsample_sizes=upsample_sizes,
                    input_dim=embed_dims,
                    dims=head_dims,
                    norm=dict(type=decoder_norm, normalized_shape=embed_dims),
                    act=dict(type=nn.SiLU, inplace=True),
                    norm_act_idx=[0, 1, 2],
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
                    max_position_embeddings=256,
                ),
                pred_steps=config["pred_steps"],
                timestep_norm_layer=dict(
                    type=AdaRMSNorm,
                    normalized_shape=embed_dims,
                    condition_dims=256,
                    zero=True,
                ),
                operation_order=[
                    "t_norm",
                    "joint_self_attn",
                    "gate_msa",
                    "norm",
                    "temp_cross_attn",
                    "norm",
                    "img_cross_attn",
                    "norm",
                    *(
                        [
                            None,
                            None,
                        ]
                        if not multi_task
                        else [
                            "text_cross_attn",
                            "norm",
                        ]
                    ),
                    "scale_shift",
                    "ffn",
                    "gate_mlp",
                ]
                * 6,
                feature_level=[1, 2],
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
                state_loss_weights=state_loss_weights,
                fk_loss_weight=state_loss_weights,
                state_dims=state_dims,
            ),
        )
    )
    return model


def build_transforms(config, calibration=None):
    from robo_orchard_lab.dataset.robotwin.transforms import (
        AddItems,
        AddScaleShift,
        CalibrationToExtrinsic,
        ConvertDataType,
        DualArmKinematics,
        GetProjectionMat,
        IdentityTransform,
        ImageChannelFlip,
        ItemSelection,
        JointStateNoise,
        Resize,
        SimpleStateSampling,
        ToTensor,
    )

    add_data_relative_items = dict(
        type=AddItems,
        T_base2world=config.get(
            "T_base2world",
            [
                [0, -1, 0, 0],
                [1, 0, 0, -0.65],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
    )
    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
    )
    resize = dict(
        type=Resize,
        dst_wh=(320, 256),
        dst_intrinsic=[
            [358.6422, 0.0000, 160.0000, 0.0000],
            [0.0000, 382.5517, 128.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
    )
    if config.get("img_channel_flip"):
        img_channel_flip = dict(
            type=ImageChannelFlip, output_channel=[2, 1, 0]
        )
    else:
        img_channel_flip = dict(type=IdentityTransform)
    to_tensor = dict(type=ToTensor)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="base")
    convert_dtype = dict(
        type=ConvertDataType,
        convert_map=dict(
            imgs="float32",
            depths="float32",
            image_wh="float32",
            projection_mat="float32",
            embodiedment_mat="float32",
        ),
    )
    scale_shift_list = config.get(
        "scale_shift",
        [
            [0.82115489, 0.00280333],
            [1.26673863, 1.26673677],
            [1.38083194, 1.38080194],
            [0.94487381, -0.94485653],
            [0.13566405, 0.05423572],
            [0.90747011, 0.05119371],
            [0.425, 0.575],
            [0.82115489, 0.00280333],
            [1.26673863, 1.26673677],
            [1.38083194, 1.38080194],
            [0.94487381, -0.94485653],
            [0.13566405, 0.05423572],
            [0.90747011, 0.05119371],
            [0.425, 0.575],
        ],  # defalut scale shift for robotwin2.0 example single task
    )

    kinematics = dict(
        type=DualArmKinematics,
        urdf=config["urdf"],
        **config.get("kinematics_config", {}),
    )
    scale_shift = dict(
        type=AddScaleShift,
        scale_shift=scale_shift_list,
    )
    joint_state_noise = dict(
        type=JointStateNoise,
        noise_range=[
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.0, 0.0],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.0, 0.0],
        ],
    )
    item_selection = dict(
        type=ItemSelection,
        keys=[
            "imgs",
            "depths",
            "image_wh",
            "projection_mat",
            "embodiedment_mat",
            "hist_robot_state",
            "pred_robot_state",
            "joint_relative_pos",
            "joint_scale_shift",
            "kinematics",
            "text",
            "uuid",
        ],
    )

    if calibration is not None:
        calib_to_ext = dict(
            type=CalibrationToExtrinsic,
            calibration=calibration,
            **config.get("kinematics_config", {}),
        )
    else:
        calib_to_ext = dict(type=IdentityTransform)

    train_transforms = [
        add_data_relative_items,
        state_sampling,
        resize,
        img_channel_flip,
        to_tensor,
        calib_to_ext,
        projection_mat,
        scale_shift,
        joint_state_noise,
        convert_dtype,
        kinematics,
        item_selection,
    ]
    val_transforms = [
        add_data_relative_items,
        state_sampling,
        resize,
        img_channel_flip,
        to_tensor,
        calib_to_ext,
        projection_mat,
        scale_shift,
        convert_dtype,
        kinematics,
        item_selection,
    ]
    return train_transforms, val_transforms


def build_dataset(config, lazy_init=False):
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    train_transforms, val_transforms = build_transforms(config)
    train_dataset = RoboTwinLmdbDataset(
        paths=config["data_path"],
        task_names=config["task_names"],
        lazy_init=lazy_init,
        transforms=train_transforms,
    )

    val_dataset = RoboTwinLmdbDataset(
        paths=None,  # only for data preprocessing of close loop eval
        task_names=config["task_names"],
        lazy_init=True,
        transforms=val_transforms,
        instruction_keys=("seen",),
    )
    return train_dataset, val_dataset


def build_optimizer(config, model):
    from torch import optim

    base_lr = config["lr"]
    max_step = config["max_step"]

    backbone_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "backbone." in name or "text_encoder." in name:
            backbone_params.append(p)
        else:
            other_params.append(p)
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * 0.1},
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


def build_processor(config):
    from robo_orchard_lab.dataset.robotwin.transforms import UnsqueezeBatch
    from robo_orchard_lab.models.sem_modules import (
        SEMProcessor,
        SEMProcessorCfg,
    )

    transforms = build_transforms(config, config.get("calibration"))[1]
    transforms.append(dict(type=UnsqueezeBatch))
    processor = SEMProcessor(
        SEMProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            transforms=transforms,
        )
    )
    return processor
