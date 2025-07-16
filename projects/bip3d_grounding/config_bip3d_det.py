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
    mode="det",  # "det" or "grounding" or "det_grounding"
    data_version="v1",
    trainval=False,
    embed_dims=256,
    with_depth=True,
    with_depth_loss=True,
    min_depth=0.25,
    max_depth=10,
    num_depth=64,
    batch_size=1,
    val_batch_size=1,
    step_log_freq=25,
    num_workers=8,
    lr=2e-4,
    eval_only=False,
    checkpoint="./ckpt/groundingdino_swint_ogc_mmdet-822d7e9d-rename.pth",
    bert_checkpoint="./ckpt/bert-base-uncased",
    anchor_file="./anchor_files/embodiedscan_kmeans_det_cam_log_z-0.2-3.npy",
    data_root="./data/",
)

if config["mode"] == "det":
    config.update(
        dict(
            max_epoch=240,
            epoch_eval_freq=10,
            save_epoch_freq=10,
        )
    )
elif config["mode"] == "grounding":
    config.update(
        dict(
            max_epoch=2,
            epoch_eval_freq=1,
            save_epoch_freq=1,
        )
    )
elif config["mode"] == "det_grounding":
    config.update(
        dict(
            max_step=int(5e4),
            step_eval_freq=25000,
            save_step_freq=25000,
        )
    )


def build_model(config):
    from torch import nn

    from robo_orchard_lab.models.bip3d.bert import BertModel
    from robo_orchard_lab.models.bip3d.feature_enhancer import (
        TextImageDeformable2DEnhancer,
    )
    from robo_orchard_lab.models.bip3d.grounding_decoder import (
        BBox3DDecoder,
        DoF9BoxEncoder,
        DoF9BoxLoss,
        FocalLoss,
        Grounding3DTarget,
        GroundingBox3DPostProcess,
        GroundingRefineClsHead,
        InstanceBank,
        SparseBox3DKeyPointsGenerator,
    )
    from robo_orchard_lab.models.bip3d.grounding_decoder.deformable_aggregation import (  # noqa: E501
        DeformableFeatureAggregation,
    )
    from robo_orchard_lab.models.bip3d.spatial_enhancer import (
        BatchDepthProbGTGenerator,
        DepthFusionSpatialEnhancer,
    )
    from robo_orchard_lab.models.bip3d.structure import (
        BIP3D,
        BIP3DConfig,
    )
    from robo_orchard_lab.models.layers.data_preprocessors import (
        BaseDataPreprocessor,
        GridMask,
    )
    from robo_orchard_lab.models.layers.transformer_layers import (
        FFN,
        MultiheadAttention,
    )
    from robo_orchard_lab.models.modules.channel_mapper import ChannelMapper
    from robo_orchard_lab.models.modules.resnet import ResNet
    from robo_orchard_lab.models.modules.swin_transformer import (
        SwinTransformer,
    )

    embed_dims = config["embed_dims"]
    model = BIP3D(
        cfg=BIP3DConfig(
            data_preprocessor=dict(
                type=BaseDataPreprocessor,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                channel_flip=True,
                batch_transforms=[
                    dict(
                        type=BatchDepthProbGTGenerator,
                        min_depth=config.get("min_depth", 0.25),
                        max_depth=config.get("max_depth", 10),
                        num_depth=config.get("num_depth", 64),
                        origin_stride=4,
                        valid_threshold=0.0,
                        stride=(8, 16, 32, 64),
                    ),
                    dict(
                        type=GridMask,
                        apply_grid_mask_keys=["imgs", "depths"],
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
                num_outs=4,
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
                    num_outs=4,
                )
                if config.get("with_depth")
                else None
            ),
            text_encoder=dict(
                type=BertModel,
                special_tokens_list=["[CLS]", "[SEP]"],
                name=config["bert_checkpoint"],
                pad_to_max=False,
                use_sub_sentence_represent=True,
                add_pooling_layer=False,
                max_tokens=768,
                use_checkpoint=True,
                return_tokenized=True,
            ),
            feature_enhancer=dict(
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
                        num_levels=4,
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
                type=BBox3DDecoder,
                look_forward_twice=True,
                instance_bank=dict(
                    type=InstanceBank,
                    num_anchor=50,
                    anchor=config["anchor_file"],
                    embed_dims=256,
                    anchor_in_camera=True,
                ),
                anchor_encoder=dict(
                    type=DoF9BoxEncoder,
                    embed_dims=256,
                    rot_dims=3,
                ),
                graph_model=dict(
                    type=MultiheadAttention,
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True,
                ),
                ffn=dict(
                    type=FFN,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                ),
                norm_layer=dict(type=nn.LayerNorm, normalized_shape=256),
                deformable_model=dict(
                    type=DeformableFeatureAggregation,
                    embed_dims=256,
                    num_groups=8,
                    num_levels=4,
                    use_camera_embed=True,
                    with_depth=True,
                    min_depth=config.get("min_depth", 0.25),
                    max_depth=config.get("max_depth", 10),
                    kps_generator=dict(
                        type=SparseBox3DKeyPointsGenerator,
                        fix_scale=[
                            [0, 0, 0],
                            [0.45, 0, 0],
                            [-0.45, 0, 0],
                            [0, 0.45, 0],
                            [0, -0.45, 0],
                            [0, 0, 0.45],
                            [0, 0, -0.45],
                        ],
                        num_learnable_pts=9,
                    ),
                    with_value_proj=True,
                    filter_outlier=True,
                ),
                text_cross_attn=dict(
                    type=MultiheadAttention,
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True,
                ),
                refine_layer=dict(
                    type=GroundingRefineClsHead,
                    embed_dims=256,
                    output_dim=9,
                    cls_bias=True,
                ),
                loss_cls=dict(
                    type=FocalLoss,
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0,
                ),
                loss_reg=dict(
                    type=DoF9BoxLoss,
                    loss_weight_wd=1.0,
                    loss_weight_cd=0.8,
                ),
                sampler=dict(
                    type=Grounding3DTarget,
                    cls_weight=1.0,
                    box_weight=1.0,
                    num_dn=100,
                    cost_weight_wd=1.0,
                    cost_weight_cd=0.8,
                    with_dn_query=True,
                    num_classes=284,
                    embed_dims=256,
                ),
                gt_reg_key="gt_bboxes_3d",
                gt_cls_key="tokens_positive",
                post_processor=dict(
                    type=GroundingBox3DPostProcess,
                    num_output=1000,
                ),
            ),
        )
    )
    return model


def build_transform(config, detection=True):
    import copy

    from robo_orchard_lab.dataset.embodiedscan.transforms import (
        CategoryGroundingDataPrepare,
        Format,
        LoadMultiViewImageDepthFromFile,
    )

    sep_token = "[SEP]"
    z_range = [-0.2, 3]
    load_img_depth = dict(
        type=LoadMultiViewImageDepthFromFile,
        num_views=1,
        max_num_views=18,
        sample_mode="random" if detection else "fix",
        rotate_3rscan=True,
        load_img=True,
        load_depth=True,
        dst_intrinsic=[
            [432.57943431339237, 0.0, 256],
            [0.0, 539.8570854208559, 256],
            [0.0, 0.0, 1.0],
        ],
        dst_wh=(512, 512),
        random_crop_range=((0, 0), (-0.05, 0.05)),
    )

    val_load_img_depth = copy.deepcopy(load_img_depth)
    val_load_img_depth.update(
        max_num_views=50,
        sample_mode="fix",
        random_crop_range=None,
    )

    category_grounding_data_prepare = dict(
        type=CategoryGroundingDataPrepare,
        filter_others=True,
        filter_invisible=True,
        sep_token=sep_token,
        max_class=128,
        training=True,
        z_range=z_range,
    )
    val_category_grounding_data_prepare = dict(
        type=CategoryGroundingDataPrepare,
        sep_token=sep_token,
        training=False,
    )

    # depth_prob_label_gen = dict(
    #     type=DepthProbLabelGenerator,
    #     origin_stride=4,
    #     min_depth=config["min_depth"],
    #     max_depth=config["max_depth"],
    #     num_depth=config["num_depth"],
    # )
    format_data = dict(type=Format)

    if detection:
        train_transforms = [
            load_img_depth,
            category_grounding_data_prepare,
            # depth_prob_label_gen,
            format_data,
        ]
        val_transforms = [
            val_load_img_depth,
            val_category_grounding_data_prepare,
            format_data,
        ]
    else:
        train_transforms = [
            load_img_depth,
            # depth_prob_label_gen,
            format_data,
        ]
        val_transforms = [
            val_load_img_depth,
            format_data,
        ]
    return train_transforms, val_transforms


def build_dataset(config, lazy_init=False):
    import torch

    from robo_orchard_lab.dataset.embodiedscan.embodiedscan_det_grounding_dataset import (  # noqa: E501
        EmbodiedScanDetGroundingDataset,
    )

    data_root = config["data_root"]
    if config["data_version"] == "v1":
        train_ann_file = "embodiedscan/embodiedscan_infos_train.pkl"
        val_ann_file = "embodiedscan/embodiedscan_infos_val.pkl"
        train_vg_file = "embodiedscan/embodiedscan_train_vg_all.json"
        val_vg_file = "embodiedscan/embodiedscan_val_vg_all.json"
    elif config["data_version"] == "v1-mini":
        train_ann_file = "embodiedscan/embodiedscan_infos_train.pkl"
        val_ann_file = "embodiedscan/embodiedscan_infos_val.pkl"
        train_vg_file = "embodiedscan/embodiedscan_train_mini_vg.json"
        val_vg_file = "embodiedscan/embodiedscan_val_mini_vg.json"
    elif config["data_version"] == "v2":
        train_ann_file = "embodiedscan-v2/embodiedscan_infos_train.pkl"
        val_ann_file = "embodiedscan-v2/embodiedscan_infos_val.pkl"
        train_vg_file = "embodiedscan-v2/embodiedscan_train_vg.json"
        val_vg_file = "embodiedscan-v2/embodiedscan_val_vg.json"
    else:
        raise ValueError(
            "data_version should in ['v1', 'v1-mini', 'v2'], "
            f"but got {config['data_version']}"
        )

    det_flag = "det" in config["mode"]
    if det_flag:
        train_transforms, val_transforms = build_transform(config, True)
    else:
        train_transforms, val_transforms = build_transform(config, False)

    train_dataset = EmbodiedScanDetGroundingDataset(
        data_root=data_root,
        ann_file=train_ann_file,
        transforms=train_transforms,
        test_mode=False,
        lazy_init=lazy_init or config.get("eval_only"),
        mode="detection" if det_flag else "grounding",
        vg_file=train_vg_file,
    )
    if config["trainval"]:
        train_dataset = torch.utils.data.ConcatDataset(
            [
                train_dataset,
                EmbodiedScanDetGroundingDataset(
                    data_root=data_root,
                    ann_file=val_ann_file,
                    transforms=train_transforms,
                    test_mode=False,
                    lazy_init=lazy_init or config.get("eval_only"),
                    mode="detection" if det_flag else "grounding",
                    vg_file=val_vg_file,
                ),
            ]
        )

    val_dataset = EmbodiedScanDetGroundingDataset(
        data_root=data_root,
        ann_file=val_ann_file,
        transforms=val_transforms,
        test_mode=True,
        lazy_init=lazy_init,
        mode="detection" if det_flag else "grounding",
        vg_file=val_vg_file,
    )

    if ("det" in config["mode"]) and ("grounding" in config["mode"]):
        g_train_transforms, _ = build_transform(config, False)
        g_train_dataset = EmbodiedScanDetGroundingDataset(
            data_root=data_root,
            ann_file=train_ann_file,
            transforms=g_train_transforms,
            test_mode=False,
            lazy_init=lazy_init or config.get("eval_only"),
            mode="grounding",
            vg_file=train_vg_file,
        )
        if config["trainval"]:
            train_dataset = torch.utils.data.ConcatDataset(
                [
                    g_train_dataset,
                    EmbodiedScanDetGroundingDataset(
                        data_root=data_root,
                        ann_file=train_ann_file,
                        transforms=g_train_transforms,
                        test_mode=False,
                        lazy_init=lazy_init or config.get("eval_only"),
                        mode="grounding",
                        vg_file=val_vg_file,
                    ),
                ]
            )

        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, g_train_dataset],
        )
    return train_dataset, val_dataset


def build_optimizer(config, model):
    from torch import optim

    base_lr = config["lr"]
    backbone_params = []
    text_encoder_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "backbone." in name:
            backbone_params.append(p)
        elif "text_encoder." in name:
            text_encoder_params.append(p)
        else:
            other_params.append(p)
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * 0.1},
            {"params": text_encoder_params, "lr": base_lr * 0.05},
            {"params": other_params},
        ],
        lr=base_lr,
        weight_decay=config.get("weight_decay", 0.0005),
    )

    if "max_step" in config:
        total_step = config["max_step"]
    elif config["mode"] == "det":
        step_per_epoch = 3113 // 8
        total_step = step_per_epoch * config["max_epoch"]
    elif config["mode"] == "grounding":
        if config["data_version"] == "v1-mini":
            step_per_epoch = 3200
        else:
            step_per_epoch = 32000
        total_step = step_per_epoch * config["max_epoch"]

    lr_scheduler = optim.lr_scheduler.ChainedScheduler(
        [
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, total_iters=500
            ),
            optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(total_step * 8 / 12),
                    int(total_step * 11 / 12),
                ],
                gamma=0.1,
            ),
        ]
    )
    return optimizer, lr_scheduler
