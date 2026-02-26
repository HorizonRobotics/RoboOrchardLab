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

import importlib
import logging
import os
import sys
from typing import Optional

import requests
import torch
from safetensors.torch import load_model
from terminaltables import AsciiTable

from robo_orchard_lab.utils import as_sequence
from robo_orchard_lab.utils.huggingface import (
    auto_add_repo_type,
    download_hf_resource,
)

logger = logging.getLogger(__file__)


def load_config(config_file):
    assert config_file.endswith(".py")
    config_dir, module_name = os.path.split(config_file)
    sys.path.insert(0, config_dir)
    module_name = module_name[:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class GetFile:
    def __init__(self, url):
        self.url = url

    def __enter__(self):
        if self.url.startswith("http"):
            file_name = "_" + self.url.split("/")[-1]
            with requests.get(self.url, stream=True) as r:
                r.raise_for_status()
                with open(file_name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            self.url = file_name
            return file_name
        elif self.url.startswith("hf://"):
            return download_hf_resource(auto_add_repo_type(self.url))
        elif os.path.exists(self.url):
            return self.url
        else:
            raise ValueError("Invalid checkpoint url: {self.url}.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def load_checkpoint(model, checkpoint=None, accelerator=None, **kwargs):
    if checkpoint is None:
        return

    logger.info(f"load checkpoint: {checkpoint}")
    with GetFile(checkpoint) as checkpoint:
        if checkpoint.endswith(".safetensors"):
            missing_keys, unexpected_keys = load_model(
                model, checkpoint, strict=False, **kwargs
            )
        else:
            state_dict = torch.load(checkpoint, weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False, **kwargs
            )
        if accelerator is None or accelerator.is_main_process:
            logger.info(
                f"num of missing_keys: {len(missing_keys)},"
                f"num of unexpected_keys: {len(unexpected_keys)}"
            )
            logger.info(
                f"missing_keys:\n {missing_keys}\n"
                f"unexpected_keys:\n {unexpected_keys}"
            )


class ActionMetric:
    def __init__(
        self,
        num_modes: int | list[int] = 1,
        eval_horizons: Optional[int | list[int]] = None,
        end_effector_idx: Optional[int | list[int]] = None,
    ):
        self.num_modes = as_sequence(num_modes)
        self.eval_horizons = (
            as_sequence(eval_horizons) if eval_horizons is not None else None
        )
        self.reset()
        self.end_effector_idx = (
            as_sequence(end_effector_idx)
            if end_effector_idx is not None
            else None
        )
        self.reset()

    def reset(self):
        self.results = []

    def compute(self, accelerator):
        results = accelerator.gather_for_metrics(
            self.results, use_gather_object=True
        )
        if accelerator.is_main_process:
            metrics = self.compute_metrics(results)
        else:
            metrics = None
        return metrics

    def update(self, batch, model_outputs):
        for i, output in enumerate(model_outputs):
            self.results.append(
                dict(
                    pred_actions=output["pred_actions"].cpu(),
                    gt_actions=batch["pred_robot_state"][i].cpu(),
                )
            )

    def compute_metrics(self, results):
        if isinstance(self.eval_horizons, (tuple, list)):
            metrics = dict()
            for h in self.eval_horizons:
                metrics.update(self._compute_metrics(results, h))
            horizons = [x for x in self.eval_horizons]
        else:
            metrics = self._compute_metrics(results)
            horizons = [results[0]["pred_actions"].shape[1]]

        horizons = [str(x) for x in horizons]
        table_rows = [["metric", "joint_idx", "mode"] + horizons]
        mean_table_rows = [["metric", "joint_idx", "mode"] + horizons]
        ee_table_rows = [["ee_metric", "joint_idx", "mode"] + horizons]
        num_joint = results[0]["pred_actions"].shape[2]
        joints = ["mean"] + [str(x) for x in range(num_joint)]

        if self.end_effector_idx is None:
            end_effector_idx = [f"{num_joint - 1}"]
        else:
            end_effector_idx = [f"{x}" for x in self.end_effector_idx]

        for metric in [
            "average_joint",
            "final_joint",
            "average_xyz",
            "final_xyz",
            "average_quat",
            "final_quat",
            "jerk",
            "jerk_xyz",
        ]:
            for mode in self.num_modes:
                for joint in joints:
                    values = []
                    for horizon in horizons:
                        values.append(
                            "{:.6f}".format(
                                metrics[f"{metric}@{joint}@{mode}@{horizon}"]
                            )
                        )
                    row = [metric, joint, mode] + values
                    table_rows.append(row)
                    if joint == "mean":
                        mean_table_rows.append(row)
                    if joint in end_effector_idx:
                        ee_table_rows.append(row)
            table_rows.append([])

        for rows in [table_rows, ee_table_rows, mean_table_rows]:
            table = AsciiTable(rows)
            logger.info("\n" + table.table)
        return metrics

    def _compute_metrics(self, results, horizon=None):
        average_joint_errors = []
        final_joint_errors = []
        average_xyz_errors = []
        final_xyz_errors = []
        average_quat_errors = []
        final_quat_errors = []
        jerks = []
        jerks_xyz = []

        A, XYZ, ROT = (0,), (1, 2, 3), (4, 5, 6, 7)  # noqa: N806

        for ret in results:
            pred = ret["pred_actions"]
            gt = ret["gt_actions"]
            if horizon is not None:
                pred = ret["pred_actions"][:, :horizon]
                gt = ret["gt_actions"][:horizon]
            error = torch.abs(pred - gt)
            average_error = error.mean(dim=1)
            final_error = error[:, -1]
            average_joint_errors.append(average_error[..., A])
            final_joint_errors.append(final_error[..., A])
            average_xyz_errors.append(
                torch.norm(average_error[..., XYZ], dim=-1)
            )
            final_xyz_errors.append(torch.norm(final_error[..., XYZ], dim=-1))
            average_quat_errors.append(
                torch.norm(average_error[..., ROT], dim=-1)
            )
            final_quat_errors.append(torch.norm(final_error[..., ROT], dim=-1))
            jerk = pred[..., A].diff(n=3, dim=1).abs().mean(dim=1)
            jerks.append(jerk)
            jerk_xyz = pred[..., XYZ].diff(n=3, dim=1).norm(dim=-1).mean(dim=1)
            jerks_xyz.append(jerk_xyz)

        values = dict(
            average_joint=torch.stack(average_joint_errors),
            final_joint=torch.stack(final_joint_errors),
            average_xyz=torch.stack(average_xyz_errors),
            final_xyz=torch.stack(final_xyz_errors),
            average_quat=torch.stack(average_quat_errors),
            final_quat=torch.stack(final_quat_errors),
            jerk=torch.stack(jerks),
            jerk_xyz=torch.stack(jerks_xyz),
        )

        num_joint = average_joint_errors[0].shape[1]
        metrics = dict()
        if horizon is None:
            horizon = ret["pred_actions"].shape[1]

        for num_mode in self.num_modes:
            for k, v in values.items():
                assert num_mode <= v.shape[1]
                metrics[f"{k}@mean@{num_mode}@{horizon}"] = (
                    v[:, :num_mode].mean(dim=2).min(dim=1)[0].mean()
                )
                for i in range(num_joint):
                    metrics[f"{k}@{i}@{num_mode}@{horizon}"] = (
                        v[:, :num_mode, i].min(dim=1)[0].mean()
                    )
        return metrics
