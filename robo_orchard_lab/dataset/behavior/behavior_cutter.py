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
import json
import os
import re

import cv2
import numpy as np
from wordfreq import zipf_frequency

from robo_orchard_lab.dataset.behavior import utils
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)
from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb


def build_skill(desc, objs):
    def is_meaningful_word(word, threshold=2.5):
        return zipf_frequency(word.lower(), "en") > threshold

    def clean_name(x):
        if isinstance(x, (list, tuple)):
            cleaned = [clean_name(i) for i in x]
            if all(isinstance(i, str) for i in cleaned):
                seen, uniq = set(), []
                for s in cleaned:
                    if s not in seen:
                        seen.add(s)
                        uniq.append(s)
                return uniq
            return cleaned

        if not isinstance(x, str):
            return str(x)

        name = x
        name = re.sub(r"__+", "_", name)
        name = re.sub(r"(?:_\d+)+$", "", name)

        m = re.search(r"_([A-Za-z]{6}\d*)$", name)
        if m:
            token = m.group(1)
            if (
                token.isalpha()
                and len(token) == 6
                and not is_meaningful_word(token)
            ):
                name = name[: m.start()]  # 直接截掉该后缀

        name = name.strip("_").replace("_", " ").strip()
        return name

    objs_clean = clean_name(objs)

    # 1. attach
    if desc.startswith("attach"):
        if isinstance(objs_clean, (list, tuple)) and len(objs_clean) >= 2:
            return f"attach {objs_clean[0]} to {objs_clean[1]}"

    # 2. chop
    if desc.startswith("chop"):
        if isinstance(objs_clean, (list, tuple)) and len(objs_clean) >= 1:
            tool = (
                objs_clean[0][0]
                if isinstance(objs_clean[0], list)
                else objs_clean[0]
            )
            targets = []
            for group in objs_clean[1:]:
                if isinstance(group, (list, tuple)):
                    for o in group:
                        clean_o = re.sub(
                            r"^ha(?:lf|fl)\s+", "", o, flags=re.IGNORECASE
                        ).strip()
                        if clean_o and "drop" not in clean_o:
                            targets.append(clean_o)
                else:
                    targets.append(group)
            target_phrase = " and ".join(sorted(set(targets)))
            return f"chop {target_phrase} with {tool}"

    # 3–5. close door / drawer / lid
    if desc.startswith("close door"):
        if (
            isinstance(objs_clean, list)
            and len(objs_clean) == 1
            and objs_clean[0] == "door"
        ):
            return "close door"
        elif objs_clean:
            return f"close {objs_clean[0]} door"

    if desc.startswith("close drawer") and objs_clean:
        return f"close {objs_clean[0]} drawer"

    if desc.startswith("close lid") and objs_clean:
        return f"close {objs_clean[0]} lid"

    # 6. hand over
    if desc.startswith("hand over"):
        if isinstance(objs_clean, (list, tuple)) and len(objs_clean) >= 3:
            return (
                f"hand over {objs_clean[0]} "
                f"from {objs_clean[1]} "
                f"to {objs_clean[2]}"
            )

    # 7. hang
    if (
        desc.startswith("hang")
        and isinstance(objs_clean, (list, tuple))
        and len(objs_clean) >= 2
    ):
        return f"hang {objs_clean[0]} on {objs_clean[1]}"

    # 8. ignite
    if (
        desc.startswith("ignite")
        and isinstance(objs_clean, (list, tuple))
        and len(objs_clean) >= 2
    ):
        src, tgt = objs_clean[:2]
        if src == "lighter":
            return f"ignite {tgt} with {src}"
        return f"ignite {src} with {tgt}"

    # 9. insert
    if (
        desc.startswith("insert")
        and isinstance(objs_clean, (list, tuple))
        and len(objs_clean) >= 2
    ):
        return f"insert {objs_clean[0]} into {objs_clean[1]}"

    # 10. move to
    if desc.startswith("move to"):
        if isinstance(objs_clean, (list, tuple)):
            if len(objs_clean) == 1:
                return f"move to {objs_clean[0]}"
            elif len(objs_clean) >= 2:
                return f"move {objs_clean[0]} to {objs_clean[1]}"

    # 11–13. open door / drawer / lid
    if desc.startswith("open door"):
        if (
            isinstance(objs_clean, list)
            and len(objs_clean) == 1
            and objs_clean[0] == "door"
        ):
            return "open door"
        elif objs_clean:
            return f"open {objs_clean[0]} door".replace(" no top", "")

    if desc.startswith("open drawer") and objs_clean:
        return f"open {objs_clean[0]} drawer"

    if desc.startswith("open lid") and objs_clean:
        return f"open {objs_clean[0]} lid"

    # 14. pick up from
    if "pick up" in desc and "from" in desc:
        if isinstance(objs_clean, (list, tuple)) and len(objs_clean) == 2:
            picked, src = objs_clean
            picked_items = picked if isinstance(picked, list) else [picked]
            src_name = src[-1] if isinstance(src, list) else src
            return (
                f"pick up {' and '.join(picked_items)} from {src_name}"
                .replace(" no top", "")
            )

    # 15–19. place 系列
    if desc.startswith("place"):
        # Case 1: 标准结构 [['obj1', 'obj2'], 'target', ['extra1', 'extra2']]
        if isinstance(objs_clean, (list, tuple)) and len(objs_clean) >= 2:
            placed = objs_clean[0]
            placed_items = placed if isinstance(placed, list) else [placed]

            target_main = objs_clean[1]
            target_main_name = (
                target_main[-1]
                if isinstance(target_main, list)
                else target_main
            )

            # 支持 next to 的附加目标（可能多个）
            extra_targets = []
            if len(objs_clean) >= 3:
                extra = objs_clean[2]
                extra_targets = (
                    extra if isinstance(extra, (list, tuple)) else [extra]
                )

            placed_str = " and ".join(placed_items)
            target_str = target_main_name

            # ---- 多种介词情况 ----
            if "in next to" in desc and extra_targets:
                return (
                    f"place {placed_str} in {target_str} next to "
                    f"{' and '.join(extra_targets)}"
                )
            if "on next to" in desc and extra_targets:
                return (
                    f"place {placed_str} on {target_str} next to "
                    f"{' and '.join(extra_targets)}"
                )
            if "in" in desc:
                return f"place {placed_str} in {target_str}"
            if "on" in desc:
                return f"place {placed_str} on {target_str}"
            if "under" in desc:
                return f"place {placed_str} under {target_str}"

        # Case 2: 非标准结构 ['storage_box_80', 'storage_box_79']
        if isinstance(objs_clean, (list, tuple)) and all(
            isinstance(o, str) for o in objs_clean
        ):
            return f"place {' and '.join(objs_clean)}"

        # Case 3: 单字符串
        if isinstance(objs_clean, str):
            return f"place {objs_clean}"

    # 20. pour
    if desc.startswith("pour"):

        def uniq(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        def flatten_once(x):
            out = []
            for i in x:
                if isinstance(i, (list, tuple)):
                    out.extend(i)
                else:
                    out.append(i)
            return out

        def normalize_list(x):
            if isinstance(x, str):
                return [x]
            if isinstance(x, (list, tuple)):
                return list(x)
            return [str(x)]

        # --- Case 1: 标准结构 [objects, src, dst] ---
        if isinstance(objs_clean, (list, tuple)) and len(objs_clean) >= 3:
            src_objs, src_container, dst_container = objs_clean

            # 多层物体组，如 [[['pepperoni'], ['mushroom']], ...]
            if isinstance(src_objs, (list, tuple)) and all(
                isinstance(g, (list, tuple)) for g in src_objs
            ):
                src_groups = []
                for g in src_objs:
                    src_groups.append(" and ".join(uniq(normalize_list(g))))
                src_str = " and ".join(src_groups)
            else:
                src_items = uniq(flatten_once(normalize_list(src_objs)))
                src_str = " and ".join(src_items)

            src_container_str = " and ".join(
                uniq(normalize_list(src_container))
            )
            dst_container_str = " and ".join(
                uniq(normalize_list(dst_container))
            )

            return (
                f"pour {src_str} from {src_container_str} "
                f"onto {dst_container_str}"
            )

        # --- Case 2: 不完整结构 [['obj1','obj2'], 'target'] ---
        if isinstance(objs_clean, (list, tuple)) and len(objs_clean) == 2:
            src_objs, dst_container = objs_clean
            src_str = " and ".join(
                uniq(flatten_once(normalize_list(src_objs)))
            )
            dst_str = " and ".join(uniq(normalize_list(dst_container)))
            return f"pour {src_str} onto {dst_str}"

        # --- Case 3: 只有物体列表 ---
        if isinstance(objs_clean, (list, tuple)) and all(
            isinstance(o, str) for o in objs_clean
        ):
            return f"pour {' and '.join(uniq(objs_clean))}"

        # --- Case 4: 单字符串 ---
        if isinstance(objs_clean, str):
            return f"pour {objs_clean}"

    # 21. press
    if desc.startswith("press") and objs_clean:
        return f"press {objs_clean[0]}"

    # 22. pull tray
    if desc.startswith("pull tray") and objs_clean:
        return f"pull {objs_clean[0]} tray"

    # 23. push to
    if desc.startswith("push to") and len(objs_clean) >= 2:

        def norm(x):
            if isinstance(x, (list, tuple)):
                flat = []
                for i in x:
                    if isinstance(i, (list, tuple)):
                        flat.extend(norm(i))
                    else:
                        flat.append(str(i))
                # 去重 + 保序
                seen = set()
                uniq = []
                for t in flat:
                    if t not in seen:
                        uniq.append(t)
                        seen.add(t)
                return uniq
            return [str(x)]

        src = " and ".join(norm(objs_clean[0]))
        dst = " and ".join(norm(objs_clean[1]))
        return f"push {src} to {dst}"

    # 24. spray
    if desc.startswith("spray"):
        if len(objs_clean) == 1:
            return f"spray {objs_clean[0]}"
        else:
            return f"spray {objs_clean[0]} on {objs_clean[1]}"

    # 25. sweep surface
    if desc.startswith("sweep surface") and len(objs_clean) >= 2:
        obj_str = objs_clean[0]
        if isinstance(obj_str, (list, tuple)):
            obj_str = " and ".join(sorted(set(obj_str)))
        return f"sweep {objs_clean[1]} with {obj_str}"

    # 26. tip over
    if desc.startswith("tip over") and objs_clean:
        return f"tip over {objs_clean[0]}"

    # 27–28. turn on/off switch
    if desc.startswith("turn on switch") and objs_clean:
        return f"turn on {objs_clean[0]}"

    if desc.startswith("turn off") and objs_clean:
        return f"turn off {objs_clean[0]}"

    # 29. turn to
    if desc.startswith("turn to") and len(objs_clean) >= 2:
        return f"turn {objs_clean[0]} to {objs_clean[1]}"

    # 30. wipe hard
    if desc.startswith("wipe hard") and len(objs_clean) >= 2:
        return f"wipe {objs_clean[0]} hard on {objs_clean[1]}"

    # fallback
    return desc.strip()


class BehaviorCut(BaseLmdbManipulationDataPacker):
    def __init__(
        self,
        input_path,
        output_path,
        data_root="/work/jfs/behavior-1k/2025-challenge-demos",
        commit_step=500,
        frame_sample_rate=1,
        move_sample_rate=1,
        pad=10,
        keep_move=False,
        write_video=False,
        **kwargs,
    ):
        super().__init__(
            input_path, output_path, commit_step=commit_step, **kwargs
        )

        self.output_path = output_path
        self.data_root = data_root
        # 数据可能被采样过，这是采样用的downsample rate
        self.frame_sample_rate = frame_sample_rate
        # 机器人move部分，downsample rate
        self.move_sample_rate = move_sample_rate
        self.keep_move = keep_move
        self.write_video = write_video
        self.pad = pad

        self.index_lmdb = Lmdb(f"{input_path}/index/", writable=False)
        self.meta_lmdb = Lmdb(f"{input_path}/meta/", writable=False)
        self.img_lmdb = Lmdb(f"{input_path}/image/", writable=False)
        self.depth_lmdb = Lmdb(f"{input_path}/depth/", writable=False)

    def load_anno(self, path: str) -> utils.Episode:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return utils.Episode(
            task_id=data.get("task_id", ""),
            episode_id=data.get("episode_id", ""),
            task_name=data.get("task_name", ""),
            skill_annotation=[
                utils.to_skill(x) for x in data.get("skill_annotation", [])
            ],
            primitive_annotation=[
                utils.to_primitive(x)
                for x in data.get("primitive_annotation", [])
            ],
        )

    def get(self, key):
        idx = self.index_lmdb[key]
        uuid = idx["uuid"]
        num_step = idx["num_steps"]

        ori_state = self.meta_lmdb[f"{uuid}/observation/robot_state/ori_state"]

        # robot pose
        robot_pos = ori_state[
            :, utils.PROPRIOCEPTION_INDICES["R1Pro"]["robot_pos"]
        ]
        robot_ori_sin = ori_state[
            :, utils.PROPRIOCEPTION_INDICES["R1Pro"]["robot_ori_sin"]
        ]
        robot_ori_cos = ori_state[
            :, utils.PROPRIOCEPTION_INDICES["R1Pro"]["robot_ori_cos"]
        ]

        action = self.meta_lmdb[f"{uuid}/action"]
        camera_names = self.meta_lmdb[f"{uuid}/camera_names"]
        extrinsic = self.meta_lmdb[f"{uuid}/extrinsic"]
        intrinsic = self.meta_lmdb[f"{uuid}/intrinsic"]
        instruction = self.meta_lmdb[f"{uuid}/instruction"]

        eef = self.meta_lmdb[
            f"{uuid}/observation/robot_state/cartesian_position"
        ]
        joint = self.meta_lmdb[
            f"{uuid}/observation/robot_state/joint_positions"
        ]
        base_qvel = self.meta_lmdb[f"{uuid}/observation/robot_state/base_qvel"]

        (
            rgb_lefts,
            rgb_rights,
            rgb_heads,
            depth_lefts,
            depth_rights,
            depth_heads,
        ) = [], [], [], [], [], []

        for i in range(num_step):
            rgb_lefts.append(self.img_lmdb[f"{uuid}/rgb_left_wrist/{i}"])
            rgb_rights.append(self.img_lmdb[f"{uuid}/rgb_right_wrist/{i}"])
            rgb_heads.append(self.img_lmdb[f"{uuid}/rgb_head/{i}"])

            depth_lefts.append(self.depth_lmdb[f"{uuid}/depth_left_wrist/{i}"])
            depth_rights.append(
                self.depth_lmdb[f"{uuid}/depth_right_wrist/{i}"]
            )
            depth_heads.append(self.depth_lmdb[f"{uuid}/depth_head/{i}"])

        return idx, {
            "uuid": uuid,
            "action": action,
            "camera_names": camera_names,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "instruction": instruction,
            "eef": eef,
            "joint": joint,
            "base_qvel": base_qvel,
            "rgb": {
                "left": rgb_lefts,
                "right": rgb_rights,
                "head": rgb_heads,
            },
            "depth": {
                "left": depth_lefts,
                "right": depth_rights,
                "head": depth_heads,
            },
            "ori_state": ori_state,
            "robot_pos": robot_pos,
            "robot_ori_sin": robot_ori_sin,
            "robot_ori_cos": robot_ori_cos,
        }

    def _write_video(self, output_video_path, frames, subtask="", fps=30):
        root_dir = os.path.dirname(output_video_path)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = 720, 720
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"VideoWriter can't open：{output_video_path}")
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 255)
        for frame in frames:
            frame_data = np.frombuffer(frame, np.uint8)
            frame_decoded = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            # frame_decoded = frame_decoded[..., ::-1]
            # text_size = cv2.getTextSize(
            #    subtask, font, font_scale, font_thickness
            # )[0]
            # text_x = (frame_decoded.shape[1] - text_size[0])
            text_x = 40
            text_y = 40
            cv2.putText(
                frame_decoded,
                subtask,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

            out.write(frame_decoded)

        out.release()

    def pad_intervals(self, intervals, pad_value=0, total_len=None):
        intervals = sorted(intervals, key=lambda x: x[0])
        orig_starts = [s for s, _, _, _ in intervals]

        adjusted = []
        prev_end_capped = None

        for i, (start, end, stype, skill) in enumerate(intervals):
            new_start = max(0, start - pad_value)
            new_end = end + pad_value

            if total_len is not None:
                new_end = min(new_end, total_len - 1)

            if i + 1 < len(intervals):
                next_orig_start = orig_starts[i + 1]
                new_end = min(new_end, next_orig_start)

            if prev_end_capped is not None:
                new_start = max(new_start, prev_end_capped)

            if new_start > new_end:
                new_start = new_end

            adjusted.append([new_start, new_end, stype, skill])
            prev_end_capped = new_end

        return adjusted

    def _pack(self):
        ep_idx = 0
        for key in self.index_lmdb.keys():
            index, sample = self.get(key)
            uuid = sample["uuid"]
            task_id, episode_id = uuid.split("_", maxsplit=1)

            print(uuid)
            anno_file = (
                f"{self.data_root}/annotations/{task_id}/{episode_id}.json"
            )
            if not os.path.exists(anno_file):
                continue

            anno = self.load_anno(anno_file)
            # task_name = "_".join(anno.task_name)

            intervals = []
            for skill in anno.skill_annotation:
                skill_type = skill.skill_type[0]
                skill_desc = skill.skill_description
                object_id = skill.object_id
                subtask_skill = build_skill(skill_desc[0], object_id[0])
                # badcase: https://huggingface.co/datasets/behavior-1k/2025-challenge-demos/blob/main/annotations/task-0020/episode_00200120.json#L78
                for frame_start, frame_end in skill.frame_duration:
                    intervals.append(
                        [frame_start, frame_end, skill_type, subtask_skill]
                    )

            # intervals_old = intervals.copy()
            intervals = self.pad_intervals(intervals, self.pad)
            # for b, a in zip(intervals_old, intervals):
            #    print(b, a)

            for frame_start, frame_end, skill_type, subtask_skill in intervals:
                if skill_type == "navigation" and not self.keep_move:
                    continue

                sample_rate = (
                    self.move_sample_rate if skill_type == "navigation" else 1
                )
                frame_start, frame_end = (
                    frame_start // self.frame_sample_rate,
                    frame_end // self.frame_sample_rate + 1,
                )

                # substask info
                uuid_new = f"{uuid}_{frame_start}_{frame_end}"
                intrinsic = sample["intrinsic"]
                extrinsic = {}
                extrinsic["left_wrist"] = sample["extrinsic"]["left_wrist"][
                    frame_start:frame_end:sample_rate, :, :
                ]
                extrinsic["right_wrist"] = sample["extrinsic"]["right_wrist"][
                    frame_start:frame_end:sample_rate, :, :
                ]
                extrinsic["head"] = sample["extrinsic"]["head"][
                    frame_start:frame_end:sample_rate, :, :
                ]

                instruction = sample["instruction"]
                camera_names = sample["camera_names"]

                ori_state = sample["ori_state"][
                    frame_start:frame_end:sample_rate, :
                ]

                # robot pose
                robot_pos = sample["robot_pos"][
                    frame_start:frame_end:sample_rate, :
                ]
                robot_ori_sin = sample["robot_ori_sin"][
                    frame_start:frame_end:sample_rate, :
                ]
                robot_ori_cos = sample["robot_ori_cos"][
                    frame_start:frame_end:sample_rate, :
                ]

                eef = sample["eef"][frame_start:frame_end:sample_rate, :]
                joint = sample["joint"][frame_start:frame_end:sample_rate, :]
                base_qvel = sample["base_qvel"][
                    frame_start:frame_end:sample_rate, :
                ]
                action = sample["action"][frame_start:frame_end:sample_rate, :]

                rgb_left = sample["rgb"]["left"][
                    frame_start:frame_end:sample_rate
                ]
                rgb_right = sample["rgb"]["right"][
                    frame_start:frame_end:sample_rate
                ]
                rgb_head = sample["rgb"]["head"][
                    frame_start:frame_end:sample_rate
                ]
                depth_left = sample["depth"]["left"][
                    frame_start:frame_end:sample_rate
                ]
                depth_right = sample["depth"]["right"][
                    frame_start:frame_end:sample_rate
                ]
                depth_head = sample["depth"]["head"][
                    frame_start:frame_end:sample_rate
                ]

                assert len(joint) == len(rgb_left), (
                    "joint vs imgs has different length"
                )
                assert len(robot_pos) == len(rgb_left), (
                    "robot pose vs imgs has different length"
                )

                if self.write_video:
                    print(
                        f"{self.output_path}/videos/{episode_id}/{uuid_new}.mp4"
                    )
                    self._write_video(
                        f"{self.output_path}/videos/{episode_id}/{uuid_new}.mp4",
                        rgb_head,
                        subtask_skill,
                    )

                # meta
                self.meta_pack_file.write(
                    f"{uuid_new}/camera_names", camera_names
                )
                self.meta_pack_file.write(f"{uuid_new}/extrinsic", extrinsic)
                self.meta_pack_file.write(f"{uuid_new}/intrinsic", intrinsic)
                self.meta_pack_file.write(
                    f"{uuid_new}/instruction", instruction
                )
                self.meta_pack_file.write(f"{uuid_new}/subtask", subtask_skill)

                # state
                self.meta_pack_file.write(
                    f"{uuid_new}/observation/robot_state/ori_state", ori_state
                )

                self.meta_pack_file.write(
                    f"{uuid_new}/observation/robot_state/robot_pos", robot_pos
                )
                self.meta_pack_file.write(
                    f"{uuid_new}/observation/robot_state/robot_ori_sin",
                    robot_ori_sin,
                )
                self.meta_pack_file.write(
                    f"{uuid_new}/observation/robot_state/robot_ori_cos",
                    robot_ori_cos,
                )

                self.meta_pack_file.write(
                    f"{uuid_new}/observation/robot_state/cartesian_position",
                    eef,
                )
                self.meta_pack_file.write(
                    f"{uuid_new}/observation/robot_state/joint_positions",
                    joint,
                )
                self.meta_pack_file.write(
                    f"{uuid_new}/observation/robot_state/base_qvel", base_qvel
                )

                self.meta_pack_file.write(f"{uuid_new}/action", action)

                for i in range(len(joint)):
                    self.image_pack_file.write(
                        f"{uuid_new}/rgb_left_wrist/{i}", rgb_left[i]
                    )
                    self.image_pack_file.write(
                        f"{uuid_new}/rgb_right_wrist/{i}", rgb_right[i]
                    )
                    self.image_pack_file.write(
                        f"{uuid_new}/rgb_head/{i}", rgb_head[i]
                    )
                    self.depth_pack_file.write(
                        f"{uuid_new}/depth_left_wrist/{i}", depth_left[i]
                    )
                    self.depth_pack_file.write(
                        f"{uuid_new}/depth_right_wrist/{i}", depth_right[i]
                    )
                    self.depth_pack_file.write(
                        f"{uuid_new}/depth_head/{i}", depth_head[i]
                    )

                index_data = index.copy()
                index_data["uuid"] = uuid_new
                index_data["num_steps"] = len(joint)
                self.write_index(ep_idx, index_data)
                ep_idx += 1

        self.close()

    # def close(self):
    #    self.index_lmdb.close()
    #    self.meta_lmdb.close()
    #    self.img_lmdb.close()
    #    self.depth_lmdb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--frame_sample_rate", type=int, default=1)
    parser.add_argument("--move_sample_rate", type=int, default=1)
    parser.add_argument("--pad", type=int, default=60)
    parser.add_argument("--keep_move", action="store_true")
    parser.add_argument("--write_video", action="store_true")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    bc = BehaviorCut(
        args.input_path,
        args.output_path,
        data_root=args.data_root,
        frame_sample_rate=args.frame_sample_rate,
        move_sample_rate=args.move_sample_rate,
        pad=args.pad,
        write_video=args.write_video,
        keep_move=args.keep_move,
    )
    bc()
    bc.close()
