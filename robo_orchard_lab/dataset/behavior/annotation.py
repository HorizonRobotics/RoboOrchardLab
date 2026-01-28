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


import json
import re

from wordfreq import zipf_frequency


def _extract_frame_value(x, *, is_start: bool):
    """Extract a frame index.

    Return:
        - int -> itself
        - list[int] -> first (for start) or last (for end)
    """
    if isinstance(x, int):
        return x

    if isinstance(x, list) and len(x) > 0:
        return x[0] if is_start else x[-1]

    raise ValueError(f"Invalid frame value: {x}")


class Annotation:
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.data = data
        self.skills = {s["skill_idx"]: s for s in data["skill_annotation"]}
        self.primitives = {
            p["primitive_idx"]: p for p in data["primitive_annotation"]
        }

    def get_skill(self, skill_idx: int):
        skill = self.skills.get(skill_idx, None)
        skill_text = self.get_skill_text(
            skill["skill_description"][0], skill["object_id"][0]
        )
        skill["skill_text"] = skill_text

        frame_duration = skill["frame_duration"]
        start_raw, end_raw = frame_duration
        start = _extract_frame_value(start_raw, is_start=True)
        end = _extract_frame_value(end_raw,   is_start=False)
        skill["frame_duration"] = [start, end]

        return skill

    def iter_skill(self):
        for sid in sorted(self.skills.keys()):
            yield self.get_skill(sid)

    def get_primitive(self, primitive_idx: int):
        primitive = self.primitives.get(primitive_idx, None)
        primitive_desc = primitive["primitive_description"]
        object_id = primitive["object_id"]

        primitive_text = []
        for desc, obj in zip(primitive_desc, object_id, strict=True):
            text = self.get_skill_text(desc, obj)
            primitive_text.append(text)

        primitive_text = ", and ".join(primitive_text)
        primitive["primitive_text"] = primitive_text

        frame_duration = primitive["frame_duration"]
        start_raw, end_raw = frame_duration
        start = _extract_frame_value(start_raw, is_start=True)
        end = _extract_frame_value(end_raw,   is_start=False)
        primitive["frame_duration"] = [start, end]

        return primitive

    def iter_primitive(self):
        for pid in sorted(self.primitives.keys()):
            yield self.get_primitive(pid)

    def get_primitive_with_skills(self, primitive_idx: int):
        prim = self.get_primitive(primitive_idx)
        if prim is None:
            return None

        prim_range = prim["frame_duration"]
        skill_list = []

        for sid in prim["skill_idxes"]:
            skill = self.get_skill(sid)
            if skill is None:
                continue

            merged_range = [skill["frame_duration"], prim_range]

            skill_list.append(
                {
                    "skill_idx": sid,
                    "skill_description": skill["skill_description"],
                    "frame_duration": merged_range,
                    "object_id": skill.get("object_id", []),
                    "skill_type": skill.get("skill_type", []),
                }
            )

        return {
            "primitive_idx": prim["primitive_idx"],
            "primitive_description": prim["primitive_description"],
            "frame_duration": prim["frame_duration"],
            "skills": skill_list,
        }

    def get_skill_text(self, desc, objs):
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
                    name = name[: m.start()]

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

        # 15–19. place
        if desc.startswith("place"):
            # Case 1: [['obj1', 'obj2'], 'target', ['extra1', 'extra2']]
            if isinstance(objs_clean, (list, tuple)) and len(objs_clean) >= 2:
                placed = objs_clean[0]
                placed_items = placed if isinstance(placed, list) else [placed]

                target_main = objs_clean[1]
                target_main_name = (
                    target_main[-1]
                    if isinstance(target_main, list)
                    else target_main
                )

                extra_targets = []
                if len(objs_clean) >= 3:
                    extra = objs_clean[2]
                    extra_targets = (
                        extra if isinstance(extra, (list, tuple)) else [extra]
                    )

                placed_str = " and ".join(placed_items)
                target_str = target_main_name

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

            # Case 2: ['storage_box_80', 'storage_box_79']
            if isinstance(objs_clean, (list, tuple)) and all(
                isinstance(o, str) for o in objs_clean
            ):
                return f"place {' and '.join(objs_clean)}"

            # Case 3: only one obj
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

            # Case 1: [objects, src, dst]
            if isinstance(objs_clean, (list, tuple)) and len(objs_clean) >= 3:
                src_objs, src_container, dst_container = objs_clean

                # [[['pepperoni'], ['mushroom']], ...]
                if isinstance(src_objs, (list, tuple)) and all(
                    isinstance(g, (list, tuple)) for g in src_objs
                ):
                    src_groups = []
                    for g in src_objs:
                        src_groups.append(
                            " and ".join(uniq(normalize_list(g)))
                        )
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

            # Case 2: [['obj1','obj2'], 'target']
            if isinstance(objs_clean, (list, tuple)) and len(objs_clean) == 2:
                src_objs, dst_container = objs_clean
                src_str = " and ".join(
                    uniq(flatten_once(normalize_list(src_objs)))
                )
                dst_str = " and ".join(uniq(normalize_list(dst_container)))
                return f"pour {src_str} onto {dst_str}"

            # Case 3: only obj list
            if isinstance(objs_clean, (list, tuple)) and all(
                isinstance(o, str) for o in objs_clean
            ):
                return f"pour {' and '.join(uniq(objs_clean))}"

            # Case 4: only one str
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

