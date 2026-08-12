#!/usr/bin/env python3
"""Generate the four memoryvla-vs-baseline RoboDojo training configs.

Generated rather than hand-copied so the four differ in exactly two fields
and nothing else can drift between the arms -- the comparison is only worth
running if the arms are otherwise identical.

Base: submit_cfg_robodojo_train_100k.json, the config that produced the v9
100k baseline in 07_results.md.

Every generated config carries ``execute: false``. These are 100k-step,
16-GPU jobs and regenerating used to strip a guard that had been added by hand
after the arms ran, so an accidental resubmission was one command away and
invisible in the diff. Set it to true deliberately for the arm being submitted.
"""

import json
import os

HERE = "projects/holobrain_internal/common/aidi_submit_config"
BASE = os.path.join(HERE, "submit_cfg_robodojo_train_100k.json")

MEMORYVLA_ON = {
    "enable": True,
    "use_perceptual": True,
    "use_cognitive": True,
    "dataloader_type": "stream",
    "group_size": 16,
    "mem_length": 16,
    "retrieval_layers": 2,
    "use_timestep_pe": True,
    "fusion_type": "gate",
    "consolidate_type": "tome",
    "update_fused": False,
    "episode_stream_sampler": True,
}

# One field apart from MEMORYVLA_ON. At inference the bank takes one entry per
# forward, i.e. one per valid_action_step env frames; at the deployed VAS=32
# that is ~25 entries and ~9 ToMe merges per cover_blocks episode, against ~544
# and ~528 when training samples every frame. The VAS sweep showed the merge
# direction matters -- cover_blocks success 9/50 at VAS=32, 1/50 at 16, 0/50 at
# 8 -- so this arm asks whether closing that gap from the training side helps.
MEMORYVLA_ON_STRIDE32 = dict(MEMORYVLA_ON, stream_frame_stride=32)

# Step counts differ because the datasets differ by 7x and what matters for
# a small set is passes over the data, not optimizer steps. At batch 16 over
# 16 ranks, 100k steps is 79 epochs of the six Memory tasks (328,800 frames)
# but 561 of the single task (101 episodes, 47,975 frames) -- which is
# memorisation, not training. 15k puts the single task at ~84 epochs, next
# to the six-task run's 79. The single-task arm has no v9 baseline to line up
# with anyway -- v9 trained on all of RoboDojo -- so its only comparison is
# its own off arm, and matching epochs is the honest way to hold that fixed.
ARMS = [
    # (label, dataset_specs module, memory on?, max_step, save_step_freq)
    ("15k_conveyor_mem", "dataset_specs_robodojo_conveyor", True, 15000, 2500),
    ("15k_conveyor_base", "dataset_specs_robodojo_conveyor", False, 15000, 2500),  # noqa: E501
    ("100k_memory6_mem", "dataset_specs_memoryvla_robodojo_memory", True, 100000, 5000),  # noqa: E501
    ("100k_memory6_base", "dataset_specs_memoryvla_robodojo_memory", False, 100000, 5000),  # noqa: E501
    ("100k_memory6_mem_stride32", "dataset_specs_memoryvla_robodojo_memory", "stride32", 100000, 5000),  # noqa: E501
]


def main() -> None:
    base = json.load(open(BASE))
    for label, specs, memory, max_step, save_freq in ARMS:
        cfg = json.loads(json.dumps(base))
        name = "holobrain_robodojo_mvla_" + label
        cfg["job_name"] = name
        cfg["workspace_folder"] = (
            "/jfs-public/users/kun01.wu/robo_orchard_lab/aidi_workspace/"
            "submit-" + name
        )
        kwargs = {
            "dataset_specs": "configs/%s.py" % specs,
            "with_depth": True,
            "with_depth_loss": False,
            "batch_size": 16,
            "max_step": max_step,
            "save_step_freq": save_freq,
            "num_workers": 8,
        }
        if memory == "stride32":
            kwargs["memoryvla"] = MEMORYVLA_ON_STRIDE32
        elif memory:
            kwargs["memoryvla"] = MEMORYVLA_ON
        # single-quoted, matching the base: this string is pasted into a
        # shell command, so the JSON must survive word splitting intact.
        blob = json.dumps(kwargs, separators=(",", ":"))
        assert "'" not in blob, blob
        cfg["python_executable"] = (
            "train.py --workspace /job_data --logging_dir /job_tboard "
            "--config configs/config_holobrain_common.py "
            "--kwargs '%s'" % blob
        )
        # Regenerating must not re-arm a job that already ran. Flip to true
        # deliberately when submitting; see the module docstring.
        cfg["execute"] = False
        out = os.path.join(HERE, "submit_cfg_%s.json" % name)
        with open(out, "w") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
