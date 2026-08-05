#!/usr/bin/env python3
"""Generate the four memoryvla-vs-baseline RoboDojo training configs.

Generated rather than hand-copied so the four differ in exactly two fields
and nothing else can drift between the arms -- the comparison is only worth
running if the arms are otherwise identical.

Base: submit_cfg_robodojo_train_100k.json, the config that produced the v9
100k baseline in 07_results.md.
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

ARMS = [
    # (suffix, dataset_specs module, memory on?)
    ("conveyor_mem", "dataset_specs_robodojo_conveyor", True),
    ("conveyor_base", "dataset_specs_robodojo_conveyor", False),
    ("memory6_mem", "dataset_specs_memoryvla_robodojo_memory", True),
    ("memory6_base", "dataset_specs_memoryvla_robodojo_memory", False),
]


def main() -> None:
    base = json.load(open(BASE))
    for suffix, specs, memory in ARMS:
        cfg = json.loads(json.dumps(base))
        name = "holobrain_robodojo_mvla_100k_" + suffix
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
            "max_step": 100000,
            "save_step_freq": 5000,
            "num_workers": 8,
        }
        if memory:
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
        out = os.path.join(HERE, "submit_cfg_%s.json" % name)
        with open(out, "w") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
