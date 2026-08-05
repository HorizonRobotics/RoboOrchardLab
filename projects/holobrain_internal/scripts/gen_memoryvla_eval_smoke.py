#!/usr/bin/env python3
"""Generate the memoryvla eval-path smoke config.

Success is not a success rate. The 68 memoryvla tensors in this package are
randomly initialised -- it is v10 plus noise -- so the scores mean nothing
and a good one would be more surprising than a zero. What is being asked is
whether the path runs at all, and there are three greppable answers:

  * the process does not raise, and _result.json gets written;
  * "MemoryVLAMemory: inference episode N ended after F forward(s), R of
    which retrieved history" appears with R > 0, which is the only positive
    evidence that the bank was read rather than merely present;
  * "never once retrieved history" does NOT appear.

4 tasks x 5 episodes on 2 GPUs. swap_T first because its episodes are the
shortest of the Memory six (median 276 frames), so it reaches the second
episode -- and therefore the reset() path -- soonest.
"""

import json
import os

HERE = "projects/holobrain_internal/common/aidi_submit_config"
BASE = os.path.join(HERE, "submit_cfg_robodojo_eval_kun_20k_sanity.json")
NAME = "kun01wu_robodojo_eval_mvla_smoke"

PKG = (
    "/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/"
    "holobrain_v10_mvla_randominit_smoke/package/"
)
TASKS = "swap_T,match_and_pick_from_conveyor,cover_blocks,press_by_number"


def main() -> None:
    cfg = json.load(open(BASE))
    cfg["job_name"] = NAME
    cfg["workspace_folder"] = (
        "/jfs-public/users/kun01.wu/robo_orchard_lab/aidi_workspace/"
        "submit-holobrain-robodojo-eval-mvla-smoke"
    )
    vlm = None
    urdf = None
    for line in cfg["cmd"]:
        if "--vlm_ckpt_dir" in line:
            vlm = line.split("--vlm_ckpt_dir", 1)[1].split("\\")[0].strip()
        if "--urdf_dir" in line:
            urdf = line.split("--urdf_dir", 1)[1].split("\\")[0].strip()
    assert vlm and urdf, (vlm, urdf)

    cfg["cmd"] = [
        "export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH",
        "/usr/bin/python3 robodojo_eval.py \\",
        '  --policy_source "${WORKING_PATH}/holobrain_robodojo_policy" \\',
        "  --model_dir '%s' \\" % PKG,
        # renamed upstream when feature/sem_internal landed: the dataset is
        # robodojo_arx_x5a now, and train.py builds the filename from it.
        "  --model_processor robodojo_arx_x5a_processor \\",
        "  --env_config arx_x5 \\",
        "  --eval_num 5 \\",
        "  --processes_per_gpu 1 \\",
        "  --vlm_ckpt_dir %s \\" % vlm,
        "  --urdf_dir %s \\" % urdf,
        "  --tasks %s" % TASKS,
    ]
    out = os.path.join(HERE, "submit_cfg_%s.json" % NAME)
    with open(out, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    print("wrote", out)
    for line in cfg["cmd"]:
        print("   ", line)


if __name__ == "__main__":
    main()
