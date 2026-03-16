# Data Directory Structure for packer and checker
```
data/
└── [user_name]/
    └── [task_name]/
        └── [episode_id]/  # e.g. episode_2025_10_17-14_57_18
            ├── episode_2025_10_17-14_57_18_0.mcap
            ├── episode_meta.json
            └── metadata.yaml
```


# Pack

## Local run
```bash
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.mcap_packer \
    --input_path ${input_path} \
    --output_path ${output_path}/${user_names}-${task_names}-20260119 \
    --urdf "./urdf/piper_description_dualarm_new.urdf" \
    --user_names ${user_names} \
    --task_names ${task_names} \
    --num_steps_per_shard 200 \
    --date_prefix ${date_prefix} \
    $@
```

## Cluster run
```bash
RoboOrchardJob-AIDISubmit submit_from_config --config robo_orchard_lab/dataset/horizon_manipulation/tools/submit_pack.json  # example sumbit config
```

# Data check and save as video
## Local run
```bash
ulimit -n 65536
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.mcap_checker \
    --input_path ${input_path} \
    --output_path ${output_path}/${user_names}-${task_names}-20260119 \
    --urdf "./urdf/piper_description_dualarm_new.urdf" \
    --user_names ${user_names} \
    --task_names ${task_names} \
    --date_prefix ${date_prefix} \
    --num_workers 10 \
    $@
```

## Cluster run
```bash
RoboOrchardJob-AIDISubmit submit_from_config --config robo_orchard_lab/dataset/horizon_manipulation/tools/submit_check.json  # example sumbit config
```

# Upload data to bucket
```bash
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.upload_data \
    --input_path ${input_path} \
    --output_path ${output_path} \
    --user_names ${user_names} \
    --task_names ${task_names} \
    --date_prefix ${date_prefix} \
    --num_workers 10 \
    --token ${aidi_token} \
    $@
```

# Args examples
```bash
user_names=userA
# or
user_names=userA,userB

task_names=empty_cup_place
# or
task_names=empty_cup_place,place_shoes

date_prefix=2026_01_19
# or
date_prefix=2026_01_19-10_26
# or
date_prefix=2026_01_19,2026_01_20

```
