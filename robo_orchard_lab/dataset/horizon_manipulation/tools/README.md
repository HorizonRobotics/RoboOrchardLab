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


# Data Monitor Dashboard

A lightweight web application for monitoring collected data under `DATA_ROOT`.


## Date Inference Rules

The project infers the date for each record using the following priority and formats it as `YYYY-MM-DD`:

1. If `episode_id` matches the `episode_YYYY_MM_DD-...` format, use that date first
2. Otherwise, fall back to the **last modification time (`mtime`) of the episode directory**

## Configuration

Setup .env configuration:

- `DATA_ROOT`: custom data root directory
- `PORT`: service port
- `CACHE_DIR`: persistent cache directory, default is `.cache/` under the project
- `SUBMIT_CONFIG_DIR`: directory used to temporarily generate `submit_config.json` when initializing a submit job from the web UI; default is `.submit_configs/` under the project
- `SUBMIT_CONFIG_PATCH_PATH`: optional path to `submit_config_patch.json`; if the file exists, its content is merged into the generated initial submit config
- `SUBMIT_JOB_CLEAR_PROXY`: defaults to `true`; clears proxy environment variables before calling `RoboOrchardJob-AIDISubmit` to avoid inheriting `HTTP_PROXY` / `HTTPS_PROXY` from VS Code or the terminal
- `ROBO_ORCHARD_LAB_DIR`: root directory of Robo Orchard Lab; it must contain `dataset/horizon_manipulation/tools/submit_check.json` and `submit_pack.json`

The project root supports a `.env` file for centralizing these variables. Values in `.env` override existing system environment variables when the app starts. For path variables defined in `.env`:

- absolute paths are used directly
- relative paths are resolved relative to the directory containing `app.py`

## Run
By python

```bash
python3 robo_orchard_lab/dataset/horizon_manipulation/tools/app.py
```

Or by gunicorn
```bash
gunicorn -w 4 --threads 4 --timeout 3000 -b 0.0.0.0:8000 robo_orchard_lab.dataset.horizon_manipulation.tools.app:app
```

## Cache Behavior

- On the first visit to a monitored path, the app scans the disk and creates a cache
- If the monitored path has no cache yet, the first page load immediately shows the same scan progress overlay used by "Refresh Statistics" so users can see that the scan is in progress
- On later page loads, searches, and filters, data is read from the cache by default instead of rescanning the entire directory tree
- A rescan happens only when you click "Refresh Statistics" in the UI or pass `refresh=1` to the API
- The cache is stored in both:
  - in-process memory (to speed up the current service instance)
  - the `.cache/` directory on disk (so it can be reused after a service restart)

## Automatic Refresh

- The service automatically refreshes caches for all previously accessed monitored paths every day at **2:00 AM**
- Only paths already present in the cache are refreshed; it does not proactively scan new paths that have never been accessed

## API

- `GET /`: dashboard page
- `GET /api/summary`: returns summary statistics as JSON
  - optional parameters: `data_root`, `user_name`, `task_name`, `date_prefix`, `refresh`, `page`, `page_size`