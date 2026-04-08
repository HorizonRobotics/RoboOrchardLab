# 📁 1. Data Directory Structure
```
data/
└── [user_name]/
    └── [task_name]/
        └── [episode_id]/  # e.g. episode_2025_10_17-14_57_18
            ├── episode_2025_10_17-14_57_18_0.mcap
            ├── episode_meta.json
            └── metadata.yaml
```


# 📦 2. Pack

## 2.1 Local run
```bash
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.mcap_packer \
    --input_path ${input_path} \
    --output_path ${output_path}/${user_names}-${task_names}-${date_prefix} \
    --urdf "./urdf/piper_description_dualarm_new.urdf" \
    --user_names ${user_names} \
    --task_names ${task_names} \
    --num_steps_per_shard 200 \
    --date_prefix ${date_prefix} \
    $@
```

## 2.2 Cluster run
```bash
RoboOrchardJob-AIDISubmit submit_from_config --config robo_orchard_lab/dataset/horizon_manipulation/tools/submit_pack.json  # example sumbit config
```

# 🎬 3. Data check and save as video
## 3.1 Local run
```bash
ulimit -n 65536
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.mcap_checker \
    --input_path ${input_path} \
    --output_path ${output_path}/${user_names}-${task_names}-20260119 \
    --urdf "./urdf/piper_description_dualarm_new.urdf" \
    --user_names ${user_names} \
    --task_names ${task_names} \
    --date_prefix ${date_prefix} \
    --inspect_config ${inspect_config_path} \
    --num_workers 10 \
    [--enable_ffmpeg_log] \
    $@
```

For a Chinese guide focused on the current rule-based inspection pipeline, see [rule_based_check.md](rule_based_check.md).

The checker always writes inspection reports:

- `inspect_mcap_result.log`: merged list of all episodes with `PASS` / `ERROR` prefix
- `inspect_full_log.log`: full inspection log
- `inspect_error_log.log`: extracted error log summary
- `inspect_error_log.html`: HTML view of `inspect_error_log.log` with anchor links for manual review jump
- ffmpeg encoding logs are suppressed by default; add `--enable_ffmpeg_log` when you need ffmpeg console output for debugging

When video export is enabled, the checker also writes:

- per-episode rendered videos
- `concat_videos.mp4`: concatenated review video
- `manual_review.html`: standalone manual review page copied into the output directory
- `manual_review_timeline.json`: concat timeline to mcap/video mapping used by the review page
- `manual_review_failures.json`: persisted human review failures, pre-seeded with the same `ERROR` items shown in `inspect_mcap_result.log`

### 3.1.1 Manual review page

`manual_review.html` is generated only when video export is enabled. It is a standalone static page, so you can download the whole output directory or just open that HTML file directly.

Behavior:

- the page plays `concat_videos.mp4` through a relative path in the same output directory
- the page shows the current `mcap_id` and single-video filename based on playback time
- the right panel keeps the current segment metadata and review progress
- the action buttons sit beneath the video player on the left
- the failure table is prefilled with the same non-pass entries that appear as `ERROR` in `inspect_mcap_result.log`
- clicking `不通过` adds the current mcap to the top failure table
- clicking `撤销当前` removes the current mcap from the failure table
- clicking a row in the failure table seeks the concat player back to slightly before that marked point for quick re-check
- rule-seeded failures include a clickable short system note that opens `inspect_error_log.html` at the corresponding error block
- the `Current Video` item on the right is a direct link to the current clip file
- clicking `导出 JSON` downloads the latest `manual_review_failures.json`
- clicking `复制 mcap_id` copies all failed `mcap_id` values one-per-line

If you want to serve the same directory through a local HTTP endpoint, you can use the optional review server:

```bash
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.manual_review_server \
    --output_path ${check_output_path} \
    --port 7000
```

Then open:

```text
http://127.0.0.1:7000/manual-review
```

## 3.2 Cluster run
```bash
RoboOrchardJob-AIDISubmit submit_from_config --config robo_orchard_lab/dataset/horizon_manipulation/tools/submit_check.json  # example submit config
```

# ☁️ 4. Upload data to bucket
```bash
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.upload_data \
    --input_path ${input_path} \
    --output_path ${output_path} \
    --user_names ${user_names} \
    --task_names ${task_names} \
    --date_prefix ${date_prefix} \
    --num_workers 10 \
    --token ${aidi_token} \
    [--skip_existing_same_size] \
    $@
# Args examples
# user_names=userA
# # or
# user_names=userA,userB

# task_names=empty_cup_place
# # or
# task_names=empty_cup_place,place_shoes

# date_prefix=2026_01_19
# # or
# date_prefix=2026_01_19-10_26
# # or
# date_prefix=2026_01_19,2026_01_20
```


# 🌐 5. Remote upload orchestration

Use `remote_upload_orchestrator.py` on a server in the same LAN to trigger `upload_data.py` on multiple collection machines over SSH.

- Connects to each collection machine with SSH / SCP
- Uploads the local `upload_data.py` to each remote host at runtime
- Optionally activates a remote virtual environment before execution
- Supports parallel execution across multiple hosts
- Writes per-run per-host logs under `.remote_upload_logs/`
- Cleans up the uploaded remote script and tries to stop remote tasks when the local process is interrupted

## 5.1 Configuration

Copy `remote_upload_hosts.example.json` and edit it for your machines.

The config file must be a JSON object with this structure:

```json
{
  "defaults": {
    "remote_python": "python3",
    "output_path": "/horizon-bucket/.../raw_data",
    "user_names": "user_a,user_b",
    "task_names": "task_a,task_b",
    "date_prefix": "2026_03_18,2026_03_19",
    "num_workers": 32,
    "connect_timeout": 10,
    "token": "<aidi-token>"
  },
  "hosts": [
    {
      "name": "beijing_1",
      "host": "10.103.80.13",
      "input_path": "/home/user/data/inference/",
      "ssh_user": "user",
      "remote_venv_activate": "/home/user/workspace/venv/robot-env/bin/activate"
    }
  ]
}
```

### 5.1.1 Merge rules

- `defaults` defines shared values reused by all hosts
- each item in `hosts` can override any field from `defaults`
- the final runtime config for one machine is `defaults + host override`
- `hosts` must be a non-empty list

### 5.1.2 Required fields

The following fields must be available after merging `defaults` and each host item:

- `host`
  - remote machine IP or hostname
- `input_path`
  - raw data root on the remote machine
- `output_path`
  - upload destination passed to `upload_data.py`
- `user_names`
  - comma-separated user names passed to `upload_data.py`
- `task_names`
  - comma-separated task names passed to `upload_data.py`

### 5.1.3 Common optional fields

- `name`
  - host alias used in logs and status output
  - if omitted, `host` is used
- `ssh_user`
  - remote SSH username
- `ssh_key`
  - private key path used for both `scp` and `ssh`
- `port`
  - SSH port, default is `22`
- `connect_timeout`
  - SSH/SCP connect timeout in seconds, default is `10`
- `remote_python`
  - Python executable used on the remote machine
  - default is `python3`
- `remote_venv_activate`
  - optional activation script sourced before running `upload_data.py`
  - useful when the remote dependencies are installed in a venv
- `date_prefix`
  - optional comma-separated date filter passed to `upload_data.py`
- `token`
  - optional AIDI token passed to `upload_data.py`
- `num_workers`
  - worker count passed to `upload_data.py`
  - default is `4` if not provided

### 5.1.4 Field semantics and current behavior

- You do **not** need to pre-deploy `upload_data.py` on each collection machine.
  - the orchestrator copies the local `upload_data.py` to a temporary path under `/tmp/`
  - then it runs that uploaded script remotely
- `output_path` is the destination bucket/path for uploaded data, not a local remote directory
- `input_path` must point to the raw data directory on the remote host itself
- if `remote_venv_activate` is configured, the effective remote command is:
  - `source <venv_activate> && <remote_python> <uploaded_script> ...`
- if the local orchestrator process receives `SIGINT` or `SIGTERM`, it will:
  - try to stop matching remote processes
  - remove the uploaded temporary script on each active host

### 5.1.5 Practical recommendations

- Prefer storing shared fields in `defaults` and only host-specific values in `hosts`
- Use `name` values that are stable and human-readable, because per-host log filenames are derived from them
- Prefer SSH key login over password-based login
- Verify that each remote host can:
  - be reached by SSH from the orchestrator machine
  - run the configured `remote_python`
  - import all dependencies required by `upload_data.py`
  - access the configured `input_path`
- Treat `token` as sensitive data and avoid committing a real production token into the repository

## 5.2 Run

```bash
python robo_orchard_lab/dataset/horizon_manipulation/tools/remote_upload_orchestrator.py \
  --config robo_orchard_lab/dataset/horizon_manipulation/tools/remote_upload_hosts.example.json \
  --max-parallel 3
```

Arguments:

- `--config`
  - path to the remote upload JSON config
- `--max-parallel`
  - maximum number of hosts uploaded concurrently
  - default: number of configured hosts
- `--log-dir`
  - base directory for run logs
  - default: `.remote_upload_logs`

## 5.3 What happens during one run

For each host, the orchestrator will:

1. merge `defaults` with the host item
2. copy local `upload_data.py` to a temporary file on the remote machine
3. connect with SSH and run the uploaded script
4. stream stdout/stderr into the per-host log file
5. print a compact success/failure summary when that host finishes

Logs are stored under a timestamped directory like:

- `.remote_upload_logs/{datetime}/main-host logs from the orchestrator run`
- `.remote_upload_logs/{datetime}/<host-name>.log`

## 5.4 Failure handling and notes

- if one host fails, other hosts continue to run
- the overall process returns exit code `1` if any host fails
- if all hosts succeed, the script returns `0`
- if interrupted locally, the script returns `130`
- host-specific cleanup output is appended into the corresponding host log file

# 🖥️ 6. Data App

feishu docs: https://horizonrobotics.feishu.cn/wiki/AEAewYsQPiK6FckfRB9cGerrnQb?docs_banner_login=

`app.py` provides a lightweight web UI for browsing collected episodes, maintaining cache, submitting downstream jobs, and triggering multi-host remote upload.

It is designed for daily data operations around a data root such as:

- viewing statistics and trends for the current dataset
- searching/filtering episodes by `user_name`, `task_name`, and `date_prefix`
- downloading a selected episode as a zip package
- running **check** jobs from the current search result
- running **pack** jobs from the current search result
- triggering **remote upload** to multiple hosts from the browser

## 6.1 Main capabilities

### 6.1.1 Dataset statistics and visualization

The home page shows aggregated statistics for the current `data_root`, including:

- total episode count
- total duration
- per-day totals
- per-user totals
- per-task totals
- an overview chart that can switch between:
  - grouping by user or task
  - metric by duration or episode count

All statistics are computed from cached episode records unless a refresh is explicitly requested.

### 6.1.2 Search and filtering

The page supports filtering by:

- `data_root`
- `user_name`
- `task_name`
- `date_prefix`

Notes:

- `user_name`, `task_name`, and `date_prefix` accept comma-separated values
- filtering is applied on top of cached records, so common searches are fast
- pagination is supported through `page` and `page_size`

### 6.1.3 Cache and refresh

The app keeps episode metadata in cache to avoid rescanning the full directory tree on every page load.

Supported refresh modes:

- **initial scan**
  - when a `data_root` is visited for the first time and cache does not exist yet
  - the page immediately enters a loading state and starts a background scan task
- **full refresh**
  - rescans the whole monitored path and rebuilds the cache
  - used by the UI action similar to "Refresh Statistics"
- **partial refresh by date_prefix**
  - refreshes only records matching the specified `date_prefix`
  - useful when new data was collected for a known day and a full scan would be too slow
- **automatic refresh**
  - cached paths that have already been accessed are refreshed automatically every day at **2:00 AM**

Cache storage:

- in-process memory, for fast reuse in the current service instance
- on-disk cache files under `CACHE_DIR`, so cache can survive service restarts

### 6.1.4 Episode list and download

For the current search result, the page shows an episode table with:

- `episode_id`
- `user_name`
- `task_name`
- day
- duration
- episode path

Each episode can be downloaded from the UI:

- the app first reports packaging status and file count
- then it streams a zip package to the browser

### 6.1.5 Submit check / pack jobs

The app can generate submit configurations directly from the **current filtered result** and then submit jobs.

Supported job types:

- **check**: quality inspection / video generation workflow
- **pack**: dataset packaging workflow

Workflow:

1. search for the target episodes
2. click the corresponding submit action
3. the app generates an initial `submit_config.json`
4. edit the config in the browser if needed
5. confirm submission
6. monitor real-time submission logs in the page

The generated submit config inherits the current search conditions and is based on `submit_check.json` or `submit_pack.json` from `ROBO_ORCHARD_LAB_DIR`.

### 6.1.6 Remote upload from the browser

The app also exposes the multi-host remote upload workflow in the UI.

Features:

- load a default remote upload config from `REMOTE_UPLOAD_CONFIG_PATH`
- dynamically update the default config from the current search form
- edit the remote upload config in the browser before launch
- start a background remote upload task
- view:
  - main process log
  - per-host logs
- cancel an ongoing remote upload task
- minimize and restore the upload window
- keep the upload session recoverable across page navigation/refresh in the browser

When the default config is generated for the UI, the following search fields are injected into the config automatically:

- `data_root` -> `defaults.output_path`
- `user_name` -> `defaults.user_names`
- `task_name` -> `defaults.task_names`
- `date_prefix` -> `defaults.date_prefix`

## 6.2 Configuration

The app reads configuration from environment variables. A project-level `.env` file is supported and overrides inherited environment variables when the app starts.

Important variables:

- `DATA_ROOT`
  - default monitored data root used by the page when `data_root` is not specified
- `HOST`
  - bind host for the Flask development server
- `PORT`
  - service port
- `CACHE_DIR`
  - directory for persistent cache files
  - default: `.cache/` under the project root
- `SUBMIT_CONFIG_DIR`
  - temporary directory used to write generated submit config JSON files
  - default: `.submit_configs/` under the project root
- `SUBMIT_CONFIG_PATCH_PATH`
  - optional patch JSON applied on top of generated submit configs
- `SUBMIT_JOB_CLEAR_PROXY`
  - defaults to `true`
  - clears proxy-related environment variables before calling submit commands
- `ROBO_ORCHARD_LAB_DIR`
  - root directory of Robo Orchard Lab
  - must contain:
    - `dataset/horizon_manipulation/tools/submit_check.json`
    - `dataset/horizon_manipulation/tools/submit_pack.json`
- `REMOTE_UPLOAD_CONFIG_PATH`
  - path to the default remote upload config JSON template used by the UI
  - default: `remote_upload_hosts.example.json`
- `FLASK_DEBUG`
  - whether to run the Flask development server in debug mode

Path resolution rules:

- absolute paths are used as-is
- relative paths are resolved relative to the directory containing `app.py`

## 6.3 Run

Run with Python:

```bash
python3 robo_orchard_lab/dataset/horizon_manipulation/tools/app.py
```

Or run with gunicorn:

```bash
gunicorn -w 4 --threads 4 --timeout 3000 -b 0.0.0.0:8000 robo_orchard_lab.dataset.horizon_manipulation.tools.app:app
```

## 6.4 UI workflow summary

Typical workflow in the page:

1. open the dashboard for a target `data_root`
2. wait for the initial cache build if this path has never been scanned before
3. search/filter by `user_name`, `task_name`, and `date_prefix`
4. inspect statistics and episode list
5. optionally:
   - download episodes
   - submit **check** jobs
   - submit **pack** jobs
   - trigger **remote upload**

## 6.5 Background tasks exposed by the app

The app manages several kinds of background tasks in memory:

- **scan tasks**
  - full refresh or partial refresh of cached episode records
- **submit tasks**
  - submission of check/pack jobs
- **remote upload tasks**
  - execution of `remote_upload_orchestrator.py`

The UI polls task APIs to show real-time progress and status.

## 6.6 API overview

### Page and summary

- `GET /`
  - render the dashboard page
- `GET /api/summary`
  - return summary JSON for the current filters
  - query parameters:
    - `data_root`
    - `user_name`
    - `task_name`
    - `date_prefix`
    - `refresh`
    - `page`
    - `page_size`

### Scan / refresh

- `POST /api/scan-tasks`
  - create a full or partial refresh task
  - request body includes:
    - `data_root`
    - `refresh_mode` (`full` or `partial`)
    - `date_prefix` (required for `partial`)
- `GET /api/scan-tasks/<task_id>`
  - query scan progress
- `POST /api/scan-tasks/<task_id>/cancel`
  - cancel a running scan task

### Submit jobs

- `POST /api/submit-jobs/prepare`
  - generate initial submit config from the current filtered result
- `POST /api/submit-jobs/<config_id>/submit`
  - submit a prepared config
- `GET /api/submit-jobs/tasks/<task_id>`
  - query submit task status and logs

### Remote upload

- `GET /api/remote-upload/config`
  - return browser-editable default remote upload config
  - query parameters can include current search form fields:
    - `data_root`
    - `user_name`
    - `task_name`
    - `date_prefix`
- `POST /api/remote-upload/tasks`
  - start a remote upload task from a config JSON string
- `GET /api/remote-upload/tasks/<task_id>`
  - query remote upload progress and logs
- `POST /api/remote-upload/tasks/<task_id>/cancel`
  - cancel a running remote upload task

### Download

- `GET /api/download`
  - download a single episode as a zip package
- `GET /api/download-status`
  - return packaging status and file count for one episode

## 6.7 Cache behavior

- On the first visit to a monitored path, the app scans the disk and creates a cache
- If the monitored path has no cache yet, the first page load immediately shows the same scan progress overlay used by "Refresh Statistics" so users can see that the scan is in progress
- On later page loads, searches, and filters, data is read from the cache by default instead of rescanning the entire directory tree
- A rescan happens only when you click "Refresh Statistics" in the UI or pass `refresh=1` to the API
- The cache is stored in both:
  - in-process memory (to speed up the current service instance)
  - the `.cache/` directory on disk (so it can be reused after a service restart)

## 6.8 Automatic refresh

- The service automatically refreshes caches for all previously accessed monitored paths every day at **2:00 AM**
- Only paths already present in the cache are refreshed; it does not proactively scan new paths that have never been accessed
