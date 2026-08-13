# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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
import ast
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml
from robo_orchard_core.utils.logging import LoggerManager

LOG_FORMAT = (
    "%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s"
)
logger = LoggerManager(format=LOG_FORMAT, level=logging.INFO).get_child(
    __file__
)

BENCHMARK_NAME = "RoboDojo"
POLICY_NAME = "holobrain_robodojo_policy"
CHECKPOINT_NAME = "holobrain"
ACTION_TYPE = "joint"
EVAL_ENV_NAME = "RoboDojo"

BENCHMARK_DIMENSIONS = {
    "Generalization": (
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ),
    "Precision": (
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ),
    "Long-Horizon": (
        "put_bottles_into_dustbin",
        "fill_pen_holder",
        "classify_objects",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ),
    "Memory": (
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ),
    "Open": (
        "align_blocks",
        "general_pickup",
        "stack_blocks_by_language",
        "solve_equation",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ),
}
BENCHMARK_TASKS = tuple(
    task
    for dimension_tasks in BENCHMARK_DIMENSIONS.values()
    for task in dimension_tasks
)
GENERALIZATION_TASKS = frozenset(BENCHMARK_DIMENSIONS["Generalization"])
STANDALONE_EPISODES = 50
PAIRED_HALF_EPISODES = 25


@dataclass(frozen=True)
class TaskRunResult:
    task_name: str
    result_path: Path
    log_path: Path
    return_code: int


def _audit_episode_videos(result: "TaskRunResult", episode_count: int) -> None:
    """Warn when finished episodes have no video, instead of leaving it silent.

    Videos are named episode_<index>_<cam>_<tag>.mp4 with the same `index` the
    `details` map is keyed by (eval_env.py:798, :820), so the set of indices
    with videos should equal the set of episodes. Locally it does not: the
    first episode's videos are absent while every later one is present, and the
    same code on AIDI leaves all of them.

    The count is what matters, not the cause. An eval whose success
    videos are the evidence should not be able to lose one without
    saying so -- 9 successes with 8 success videos looks exactly like
    an off-by-one in the naming, and reading it that way costs more
    than this check does.
    """
    run_dir = result.result_path.parent
    try:
        indices = {
            name.split("_")[1]
            for name in os.listdir(run_dir)
            if name.startswith("episode_") and name.endswith(".mp4")
        }
    except OSError as error:
        logger.warning("Could not audit videos in %s: %s", run_dir, error)
        return
    if not indices:
        logger.warning(
            "Task %s: %d episodes but no videos at all in %s",
            result.task_name,
            episode_count,
            run_dir,
        )
        return
    missing = episode_count - len(indices)
    if missing > 0:
        expected = {f"{i:07d}" for i in range(episode_count)}
        logger.warning(
            "Task %s: %d episodes but only %d have videos (missing %s). The "
            "scores are unaffected -- _result.json is written per episode -- "
            "but that many episodes have no footage to review.",
            result.task_name,
            episode_count,
            len(indices),
            ", ".join(sorted(expected - indices)[:5]) or "unknown",
        )


def _log_task_result(result: TaskRunResult) -> None:
    if not result.result_path.is_file():
        logger.error(
            "Task %s finished without a result: exit_code=%d log=%s",
            result.task_name,
            result.return_code,
            result.log_path,
        )
        return

    try:
        payload = json.loads(result.result_path.read_text(encoding="utf-8"))
        success_rate = float(payload["success_rate"])
        eval_time = payload.get("eval_time", "unknown")
        if isinstance(eval_time, int) and eval_time > 0:
            _audit_episode_videos(result, eval_time)
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        logger.error(
            "Task %s produced an invalid result: %s log=%s",
            result.task_name,
            error,
            result.log_path,
        )
        return

    logger.info(
        "Task %s finished: success_rate=%.2f%% eval_time=%s exit_code=%d",
        result.task_name,
        success_rate * 100,
        eval_time,
        result.return_code,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_source", type=Path, required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_processor", required=True)
    parser.add_argument(
        "--valid_action_step",
        type=int,
        default=None,
        help=(
            "How many of the predicted actions to execute before observing "
            "again. Defaults to the policy's deploy.yml (32). This is also "
            "the memory's write stride at inference: the bank gets one entry "
            "per forward, so 32 means one entry per 32 env frames while "
            "training writes one per frame. Lowering it costs 32/k times the "
            "eval compute and buys alignment with how the bank was trained."
        ),
    )
    parser.add_argument(
        "--robodojo_root", type=Path, default=Path("/opt/robodojo")
    )
    parser.add_argument("--conda_root", type=Path, default=Path("/opt/conda"))
    parser.add_argument(
        "--policy_env", type=Path, default=Path("/opt/holobrain_policy_env")
    )
    parser.add_argument(
        "--assets_dir",
        type=Path,
        default=Path(
            "/horizon-bucket/robot_lab2/datasets/assets/robodojo_assets"
        ),
    )
    parser.add_argument(
        "--env_config_dir",
        type=Path,
        default=Path("/tmp/robodojo-env-config"),
    )
    parser.add_argument("--env_config", default="arx_x5")
    parser.add_argument(
        "--run_tag",
        default="",
        help=(
            "Suffix for every run id, and so for every result path. "
            "Empty by default, which keeps the AIDI result layout and "
            "the resume manifest key byte-identical to before. Local "
            "runs should pass something unique: they all funnel into "
            "one result root, so without a tag each run overwrites the "
            "previous run's _result.json and videos in place."
        ),
    )
    parser.add_argument(
        "--eval_result_dir",
        type=Path,
        default=Path("/job_data/robodojo_eval_results"),
    )
    parser.add_argument(
        "--kit_root",
        type=Path,
        default=Path("/job_data/.cache/isaacsim-kit"),
    )
    parser.add_argument("--kit_args", default="")
    parser.add_argument(
        "--vlm_ckpt_dir",
        type=Path,
        default=Path("/horizon-bucket/robot_lab/users/xuewu.lin/ckpt"),
    )
    parser.add_argument(
        "--urdf_dir",
        type=Path,
        default=Path(
            "/horizon-bucket/robot_lab/users/xuewu.lin/urdf_tmp_v20260711"
        ),
    )
    parser.add_argument("--tasks")
    parser.add_argument("--eval_num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processes_per_gpu", type=int, default=1)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help=(
            "Parallel envs inside one Isaac Sim instance. Needs a policy "
            "whose memory is keyed per env; > 1 also flips eval_batch, since "
            "main.py forces num_envs to 1 while that is false."
        ),
    )
    args = parser.parse_args(argv)
    if args.processes_per_gpu < 1:
        parser.error("--processes_per_gpu must be at least 1")
    env_config_path = Path(args.env_config)
    if (
        not args.env_config
        or env_config_path.name != args.env_config
        or env_config_path.suffix
    ):
        parser.error(
            "--env_config must be a config name without a path or extension"
        )
    return args


def _replace_symlink(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Symlink source does not exist: {source}")
    source = source.resolve()
    target.unlink(missing_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def _prepare_env_config(
    robodojo_root: Path,
    output_dir: Path,
    config_name: str,
) -> Path:
    """Create a runtime env config with camera calibration enabled.

    Args:
        robodojo_root (Path): Root of the RoboDojo installation.
        output_dir (Path): Writable runtime env config directory.
        config_name (str): Config filename stem under ``env_cfg``.

    Returns:
        Path: Path to the generated runtime config.
    """
    source_path = robodojo_root / "env_cfg" / f"{config_name}.yml"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"RoboDojo env config not found: {source_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    output_path.unlink(missing_ok=True)
    shutil.copy2(source_path, output_path)

    config = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid RoboDojo env config: {source_path}")
    if config.get("config_name") != config_name:
        raise ValueError(
            f"RoboDojo env config {source_path} must declare "
            f"config_name: {config_name}"
        )
    try:
        vision_config = config["observation"]["vision"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"RoboDojo env config has no observation.vision: {source_path}"
        ) from error
    if not isinstance(vision_config, dict):
        raise ValueError(
            f"RoboDojo observation.vision must be a mapping: {source_path}"
        )
    vision_config["intrinsic_matrix"] = True
    vision_config["extrinsic_matrix"] = True
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def _same_path(left: object, right: object) -> bool:
    return os.path.normpath(str(left)) == os.path.normpath(str(right))


PATH_OVERRIDE_PROBE = """
import json
from env.global_configs import ASSETS_PATH, ENV_CONFIG_PATH

try:
    from src.eval_client.eval_env import EVAL_RESULT_DIR
except ImportError:
    # A checkout that predates the indirection has no such symbol; it joins
    # the literal instead. Report that as a value so the caller prints the
    # mismatch, which names the fix, rather than an ImportError traceback.
    EVAL_RESULT_DIR = "<hardcoded eval_result>"

print("ROBODOJO_PREFLIGHT " + json.dumps({
    "ROBODOJO_ASSETS_DIR": ASSETS_PATH,
    "ROBODOJO_ENV_CONFIG_DIR": ENV_CONFIG_PATH,
    "ROBODOJO_EVAL_RESULT_DIR": EVAL_RESULT_DIR,
}))
"""


PROBE_NAME = ".robodojo_rename_probe"


def _check_eval_result_dir_writable(eval_result_dir: Path) -> None:
    """Fail before Isaac Sim starts if the landing cannot rename.

    Two writes in the evaluation client are renames, not creates:

      eval_env.py:935  os.replace(env<N>_<cam>.tmp.mp4, episode_...mp4)
          every video is written under a temporary name and finalised
      eval_env.py:710  os.replace(tmp, manifest_path)
          the resume manifest, so a restarted run knows what it finished

    The bucket mounts accept create, append, seek-write and truncate but
    reject rename, so pointing --eval_result_dir at one loses the videos and
    the manifest while _result.json -- an ordinary create -- lands as usual.
    The run then reports a complete success-rate table with no videos behind
    it, and nothing in the log says so. Nine AIDI configs are written that way.

    Probe with the operation itself rather than by matching the path against a
    list of mount prefixes: capability is per-mount and has been observed to
    differ between the dev box and an AIDI pod for the same bucket, so a
    prefix test would be a guess that happens to be right today.

    Every cleanup here is guarded. A filesystem that refuses rename usually
    refuses unlink too, and an unguarded cleanup raises PermissionError from
    the handler, replacing the message that names the fix with one that does
    not -- which is what the first version of this function did.
    """
    eval_result_dir.mkdir(parents=True, exist_ok=True)
    probe = eval_result_dir / (PROBE_NAME + ".tmp")
    target = eval_result_dir / (PROBE_NAME + ".final")

    def _discard(path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    try:
        probe.write_bytes(b"probe")
    except OSError as error:
        raise RuntimeError(
            f"--eval_result_dir {eval_result_dir} is not writable: {error}"
        ) from error

    try:
        os.replace(probe, target)
    except OSError as error:
        leaked = (
            ""
            if _discard(probe)
            else (
                f"\nA {probe.name} marker was left behind; "
                "this landing cannot unlink either. It is inert."
            )
        )
        raise RuntimeError(
            f"--eval_result_dir {eval_result_dir} does not support rename "
            f"({type(error).__name__}: {error}).\n"
            "Videos and the resume manifest are finalised with os.replace "
            "(eval_client/eval_env.py:935 and :710), so this run would report "
            "a full success-rate table with no videos behind it.\n"
            "Point --eval_result_dir at a filesystem that renames -- JFS "
            "locally, /job_data in an AIDI pod -- and copy the products to "
            "the bucket afterwards with cp." + leaked
        ) from error

    if not _discard(target):
        logger.warning(
            "Landing renames but cannot unlink; %s remains. Harmless, but the "
            "evaluation never deletes its own output either, so nothing else "
            "depends on it.",
            target,
        )
    logger.info("Eval-result landing supports rename: %s", eval_result_dir)


def _check_path_overrides(
    *,
    conda_root: Path,
    robodojo_root: Path,
    env: dict[str, str],
    expected: dict[str, Path],
) -> None:
    """Fail before Isaac Sim starts if RoboDojo ignores our path overrides.

    This script steers RoboDojo entirely through three environment variables.
    A checkout that predates them accepts every flag and then quietly uses its
    own baked-in paths. Only one of the three failures is audible, and it is
    audible far too late:

      ROBODOJO_ENV_CONFIG_DIR  the camera-calibration patch written by
          _prepare_env_config is never read, so an env config with
          `intrinsic_matrix: false` reaches the sim unmodified and every
          episode dies mid-run on `camera ... is missing intrinsic_matrix`
      ROBODOJO_ASSETS_DIR      --assets_dir is ignored, silently
      ROBODOJO_EVAL_RESULT_DIR results land in <robodojo_root>/eval_result,
          so _log_task_result finds nothing under --eval_result_dir and
          reports every task as failed -- indistinguishable from a model
          that scored zero

    A static grep would be cheaper and would also be a guess. Probe the real
    thing instead: import the real modules in the same conda env, cwd and
    PYTHONPATH the workers get, and compare what they actually resolve to.
    Costs a few seconds against an eval measured in hours.
    """
    command = [
        str(conda_root / "bin/conda"),
        "run",
        "-n",
        EVAL_ENV_NAME,
        "python",
        "-c",
        PATH_OVERRIDE_PROBE,
    ]
    completed = subprocess.run(
        command,
        cwd=robodojo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    marker = "ROBODOJO_PREFLIGHT "
    payload = next(
        (
            line[len(marker) :]
            for line in completed.stdout.splitlines()
            if line.startswith(marker)
        ),
        None,
    )
    if payload is None:
        # The workers are about to run this exact import, so a probe that
        # cannot even load the modules is a failure worth stopping on rather
        # than a reason to wave the run through.
        raise RuntimeError(
            "RoboDojo path-override preflight could not run in "
            f"{robodojo_root} (exit {completed.returncode}).\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    actual = json.loads(payload)
    mismatches = [
        f"  {name}: asked for {wanted}, RoboDojo resolved {actual.get(name)!r}"
        for name, wanted in expected.items()
        if not _same_path(actual.get(name), wanted)
    ]
    if mismatches:
        raise RuntimeError(
            "The RoboDojo at "
            f"{robodojo_root} ignores this script's path overrides:\n"
            + "\n".join(mismatches)
            + "\n\nIt is missing the env-var indirections in "
            "env/global_configs.py and src/eval_client/{main,eval_env}.py. "
            "Port them from the AIDI image (/opt/robodojo), or the run will "
            "read the wrong env config and write results where nobody looks."
        )
    logger.info("Path-override preflight OK for %s", robodojo_root)


def _worker_run_id(seed: int, worker_id: int, run_tag: str) -> str:
    base = f"aidi_seed_{seed}_worker_{worker_id}"
    return f"{base}_{run_tag}" if run_tag else base


def _available_gpu_ids() -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        gpu_ids = [x.strip() for x in visible_devices.split(",") if x.strip()]
        return [] if gpu_ids == ["-1"] else gpu_ids

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result = None
    if result is not None and result.returncode == 0:
        gpu_ids = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        if gpu_ids:
            return gpu_ids

    try:
        import torch
    except ImportError:
        return []
    return [str(gpu_id) for gpu_id in range(torch.cuda.device_count())]


def _resolve_task_names(args: argparse.Namespace) -> list[str]:
    task_dir = args.robodojo_root / "task/RoboDojo/tasks"
    config_dir = args.robodojo_root / "task/RoboDojo/config"
    task_names = []
    for path in sorted(task_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        classes = {
            node.name
            for node in ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            ).body
            if isinstance(node, ast.ClassDef)
        }
        if (
            path.stem in classes
            and (config_dir / f"{path.stem}.yml").is_file()
        ):
            task_names.append(path.stem)

    if not task_names:
        raise RuntimeError("RoboDojo task inventory is empty")
    if args.tasks is None:
        return task_names

    selected = {x.strip() for x in args.tasks.split(",") if x.strip()}
    unknown = sorted(selected - set(task_names))
    if unknown:
        raise ValueError(f"Unknown RoboDojo task(s): {', '.join(unknown)}")
    selected_tasks = [task for task in task_names if task in selected]
    if not selected_tasks:
        raise ValueError("No RoboDojo tasks selected")
    return selected_tasks


def _allocate_tasks(
    task_names: list[str], gpu_ids: list[str], processes_per_gpu: int
) -> list[tuple[str, list[str]]]:
    worker_count = min(len(task_names), len(gpu_ids) * processes_per_gpu)
    worker_gpu_ids = [
        gpu_ids[index % len(gpu_ids)] for index in range(worker_count)
    ]
    groups = [[] for _ in range(worker_count)]
    for index, task_name in enumerate(task_names):
        groups[index % worker_count].append(task_name)
    return list(zip(worker_gpu_ids, groups, strict=True))


def _task_result_path(
    output_dir: Path,
    task_name: str,
    env_config: str,
    seed: int,
    run_id: str,
) -> Path:
    return (
        output_dir
        / BENCHMARK_NAME
        / task_name
        / POLICY_NAME
        / env_config
        / f"{seed}_ckpt_name={CHECKPOINT_NAME},action_type={ACTION_TYPE}"
        / run_id
        / "_result.json"
    )


def _run_worker(
    *,
    worker_id: int,
    gpu_id: str,
    task_names: list[str],
    env_config: str,
    run_tag: str,
    conda_root: Path,
    policy_dir: Path,
    policy_env: Path,
    robodojo_root: Path,
    output_dir: Path,
    seed: int,
    eval_num: int,
    env: dict[str, str],
) -> list[TaskRunResult]:
    worker_run_id = _worker_run_id(seed, worker_id, run_tag)
    log_dir = output_dir / worker_run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for task_name in task_names:
        run_id = f"{worker_run_id}_{task_name}"
        result_path = _task_result_path(
            output_dir,
            task_name,
            env_config,
            seed,
            run_id,
        )
        result_path.unlink(missing_ok=True)
        log_path = log_dir / f"{task_name}.log"
        task_env = env.copy()
        task_env.update(
            EVAL_NUM=str(eval_num),
            ROBODOJO_RUN_ID=run_id,
            ROBODOJO_FATAL_RESTART_COUNT="0",
        )
        command = [
            str(conda_root / "bin/conda"),
            "run",
            "--no-capture-output",
            "-n",
            EVAL_ENV_NAME,
            "bash",
            str(policy_dir / "eval.sh"),
            BENCHMARK_NAME,
            task_name,
            CHECKPOINT_NAME,
            env_config,
            ACTION_TYPE,
            str(seed),
            gpu_id,
            gpu_id,
            str(policy_env),
            EVAL_ENV_NAME,
        ]
        logger.info(
            "Worker %d GPU %s task %s",
            worker_id,
            gpu_id,
            task_name,
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=robodojo_root,
                env=task_env,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        result = TaskRunResult(
            task_name=task_name,
            result_path=result_path,
            log_path=log_path,
            return_code=completed.returncode,
        )
        results.append(result)
        _log_task_result(result)

    return results


def _write_combined_summary(
    result_paths: dict[str, Path],
    task_names: list[str],
    output_dir: Path,
    seed: int,
    eval_num: int,
) -> list[str]:
    success_rates = {}
    for task_name in task_names:
        result_path = result_paths.get(task_name)
        if result_path is not None and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            success_rates[task_name] = float(result["success_rate"])

    task_success_rates = {
        task: success_rates[task]
        for task in task_names
        if task in success_rates
    }
    missing_tasks = [task for task in task_names if task not in success_rates]
    payload = {
        "average_success_rate": (
            sum(task_success_rates.values()) / len(task_success_rates)
            if task_success_rates
            else 0.0
        ),
        "num_tasks": len(task_names),
        "eval_num_per_task": eval_num,
        "task_success_rates": task_success_rates,
    }
    if missing_tasks:
        payload["missing_tasks"] = missing_tasks

    summary = json.dumps(payload, indent=2)
    (output_dir / f"summary_seed_{seed}.md").unlink(missing_ok=True)
    (output_dir / f"summary_seed_{seed}.json").write_text(
        summary,
        encoding="utf-8",
    )
    logger.info("===== conbined summary ===========")
    logger.info(json.dumps(payload, indent=4))
    return missing_tasks


def _load_benchmark_entries(
    result_path: Path | None,
) -> list[tuple[bool, float]] | None:
    if result_path is None or not result_path.is_file():
        return None

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict):
        return None
    details = result.get("details")
    if not isinstance(details, dict):
        return None

    entries = []
    for layout_id, entry in details.items():
        if not isinstance(entry, dict):
            continue
        try:
            numeric_layout_id = int(layout_id)
            score = float(entry.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        entries.append(
            (numeric_layout_id, bool(entry.get("success", False)), score)
        )
    entries.sort(key=lambda entry: entry[0])
    return [(success, score) for _, success, score in entries]


def _benchmark_entry_metrics(
    entries: list[tuple[bool, float]],
) -> tuple[float, float]:
    success_rate = sum(success for success, _ in entries) / len(entries)
    score = sum(score for _, score in entries) / len(entries)
    return success_rate * 100.0, score * 100.0


def _benchmark_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _collect_benchmark_entries(
    result_paths: dict[str, Path],
    task_name: str,
) -> tuple[list[tuple[bool, float]] | None, dict[str, int]]:
    if task_name in GENERALIZATION_TASKS:
        requirements = (
            (task_name, PAIRED_HALF_EPISODES),
            (f"{task_name}_random", PAIRED_HALF_EPISODES),
        )
    else:
        requirements = ((task_name, STANDALONE_EPISODES),)

    selected_entries = []
    episode_counts = {}
    complete = True
    for result_name, required_episodes in requirements:
        entries = _load_benchmark_entries(result_paths.get(result_name)) or []
        episode_counts[result_name] = len(entries)
        selected_entries.extend(entries[:required_episodes])
        complete &= len(entries) >= required_episodes
    return (selected_entries if complete else None), episode_counts


def _write_benchmark_summary(
    result_paths: dict[str, Path],
    output_dir: Path,
    seed: int,
) -> None:
    """Write one seed using the official RoboDojo benchmark protocol."""
    task_metrics = {}
    missing_tasks = []
    incomplete_tasks = {}

    for task_name in BENCHMARK_TASKS:
        entries, episode_counts = _collect_benchmark_entries(
            result_paths, task_name
        )
        if entries is None:
            if any(episode_counts.values()):
                incomplete_tasks[task_name] = episode_counts
            else:
                missing_tasks.append(task_name)
            continue

        success_rate, score = _benchmark_entry_metrics(entries)
        task_metrics[task_name] = {
            "success_rate": success_rate,
            "score": score,
        }

    dimension_metrics = {}
    for dimension, dimension_tasks in BENCHMARK_DIMENSIONS.items():
        completed = [
            task_metrics[task]
            for task in dimension_tasks
            if task in task_metrics
        ]
        dimension_metrics[dimension] = {
            "success_rate": _benchmark_mean(
                [metric["success_rate"] for metric in completed]
            ),
            "score": _benchmark_mean(
                [metric["score"] for metric in completed]
            ),
            "num_tasks": len(dimension_tasks),
            "completed_tasks": len(completed),
        }

    completed_dimensions = [
        metrics
        for metrics in dimension_metrics.values()
        if metrics["completed_tasks"]
    ]
    payload = {
        "complete": len(task_metrics) == len(BENCHMARK_TASKS),
        "seed": seed,
        "metric_unit": "percent",
        "num_tasks": len(BENCHMARK_TASKS),
        "completed_tasks": len(task_metrics),
        "num_run_configs": len(BENCHMARK_TASKS) + len(GENERALIZATION_TASKS),
        "expected_episodes": len(BENCHMARK_TASKS) * STANDALONE_EPISODES,
        "average_success_rate": _benchmark_mean(
            [metrics["success_rate"] for metrics in completed_dimensions]
        ),
        "average_score": _benchmark_mean(
            [metrics["score"] for metrics in completed_dimensions]
        ),
        "dimension_metrics": dimension_metrics,
        "task_metrics": task_metrics,
        "missing_tasks": missing_tasks,
        "incomplete_tasks": incomplete_tasks,
    }
    (output_dir / f"benchmark_summary_seed_{seed}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    logger.info("===== benchmark summary ===========")
    logger.info(json.dumps(payload, indent=4))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    env = os.environ.copy()
    policy_dir = (
        args.robodojo_root / "XPolicyLab/policy/holobrain_robodojo_policy"
    )

    isaaclab_source = args.robodojo_root / "third_party/IsaacLab/source"
    python_paths = [
        *map(str, sorted(isaaclab_source.glob("*"), reverse=True)),
        str(args.robodojo_root / "third_party/curobo"),
        str(args.robodojo_root),
        str(args.robodojo_root / "XPolicyLab"),
        env.get("PYTHONPATH", ""),
    ]
    logger.info(f"model: {args.model_dir}")
    logger.info(f"model_processor: {args.model_processor}")
    logger.info(f"urdf: {args.urdf_dir}")
    env.update(
        PATH=os.pathsep.join(
            filter(None, [str(args.conda_root / "bin"), env.get("PATH", "")])
        ),
        PYTHONPATH=os.pathsep.join(filter(None, python_paths)),
        HOLOBRAIN_MODEL_DIR=args.model_dir,
        HOLOBRAIN_MODEL_PROCESSOR=args.model_processor,
        ROBODOJO_ASSETS_DIR=str(args.assets_dir),
        ROBODOJO_ENV_CONFIG_DIR=str(args.env_config_dir),
        ROBODOJO_EVAL_RESULT_DIR=str(args.eval_result_dir),
        HOLOBRAIN_VLM_CKPT_DIR=str(args.vlm_ckpt_dir),
        HOLOBRAIN_URDF_DIR=str(args.urdf_dir),
    )
    if args.valid_action_step is not None:
        env["HOLOBRAIN_VALID_ACTION_STEP"] = str(args.valid_action_step)
        logger.info(f"valid_action_step: {args.valid_action_step}")
    if args.num_envs is not None:
        # Both, together: main.py caps num_envs at 1 unless eval_batch is on,
        # so setting only the first is silently a no-op.
        env["ROBODOJO_NUM_ENVS"] = str(args.num_envs)
        env["ROBODOJO_EVAL_BATCH"] = "true" if args.num_envs > 1 else "false"
        logger.info(f"num_envs: {args.num_envs} (eval_batch follows it)")
    env.update(
        OMNI_KIT_ACCEPT_EULA="YES",
        HOME="/tmp/robodojo-home",
    )
    for directory in (
        Path(env["HOME"]),
        args.kit_root,
        args.env_config_dir,
        args.eval_result_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _replace_symlink(args.policy_source, policy_dir)
    _prepare_env_config(
        args.robodojo_root,
        args.env_config_dir,
        args.env_config,
    )
    for name in ("camera", "robot", "scene", "sim"):
        _replace_symlink(
            args.robodojo_root / "env_cfg" / name,
            args.env_config_dir / name,
        )
    _check_path_overrides(
        conda_root=args.conda_root,
        robodojo_root=args.robodojo_root,
        env=env,
        expected={
            "ROBODOJO_ASSETS_DIR": args.assets_dir,
            "ROBODOJO_ENV_CONFIG_DIR": args.env_config_dir,
            "ROBODOJO_EVAL_RESULT_DIR": args.eval_result_dir,
        },
    )
    _check_eval_result_dir_writable(args.eval_result_dir)

    task_names = _resolve_task_names(args)
    gpu_ids = _available_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("No available GPU found for RoboDojo evaluation")

    jobs = []
    result_paths = {}
    for worker_id, (gpu_id, worker_tasks) in enumerate(
        _allocate_tasks(task_names, gpu_ids, args.processes_per_gpu)
    ):
        worker_run_id = _worker_run_id(args.seed, worker_id, args.run_tag)
        for task_name in worker_tasks:
            run_id = f"{worker_run_id}_{task_name}"
            result_paths[task_name] = _task_result_path(
                args.eval_result_dir,
                task_name,
                args.env_config,
                args.seed,
                run_id,
            )

        worker_kit_root = args.kit_root / f"worker_{worker_id}"
        worker_kit_root.mkdir(parents=True, exist_ok=True)
        worker_env = env.copy()
        worker_env["ROBODOJO_KIT_ARGS"] = " ".join(
            filter(
                None,
                [
                    args.kit_args,
                    f"--portable-root {worker_kit_root}",
                    "--/app/extensions/registryEnabled=0",
                ],
            )
        )
        logger.info(
            "Worker %d GPU %s: %s",
            worker_id,
            gpu_id,
            ",".join(worker_tasks),
        )
        jobs.append((worker_id, gpu_id, worker_tasks, worker_env))

    errors: list[Exception] = []
    failed_runs: list[TaskRunResult] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(
                _run_worker,
                worker_id=worker_id,
                gpu_id=gpu_id,
                task_names=worker_tasks,
                env_config=args.env_config,
                run_tag=args.run_tag,
                conda_root=args.conda_root,
                policy_dir=policy_dir,
                policy_env=args.policy_env,
                robodojo_root=args.robodojo_root,
                output_dir=args.eval_result_dir,
                seed=args.seed,
                eval_num=args.eval_num,
                env=worker_env,
            )
            for worker_id, gpu_id, worker_tasks, worker_env in jobs
        ]
        for future in futures:
            try:
                failed_runs.extend(
                    result
                    for result in future.result()
                    if result.return_code != 0
                )
            except Exception as error:
                errors.append(error)

    missing_tasks = _write_combined_summary(
        result_paths,
        task_names,
        args.eval_result_dir,
        args.seed,
        args.eval_num,
    )
    _write_benchmark_summary(
        result_paths,
        args.eval_result_dir,
        args.seed,
    )
    if errors:
        raise errors[0]
    if failed_runs or missing_tasks:
        failed_tasks = sorted(
            {result.task_name for result in failed_runs} | set(missing_tasks)
        )
        raise RuntimeError(
            "RoboDojo evaluation failed for task(s): "
            + ", ".join(failed_tasks)
        )


if __name__ == "__main__":
    main()
