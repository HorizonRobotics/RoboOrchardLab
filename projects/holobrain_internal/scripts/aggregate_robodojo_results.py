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

"""Merge RoboDojo eval results from several AIDI jobs into one benchmark summary.

Why this exists
---------------
The official RoboDojo protocol (see ``robodojo_eval.BENCHMARK_DIMENSIONS``) wants
50 episodes per benchmark task:

* the 12 ``Generalization`` tasks take 25 episodes from ``<task>`` plus 25 from
  ``<task>_random``  -> a run submitted with ``--eval_num 25`` already satisfies this;
* the other 30 tasks need 50 episodes from a single run-config -> those require
  ``--eval_num 50``.

So a protocol-complete summary can be assembled from two jobs: the Generalization
half from the ``--eval_num 25`` run and the remaining 30 tasks from an
``--eval_num 50`` backfill run. This script pulls the per-run-config
``_result.json`` files out of each job, then hands them to the *existing*
``robodojo_eval._write_benchmark_summary`` so the scoring rules stay identical to
the in-repo evaluator - no metric logic is reimplemented here.

Example
-------
    python aggregate_robodojo_results.py \\
        --gen-job bcloud-bj-zone1-a52719406c5c \\
        --nongen-job bcloud-bj-zone1-xxxxxxxxxxxx \\
        --label 20k --out-dir ./results_20k
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

import robodojo_eval as rde  # noqa: E402

RESULT_ROOT = "output/robodojo_eval_results"


def _run(cmd: list[str], timeout: int = 300) -> str:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed (%d): %s\n%s"
            % (proc.returncode, " ".join(cmd), proc.stderr.strip()[:400])
        )
    return proc.stdout


def _parse_listing(text: str) -> list[tuple[str, str]]:
    """Return ``(type, name)`` rows from ``aidictl job logs list`` output.

    The command prints a TSV table plus a trailing summary line and, depending on
    the client version, an upgrade notice - both are ignored here.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0] in ("DIR", "FILE"):
            rows.append((parts[0], parts[3].strip()))
    return rows


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of ``aidictl job logs cat`` output."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in output")
    return json.loads(text[start : end + 1])


def _run_config_dir(run_config: str, seed: int) -> str:
    return "/".join(
        [
            RESULT_ROOT,
            rde.BENCHMARK_NAME,
            run_config,
            rde.POLICY_NAME,
            "arx_x5",
            "%d_ckpt_name=%s,action_type=%s"
            % (seed, rde.CHECKPOINT_NAME, rde.ACTION_TYPE),
        ]
    )


def fetch_one(
    job_id: str, run_config: str, seed: int, cache_dir: Path
) -> tuple[str, Path | None, str]:
    """Fetch ``_result.json`` for one run-config. Returns (name, path, status)."""
    dest = cache_dir / ("%s.json" % run_config)
    if dest.is_file():
        try:
            json.loads(dest.read_text(encoding="utf-8"))
            return run_config, dest, "cached"
        except (OSError, json.JSONDecodeError):
            dest.unlink(missing_ok=True)

    parent = _run_config_dir(run_config, seed)
    try:
        listing = _run(["aidictl", "job", "logs", "list", job_id, parent])
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        return run_config, None, "list-failed: %s" % str(exc)[:120]

    run_ids = [name for kind, name in _parse_listing(listing) if kind == "DIR"]
    if not run_ids:
        return run_config, None, "no-run-id (task not started or no output)"
    if len(run_ids) > 1:
        # A retried task can leave several run dirs behind; the newest listing
        # entry comes first, so prefer it but say so.
        status_note = "multiple-run-ids:%d" % len(run_ids)
    else:
        status_note = "ok"

    path = "%s/%s/_result.json" % (parent, run_ids[0])
    try:
        raw = _run(["aidictl", "job", "logs", "cat", job_id, path])
        payload = _extract_json(raw)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return run_config, None, "cat-failed: %s" % str(exc)[:120]

    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_config, dest, status_note


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gen-job",
        required=True,
        help="job supplying the 12 Generalization tasks (24 run-configs)",
    )
    parser.add_argument(
        "--nongen-job",
        help="job supplying the 30 non-Generalization tasks; "
        "defaults to --gen-job (single-job mode)",
    )
    parser.add_argument("--label", default="run", help="tag used in printouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="where fetched _result.json files live (default <out-dir>/_cache)",
    )
    parser.add_argument("--jobs", type=int, default=8, help="fetch concurrency")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="RUN_CONFIG=JOB_ID",
        help="take one run-config from a specific job instead of the default "
        "gen/non-gen assignment; repeatable. Useful when a single run-config "
        "came up short and was re-run separately.",
    )
    parser.add_argument(
        "--standalone-episodes",
        type=int,
        help="override robodojo_eval.STANDALONE_EPISODES. Produces a NON-PROTOCOL "
        "summary (e.g. 25 to view a 25-episode run); omit to keep the official 50.",
    )
    args = parser.parse_args(argv)

    nongen_job = args.nongen_job or args.gen_job
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (out_dir / "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.standalone_episodes is not None:
        print(
            "!! STANDALONE_EPISODES overridden %d -> %d; the summary below is NOT "
            "the official protocol."
            % (rde.STANDALONE_EPISODES, args.standalone_episodes)
        )
        rde.STANDALONE_EPISODES = args.standalone_episodes

    gen_tasks = list(rde.BENCHMARK_DIMENSIONS["Generalization"])
    nongen_tasks = [t for t in rde.BENCHMARK_TASKS if t not in set(gen_tasks)]

    # run-config -> job that produced it
    wanted: dict[str, str] = {}
    for task in gen_tasks:
        wanted[task] = args.gen_job
        wanted[task + "_random"] = args.gen_job
    for task in nongen_tasks:
        wanted[task] = nongen_job

    for spec in args.override:
        name, sep, job = spec.partition("=")
        if not sep or not job:
            parser.error("--override expects RUN_CONFIG=JOB_ID, got %r" % spec)
        if name not in wanted:
            parser.error("--override names unknown run-config %r" % name)
        wanted[name] = job
        print("  override: %s <- %s" % (name, job))

    print(
        "[%s] %d benchmark tasks -> %d run-configs "
        "(Generalization %d x2 from %s, other %d from %s)"
        % (
            args.label,
            len(rde.BENCHMARK_TASKS),
            len(wanted),
            len(gen_tasks),
            args.gen_job,
            len(nongen_tasks),
            nongen_job,
        )
    )

    result_paths: dict[str, Path] = {}
    problems: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(fetch_one, job, name, args.seed, cache_dir)
            for name, job in sorted(wanted.items())
        ]
        for done, future in enumerate(futures, 1):
            name, path, status = future.result()
            if path is None:
                problems[name] = status
            else:
                result_paths[name] = path
                if status not in ("ok", "cached"):
                    problems[name] = status
            if done % 10 == 0 or done == len(futures):
                print("  fetched %d/%d" % (done, len(futures)))

    print("[%s] usable run-configs: %d/%d" % (args.label, len(result_paths), len(wanted)))
    if problems:
        print("[%s] %d run-config(s) need attention:" % (args.label, len(problems)))
        for name, status in sorted(problems.items()):
            print("    %-42s %s" % (name, status))

    # Raw per-run-config view, so the write-up does not have to re-parse anything.
    #
    # Careful: ``_result.json`` is not self-consistent about units - its
    # ``success_rate`` is a fraction in [0, 1] while its ``score`` is already a
    # percentage in [0, 100]. Rather than trust either, recompute both from the
    # per-episode entries (the same source ``_benchmark_entry_metrics`` uses) so
    # everything in this file is in percent. The reported values are kept
    # alongside as a cross-check.
    details = {}
    scale_mismatch = []
    for name, path in sorted(result_paths.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("details") or {}
        n = len(entries)
        successes = sum(1 for v in entries.values() if v.get("success"))
        score_sum = sum(float(v.get("score") or 0.0) for v in entries.values())
        sr_pct = (successes / n * 100.0) if n else None
        score_pct = (score_sum / n * 100.0) if n else None
        details[name] = {
            "success_rate_percent": sr_pct,
            "score_percent": score_pct,
            "episodes": n,
            "source_job": wanted[name],
            "reported_success_rate": payload.get("success_rate"),
            "reported_score": payload.get("score"),
        }
        # Assert the interpretation above still holds for this data.
        if n:
            rep_sr = float(payload.get("success_rate") or 0.0) * 100.0
            rep_score = float(payload.get("score") or 0.0)
            if abs(rep_sr - sr_pct) > 0.01 or abs(rep_score - score_pct) > 0.01:
                scale_mismatch.append(name)
    if scale_mismatch:
        print(
            "  !! %d run-config(s) disagree with the assumed unit convention "
            "(success_rate=fraction, score=percent): %s"
            % (len(scale_mismatch), ", ".join(scale_mismatch[:8]))
        )
    (out_dir / ("runconfig_details_seed_%d.json" % args.seed)).write_text(
        json.dumps(details, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Official aggregation - reuse the evaluator's own implementation.
    rde._write_benchmark_summary(result_paths, out_dir, args.seed)

    summary_path = out_dir / ("benchmark_summary_seed_%d.json" % args.seed)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        "[%s] complete=%s  tasks=%d/%d  avg_success_rate=%s"
        % (
            args.label,
            summary.get("complete"),
            summary.get("completed_tasks"),
            summary.get("num_tasks"),
            summary.get("average_success_rate"),
        )
    )
    for dim, metrics in summary.get("dimension_metrics", {}).items():
        print(
            "    %-16s SR=%-8s score=%-8s (%d/%d tasks)"
            % (
                dim,
                metrics.get("success_rate"),
                metrics.get("score"),
                metrics.get("completed_tasks"),
                metrics.get("num_tasks"),
            )
        )
    if summary.get("incomplete_tasks"):
        print("    incomplete: %s" % json.dumps(summary["incomplete_tasks"]))
    if summary.get("missing_tasks"):
        print("    missing: %s" % ", ".join(summary["missing_tasks"]))
    print("[%s] wrote %s" % (args.label, summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
