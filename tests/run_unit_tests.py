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

"""Run the unit-test suite in phases and merge the resulting reports."""

from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PYTHON_WARNING_FILTERS = [
    "ignore:'register_feature' is experimental:UserWarning",
    (
        "ignore:Failed to load dataset using "
        "`datasets.load_from_disk`:UserWarning"
    ),
    "ignore:pkg_resources is deprecated as an API:DeprecationWarning",
    (
        "ignore:Deprecated call to "
        "`pkg_resources.declare_namespace('google')`:DeprecationWarning"
    ),
    (
        "ignore:Deprecated call to "
        "`pkg_resources.declare_namespace('sphinxcontrib')`:"
        "DeprecationWarning"
    ),
    (
        "ignore:Deprecated call to "
        "`pkg_resources.declare_namespace('zope')`:DeprecationWarning"
    ),
    "ignore:Failed to find Vulkan ICD file:UserWarning",
    "ignore:Engine is deprecated:DeprecationWarning",
    "ignore:SapienRenderer:DeprecationWarning",
    "ignore:component.get_pose can be ambiguous thus deprecated:DeprecationWarning",  # noqa: E501
    "ignore:dep_util is Deprecated:DeprecationWarning",
    (
        "ignore:The ``compute`` method of metric BinaryAccuracy was "
        "called before the ``update`` method:UserWarning"
    ),
]


@dataclass(frozen=True)
class TestPhase:
    """One pytest phase in the unit-test pipeline."""

    name: str
    marker: str
    workers: int
    junit_xml: str
    html_report: str
    clean_allure: bool = False
    cov_append: bool = False


def parse_args() -> argparse.Namespace:
    tests_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run unit tests in multiple pytest phases and merge the "
            "coverage, JUnit XML, pytest-html, and Allure outputs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where reports and intermediate files are written.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=tests_dir / "test_robo_orchard_lab",
        help="Test root to pass to pytest.",
    )
    parser.add_argument(
        "--pytest-config",
        type=Path,
        default=tests_dir / "pytest.ini",
        help="Pytest configuration file.",
    )
    parser.add_argument(
        "--coverage-config",
        type=Path,
        default=tests_dir / ".coveragerc",
        help="Coverage configuration file.",
    )
    parser.add_argument(
        "--merge-script",
        type=Path,
        default=tests_dir / "merge_pytest_reports.py",
        help="Helper script used to merge XML and HTML reports.",
    )
    parser.add_argument(
        "--allure-dir",
        default="allure_unittest",
        help="Allure result directory under the output directory.",
    )
    parser.add_argument(
        "--coverage-file",
        default=".coverage.ut",
        help="Coverage data file name or path.",
    )
    parser.add_argument(
        "--final-junit-xml",
        default="unittest.xml",
        help="Final merged JUnit XML report filename.",
    )
    parser.add_argument(
        "--final-html",
        default="unittest.html",
        help="Final merged pytest-html report filename.",
    )
    parser.add_argument(
        "--non-sim-workers",
        type=int,
        default=6,
        help="Worker count for tests selected by 'not sim_env'.",
    )
    parser.add_argument(
        "--sim-workers",
        type=int,
        default=3,
        help="Worker count for tests selected by 'sim_env'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.test_root = args.test_root.resolve()
    args.pytest_config = args.pytest_config.resolve()
    args.coverage_config = args.coverage_config.resolve()
    args.merge_script = args.merge_script.resolve()

    return args


def build_phases(args: argparse.Namespace) -> list[TestPhase]:
    return [
        TestPhase(
            name="non_sim_env",
            marker="not sim_env",
            workers=args.non_sim_workers,
            junit_xml="unittest_non_sim_env.xml",
            html_report="unittest_non_sim_env.html",
            clean_allure=True,
        ),
        TestPhase(
            name="sim_env",
            marker="sim_env",
            workers=args.sim_workers,
            junit_xml="unittest_sim_env.xml",
            html_report="unittest_sim_env.html",
            cov_append=True,
        ),
    ]


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    print(f"$ {shlex.join(command)}")
    if dry_run:
        return

    subprocess.run(command, cwd=cwd, env=env, check=True)


def coverage_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    coverage_file = Path(args.coverage_file)
    if not coverage_file.is_absolute():
        coverage_file = args.output_dir / coverage_file
    env["COVERAGE_FILE"] = str(coverage_file)
    pythonwarnings = ",".join(PYTHON_WARNING_FILTERS)
    if env.get("PYTHONWARNINGS"):
        pythonwarnings = ",".join((pythonwarnings, env["PYTHONWARNINGS"]))
    env["PYTHONWARNINGS"] = pythonwarnings
    return env


def build_pytest_command(
    args: argparse.Namespace, phase: TestPhase
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(args.pytest_config),
        "-rf",
        "-x",
        "--dist",
        "loadgroup",
        "-n",
        str(phase.workers),
        "-m",
        phase.marker,
        str(args.test_root),
        f"--alluredir={args.allure_dir}",
        f"--cov-config={args.coverage_config}",
        "--cov",
        "--cov-report=",
        f"--junitxml={phase.junit_xml}",
        f"--html={phase.html_report}",
    ]
    if phase.clean_allure:
        command.append("--clean-alluredir")
    if phase.cov_append:
        command.append("--cov-append")
    return command


def run_pipeline(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = coverage_env(args)

    run_command(
        [sys.executable, "-m", "coverage", "erase"],
        cwd=args.output_dir,
        env=env,
        dry_run=args.dry_run,
    )

    phases = build_phases(args)
    for phase in phases:
        run_command(
            build_pytest_command(args, phase),
            cwd=args.output_dir,
            env=env,
            dry_run=args.dry_run,
        )

    for coverage_cmd in ("report", "xml", "html"):
        run_command(
            [
                sys.executable,
                "-m",
                "coverage",
                coverage_cmd,
                "--rcfile",
                str(args.coverage_config),
            ],
            cwd=args.output_dir,
            env=env,
            dry_run=args.dry_run,
        )

    merge_command = [
        sys.executable,
        str(args.merge_script),
        "--xml-out",
        args.final_junit_xml,
        "--html-out",
        args.final_html,
    ]
    for phase in phases:
        merge_command.extend(["--xml-report", phase.junit_xml])
    for phase in phases:
        merge_command.extend(["--html-report", phase.html_report])

    run_command(
        merge_command,
        cwd=args.output_dir,
        env=os.environ.copy(),
        dry_run=args.dry_run,
    )


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(args)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
