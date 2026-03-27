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

"""Merge pytest JUnit XML reports and pytest-html reports."""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import pytest_html
from pytest_html import __version__ as pytest_html_version
from pytest_html.util import _process_css, _read_template

INT_ATTRS = ("tests", "failures", "errors", "skipped")
DEFAULT_OUTCOMES = {
    "failed": {"label": "Failed", "value": 0},
    "passed": {"label": "Passed", "value": 0},
    "skipped": {"label": "Skipped", "value": 0},
    "xfailed": {"label": "Expected failures", "value": 0},
    "xpassed": {"label": "Unexpected passes", "value": 0},
    "error": {"label": "Errors", "value": 0},
    "rerun": {"label": "Reruns", "value": 0},
}
DEFAULT_TABLE_HEADER = [
    '<th class="sortable" data-column-type="result">Result</th>',
    '<th class="sortable" data-column-type="testId">Test</th>',
    '<th class="sortable" data-column-type="duration">Duration</th>',
    "<th>Links</th>",
]


class _DataContainerParser(HTMLParser):
    """Extract the pytest-html JSON payload from a report."""

    def __init__(self) -> None:
        super().__init__()
        self.data_blob: str | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "div":
            return

        attr_map = dict(attrs)
        if attr_map.get("id") != "data-container":
            return

        self.data_blob = attr_map.get("data-jsonblob")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple pytest JUnit XML reports and pytest-html "
            "reports into a single XML and HTML report."
        )
    )
    parser.add_argument(
        "--xml-out",
        type=Path,
        required=True,
        help="Output path for the merged JUnit XML report.",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        required=True,
        help="Output path for the merged pytest-html report.",
    )
    parser.add_argument(
        "--xml-report",
        action="append",
        default=[],
        type=Path,
        help="Input pytest JUnit XML report file. Repeat per source report.",
    )
    parser.add_argument(
        "--html-report",
        action="append",
        default=[],
        type=Path,
        help="Input pytest-html report file. Repeat per source report.",
    )
    args = parser.parse_args()

    if not args.xml_report:
        parser.error("at least one --xml-report is required")
    if not args.html_report:
        parser.error("at least one --html-report is required")

    return args


def iter_suites(report_path: Path) -> list[ET.Element]:
    root = ET.parse(report_path).getroot()
    if root.tag == "testsuite":
        return [root]
    if root.tag == "testsuites":
        return list(root.findall("testsuite"))
    raise ValueError(
        f"Unsupported root tag {root.tag!r} in report {report_path}."
    )


def sum_attr(suites: Iterable[ET.Element], name: str) -> int:
    return sum(int(suite.attrib.get(name, "0")) for suite in suites)


def sum_time(suites: Iterable[ET.Element]) -> float:
    return sum(float(suite.attrib.get("time", "0")) for suite in suites)


def build_merged_xml_root(inputs: list[Path]) -> ET.Element:
    merged_root = ET.Element("testsuites")
    merged_suites: list[ET.Element] = []

    for report_path in inputs:
        suites = iter_suites(report_path)
        for suite in suites:
            suite_copy = deepcopy(suite)
            merged_root.append(suite_copy)
            merged_suites.append(suite_copy)

    for attr_name in INT_ATTRS:
        merged_root.set(attr_name, str(sum_attr(merged_suites, attr_name)))
    merged_root.set("time", f"{sum_time(merged_suites):.3f}")
    return merged_root


def write_xml(report_root: ET.Element, output_path: Path) -> None:
    tree = ET.ElementTree(report_root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def load_pytest_html_data(report_path: Path) -> dict[str, Any]:
    parser = _DataContainerParser()
    parser.feed(report_path.read_text(encoding="utf-8"))
    parser.close()

    if parser.data_blob is None:
        raise ValueError(
            f"Could not find pytest-html data payload in {report_path}."
        )

    return json.loads(parser.data_blob)


def merge_environment(
    merged_environment: dict[str, Any], report_environment: dict[str, Any]
) -> None:
    for key, value in report_environment.items():
        merged_environment.setdefault(key, value)


def merge_html_data(inputs: list[Path]) -> dict[str, Any]:
    merged_environment: dict[str, Any] = {}
    merged_tests: dict[str, list[dict[str, Any]]] = {}
    render_collapsed: list[str] | None = None
    initial_sort = "result"

    for index, report_path in enumerate(inputs):
        data = load_pytest_html_data(report_path)
        merge_environment(merged_environment, data.get("environment", {}))

        if render_collapsed is None:
            render_collapsed = list(data.get("renderCollapsed", ["passed"]))
        if index == 0:
            initial_sort = data.get("initialSort", "result")

        for nodeid, reports in data.get("tests", {}).items():
            merged_tests.setdefault(nodeid, [])
            merged_tests[nodeid].extend(deepcopy(reports))

    return {
        "environment": merged_environment,
        "tests": merged_tests,
        "renderCollapsed": render_collapsed or ["passed"],
        "initialSort": initial_sort,
    }


def merged_outcomes(report_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outcomes = deepcopy(DEFAULT_OUTCOMES)
    for reports in report_data["tests"].values():
        for report in reports:
            outcome = report.get("result", "").lower()
            if outcome in outcomes:
                outcomes[outcome]["value"] += 1
    return outcomes


def _format_duration(duration: float) -> str:
    if duration < 1:
        return "{} ms".format(round(duration * 1000))

    hours = math.floor(duration / 3600)
    remaining_seconds = duration % 3600
    minutes = math.floor(remaining_seconds / 60)
    remaining_seconds = remaining_seconds % 60
    seconds = round(remaining_seconds)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_run_count(
    outcomes: dict[str, dict[str, Any]], total_duration: float
) -> str:
    relevant_outcomes = ("passed", "failed", "xpassed", "xfailed")
    count = sum(outcomes[name]["value"] for name in relevant_outcomes)
    plural = count != 1
    duration = _format_duration(total_duration)
    return f"{count} {'tests' if plural else 'test'} took {duration}."


def write_pytest_html_report(
    report_data: dict[str, Any],
    output_path: Path,
    total_duration: float,
) -> None:
    resources_path = Path(pytest_html.__file__).resolve().parent / "resources"
    template = _read_template([resources_path])
    processed_css = _process_css(resources_path / "style.css", [])

    assets_path = output_path.parent / "assets"
    assets_path.mkdir(parents=True, exist_ok=True)
    (assets_path / "style.css").write_text(processed_css, encoding="utf-8")

    outcomes = merged_outcomes(report_data)
    generated = dt.datetime.now()
    rendered_report = template.render(
        title=output_path.name,
        date=generated.strftime("%d-%b-%Y"),
        time=generated.strftime("%H:%M:%S"),
        version=pytest_html_version,
        styles=str(Path("assets") / "style.css"),
        run_count=build_run_count(outcomes, total_duration),
        running_state="finished",
        self_contained=False,
        outcomes=outcomes,
        test_data=json.dumps(report_data),
        table_head=DEFAULT_TABLE_HEADER,
        additional_summary={"prefix": [], "summary": [], "postfix": []},
    )
    output_path.write_text(rendered_report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    merged_xml_root = build_merged_xml_root(args.xml_report)
    write_xml(merged_xml_root, args.xml_out)

    merged_html_data = merge_html_data(args.html_report)
    total_duration = float(merged_xml_root.attrib.get("time", "0"))
    write_pytest_html_report(merged_html_data, args.html_out, total_duration)


if __name__ == "__main__":
    main()
