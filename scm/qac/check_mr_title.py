# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import os
import re

rule = (
    r"^(feat|fix|bugfix|docs|style|refactor|perf|test|chore|scm)"
    r"\(.*\): [A-Z].*"
)

error_hint = """
Merge Request Title Validation Failed:

{mr_title}

The title does not comply with rule: {rule}

Required Format: <type>(<scope>): <description>

Examples:
feat(api): Implement new authentication module

Validation Checklist:
1. Valid type? (feat|fix|bugfix|docs|style|perf|refactor|test|chore|scm)
2. Description starts with capital letter
3. Colon has spaces on both sides
4. Contains non-ASCII characters
"""  # noqa


def main():
    branch = os.environ.get("gitlabTargetBranch")

    if branch != "master":
        return

    merge_request_title = os.environ["gitlabMergeRequestTitle"]

    if len(merge_request_title) > 180:
        raise ValueError(
            "The maximum length of merge request title is 180, but get {} for merge request title: {}".format(  # noqa
                len(merge_request_title), merge_request_title
            )
        )

    if not re.match(rule, merge_request_title):
        raise ValueError(
            error_hint.format(mr_title=merge_request_title, rule=rule)
        )


if __name__ == "__main__":
    if os.environ.get("gitlabActionType", "") == "MERGE":
        main()
