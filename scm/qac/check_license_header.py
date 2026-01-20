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

import sys

LICENSE_HEADER_TEMPLATE = """# Project RoboOrchard
#
# Copyright (c) {} Horizon Robotics. All Rights Reserved.
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
"""

LICENSE_HEADERS = [
    LICENSE_HEADER_TEMPLATE.format(year)
    for year in ["2024", "2025", "2024-2025", "2024-2026"]
]


class Colors:
    """A simple class for ANSI color codes for terminal output."""

    # Check if the output stream supports ANSI codes
    if sys.stdout.isatty():
        RED = "\033[91m"
        ENDC = "\033[0m"  # Resets the color
    else:
        # If not a TTY, don't use color codes
        RED = ""
        ENDC = ""


def check_file_header(filepath) -> bool:
    """Checks if a single file starts with the specified license header."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
            return any(header in file_content for header in LICENSE_HEADERS)
    except Exception:
        return False


def main():
    """Iterates through the file list passed by pre-commit and checks their headers.

    Reports errors in the format 'filename: error message'.
    """  # noqa: E501

    # pre-commit passes file paths as command-line arguments
    files_to_check = sys.argv[1:]

    # Track the overall result. 0 for success, 1 for failure.
    final_exit_code = 0

    for filepath in files_to_check:
        if not check_file_header(filepath):
            colored_filepath = f"{Colors.RED}{filepath}{Colors.ENDC}"
            print(
                f"{colored_filepath}: Missing or incorrect license header.",
                file=sys.stderr,
            )
            final_exit_code = 1

    # Exit with a non-zero code if any file failed the check
    return final_exit_code


if __name__ == "__main__":
    sys.exit(main())
