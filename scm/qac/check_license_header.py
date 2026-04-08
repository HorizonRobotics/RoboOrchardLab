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

import pathlib
import re
import sys

COPYRIGHT_YEARS_PATTERN = r"(?:20\d{2}|2024-20\d{2})"
_LICENSE_TEXT = [
    'Licensed under the Apache License, Version 2.0 (the "License");',
    "you may not use this file except in compliance with the License.",
    "You may obtain a copy of the License at",
    "",
    "      http://www.apache.org/licenses/LICENSE-2.0",
    "",
    ("Unless required by applicable law or agreed to in writing, software"),
    'distributed under the License is distributed on an "AS IS" BASIS,',
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or",
    "implied. See the License for the specific language governing",
    "permissions and limitations under the License.",
]

HASH_COMMENT_SUFFIXES = {".py", ".pyi"}
BLOCK_COMMENT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".h",
    ".hpp",
    ".proto",
}
CODING_RE = re.compile(r"#.*coding[:=][ \t]*[-\w.]+")
HASH_COPYRIGHT_LINE = (
    rf"# Copyright \(c\) {COPYRIGHT_YEARS_PATTERN} "
    r"Horizon Robotics\. All Rights Reserved\."
)
BLOCK_COPYRIGHT_LINE = (
    rf" \* Copyright \(c\) {COPYRIGHT_YEARS_PATTERN} "
    r"Horizon Robotics\. All Rights Reserved\."
)


def _build_hash_header_re() -> re.Pattern[str]:
    lines = [
        r"# Project RoboOrchard",
        r"#",
        HASH_COPYRIGHT_LINE,
        r"#",
    ]
    lines.extend(
        re.escape(f"# {line}" if line else "#") for line in _LICENSE_TEXT
    )
    return re.compile(r"\A" + r"\n".join(lines) + r"(?:\n|\Z)")


def _build_block_header_re() -> re.Pattern[str]:
    lines = [
        r"/\*",
        r" \* Project RoboOrchard",
        r" \*",
        BLOCK_COPYRIGHT_LINE,
        r" \*",
    ]
    lines.extend(
        re.escape(f" * {line}" if line else " *") for line in _LICENSE_TEXT
    )
    lines.append(r" \*/")
    return re.compile(r"\A" + r"\n".join(lines) + r"(?:\n|\Z)")


HASH_HEADER_RE = _build_hash_header_re()
BLOCK_HEADER_RE = _build_block_header_re()


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


def _strip_bom(text: str) -> str:
    return text[1:] if text.startswith("\ufeff") else text


def _strip_python_preamble(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start_idx = 0

    if lines and lines[0].startswith("#!"):
        start_idx = 1

    if start_idx < len(lines) and CODING_RE.fullmatch(
        lines[start_idx].rstrip("\n")
    ):
        start_idx += 1

    return "".join(lines[start_idx:])


def check_file_header(filepath: str) -> bool:
    """Check whether a file starts with a supported license header."""
    try:
        path = pathlib.Path(filepath)
        file_content = _strip_bom(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    suffix = path.suffix.lower()
    if suffix in HASH_COMMENT_SUFFIXES:
        return bool(HASH_HEADER_RE.match(_strip_python_preamble(file_content)))
    if suffix in BLOCK_COMMENT_SUFFIXES:
        return bool(BLOCK_HEADER_RE.match(file_content))
    return True


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
            error_message = (
                f"{colored_filepath}: Missing or incorrect license header."
                " Expected the standard RoboOrchard Apache 2.0 header at"
                " the top of the file."
            )
            print(
                error_message,
                file=sys.stderr,
            )
            final_exit_code = 1

    # Exit with a non-zero code if any file failed the check
    return final_exit_code


if __name__ == "__main__":
    sys.exit(main())
