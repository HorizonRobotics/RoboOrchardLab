# License Header Guideline

Use the standard RoboOrchard Apache 2.0 file header for every newly created
repository-owned source file unless the file lives under an excluded
path or the task explicitly targets external or third-party code.

Canonical templates:

Line-comment form for Python-style files:

```text
# Project RoboOrchard
#
# Copyright (c) <copyright-years> Horizon Robotics. All Rights Reserved.
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
```

Block-comment form for C/C++/CUDA/proto-style files:

```text
/*
 * Project RoboOrchard
 *
 * Copyright (c) <copyright-years> Horizon Robotics. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
```

Year rendering:
- Treat `<copyright-years>` as a placeholder in guidance only. Source files
  must contain a concrete year string, not the placeholder itself.
- For this repository, the local editor template in `.vscode/settings.json`
  currently renders the copyright line as `2024-<<year>>` for configured
  source languages.
- Use `2024-<<year>>` as the default template form for new headers, which then
  renders to a concrete year range such as `2024-2026` in source files.

Placement:
- For Python-style files, allow a shebang or encoding declaration before the
  header only when required; otherwise place the header first.
- For other source files, place the license header at the very top of the file.
- Keep the comment style aligned with the target language instead of reusing
  the Python form verbatim.
- Keep the wording exact; do not shorten, paraphrase, or reflow the license
  text.

Validation:
- If `scm/qac/check_license_header.py` exists for the target repository and is
  wired through `.pre-commit-config.yaml`, the final rendered year string must
  match the forms it accepts.
- If the accepted year forms drift from the editor template or current year,
  update the checker and template together instead of hard-coding another year
  into agent instructions.