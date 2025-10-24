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
from typing import Iterable

import jinja2

__all__ = ["gen_index"]


def gen_index(
    jinja_template_path: str, gallery_dirs_dict: dict[str, Iterable[str]]
):
    template_loader = jinja2.FileSystemLoader(
        searchpath=os.path.dirname(jinja_template_path)
    )
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(os.path.basename(jinja_template_path))
    index_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "index.rst",
    )

    render_data = {"gallery_toctree": ""}

    for _, gallery_dirs in gallery_dirs_dict.items():
        gallery_toctree = "\n    ".join(
            os.path.join(gallery_dir, "index") for gallery_dir in gallery_dirs
        )

        render_data["gallery_toctree"] = (
            render_data["gallery_toctree"]
            + f"""

.. toctree::
    :maxdepth: 2
    :includehidden:

    {gallery_toctree}
"""
        )

    with open(index_file, "w") as f:
        f.write(template.render(render_data))
