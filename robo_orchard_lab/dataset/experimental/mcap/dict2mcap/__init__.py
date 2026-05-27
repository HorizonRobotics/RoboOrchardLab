# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

"""Dict2Mcap public API."""

from robo_orchard_lab.dataset.experimental.mcap.dict2mcap.converters import (
    MessageDict,
    ToMcapMessageConverter,
    ToMcapMessageFactory,
    align_converted_topic_map,
)
from robo_orchard_lab.dataset.experimental.mcap.dict2mcap.default_converters import (  # noqa: E501
    create_default_to_mcap_message_factory,
)
from robo_orchard_lab.dataset.experimental.mcap.dict2mcap.topic_map import (
    TopicStampedMessage,
    flatten_input_order_records,
    write_records,
)
from robo_orchard_lab.dataset.experimental.mcap.dict2mcap.writer import (
    Dict2Mcap,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage

__all__ = [
    "Dict2Mcap",
    "MessageDict",
    "StampedMessage",
    "ToMcapMessageConverter",
    "ToMcapMessageFactory",
    "TopicStampedMessage",
    "align_converted_topic_map",
    "create_default_to_mcap_message_factory",
    "flatten_input_order_records",
    "write_records",
]
