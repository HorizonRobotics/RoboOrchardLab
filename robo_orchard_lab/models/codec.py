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

from abc import ABCMeta, abstractmethod

import torch


class CodecMixin(metaclass=ABCMeta):
    """CodecMixin is an API for encoding and decoding data.

    The codec concept is suitable for `Tokenizer`, `VAE`, and other models
    that transform data into a different representation and back.

    """

    @abstractmethod
    def encode(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Encode data."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def decode(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Decode data."""
        raise NotImplementedError("Subclasses should implement this method.")
