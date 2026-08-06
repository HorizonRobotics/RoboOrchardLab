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

from __future__ import annotations
import logging
from typing import overload

import torch
from accelerate import Accelerator
from pydantic import Field
from typing_extensions import deprecated

from robo_orchard_lab.pipeline.hooks.mixin import (
    ClassType,
    HookContext,
    ModelOutput,
    ModelOutputHasLossKeys,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = [
    "LossTracker",
    "LossTrackerConfig",
    "LossMovingAverageTrackerConfig",
    "LossMovingAverageTracker",
]


logger = logging.getLogger(__name__)


class LossTracker(PipelineHooks):
    """Aggregate model-output losses and log them on committed optimizer steps.

    The hook reads only ``PipelineHookArgs.model_outputs`` at ``on_step``
    after-hooks. For outputs implementing ``ModelOutputHasLossKeys``, the
    declared ``loss_keys()`` select tracked entries; otherwise keys containing
    ``"loss"`` are selected. The separate backward loss returned as the second
    value of ``SimpleStepProcessor.forward`` is not part of this contract and
    is never inspected by this hook.

    Selected losses are accumulated as detached local sum/count tensors. One
    packed distributed reduction runs only when a committed optimizer step
    reaches the logging boundary. Logging and reset happen from
    ``on_optimizer_step`` after-hooks once the trainer knows whether the
    optimizer boundary committed.

    Each logging window establishes its complete loss-key set on the first
    update. Every later update in that window must provide exactly the same
    set; key presence is not synchronized across ranks, so callers must keep
    the set identical on every rank. The former two-argument ``update`` call
    shape remains accepted for source compatibility, but the accelerator is
    ignored and the retired mutable cache fields are not part of the
    supported contract. A custom ``update`` override must delegate to
    ``super().update`` to contribute to this canonical aggregation; the
    tracker does not reconstruct arbitrary override-owned state.
    """

    cfg: LossTrackerConfig

    def __init__(self, cfg: LossTrackerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._reset_window()

        self.register_hook(
            "on_step", HookContext.from_callable(after=self._on_step_end)
        )
        self.register_hook(
            "on_optimizer_step",
            HookContext.from_callable(after=self._on_optimizer_step_end),
        )

    def _reset_window(self) -> None:
        """Clear canonical aggregation state for the current logging window."""

        self._cached_loss_sums: dict[str, torch.Tensor] = {}
        self._cached_loss_counts: dict[str, int] = {}
        self._expected_loss_keys: frozenset[str] | None = None

    def reset_cached_loss(self) -> None:
        """Reset the current loss logging aggregation window.

        This public name remains for callers that explicitly reset a tracker.
        Internal lifecycle paths use the non-overridable canonical reset so a
        legacy subclass cannot leave aggregation state partially initialized.
        """

        self._reset_window()

    @overload
    def update(self, model_outputs: ModelOutput, /) -> None: ...

    @overload
    def update(
        self,
        accelerator: Accelerator,
        model_outputs: ModelOutput,
    ) -> None: ...

    @overload
    def update(self, *, model_outputs: ModelOutput) -> None: ...

    def update(
        self,
        accelerator: Accelerator | ModelOutput | None = None,
        model_outputs: ModelOutput | None = None,
    ) -> None:
        """Accumulate loss entries selected from one model output.

        ``accelerator`` is accepted only for compatibility with the former
        ``update(accelerator, model_outputs)`` public API. It is ignored:
        distributed reduction belongs to the committed optimizer-step hook,
        after all local micro-step losses have been accumulated.

        Args:
            accelerator: Retired compatibility argument. When
                ``model_outputs`` is omitted, the sole positional argument is
                interpreted as the canonical model output instead.
            model_outputs (ModelOutput): Model outputs containing scalar loss
                entries or entries that can be reduced to a scalar mean. This
                value does not include the processor's separate backward-loss
                return value. Non-0-D tensors are reduced with ``mean()``.
                Every update in one logging window must expose the same
                complete loss-key set.

        Raises:
            TypeError: If no model outputs are provided.
            ValueError: If a logging window changes its loss-key set or
                publishes duplicate loss keys.
        """

        if model_outputs is None:
            if not isinstance(accelerator, ModelOutput):
                raise TypeError("LossTracker.update requires model_outputs")
            model_outputs = accelerator

        # find loss keys
        if isinstance(model_outputs, ModelOutputHasLossKeys):
            loss_keys = tuple(model_outputs.loss_keys())
        else:
            loss_keys = tuple(k for k in model_outputs.keys() if "loss" in k)

        if any(not isinstance(k, str) for k in loss_keys):
            self._reset_window()
            raise TypeError("LossTracker loss keys must be strings")
        if len(loss_keys) != len(set(loss_keys)):
            self._reset_window()
            raise ValueError(
                "LossTracker model outputs must not publish duplicate "
                "loss keys"
            )
        loss_key_set = frozenset(loss_keys)
        if self._expected_loss_keys is None:
            self._expected_loss_keys = loss_key_set
        elif loss_key_set != self._expected_loss_keys:
            expected = sorted(self._expected_loss_keys)
            actual = sorted(loss_key_set)
            self._reset_window()
            raise ValueError(
                "LossTracker loss keys must remain complete and stable within "
                f"a logging window: expected {expected}, got {actual}"
            )

        try:
            for k in loss_keys:
                loss = model_outputs[k]
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(f"Loss {k} is not a tensor.")

                local_loss = loss.detach()
                if local_loss.ndim != 0:
                    logger.warning(
                        f"The loss {k} is not a scalar, "
                        f"got shape {local_loss.shape}. "
                        "Using mean value as the loss."
                    )
                    local_loss = local_loss.mean()
                local_loss = local_loss.float()

                if self.cfg.moving_average and k in self._cached_loss_sums:
                    self._cached_loss_sums[k].add_(local_loss)
                    self._cached_loss_counts[k] += 1
                else:
                    self._cached_loss_sums[k] = local_loss.new_zeros(()).add_(
                        local_loss
                    )
                    self._cached_loss_counts[k] = 1
        except Exception:
            self._reset_window()
            raise

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        """Collect loss values at the end of each micro step.

        Loss logging is finalized by ``_on_optimizer_step_end`` after the
        trainer knows whether the optimizer boundary committed.

        Args:
            args (PipelineHookArgs): Arguments containing the current
                micro-step model outputs and training progress.
        """
        if args.exception is not None:
            self._reset_window()
            return

        if args.model_outputs is not None:
            # Keep the legacy two-argument call shape during migration so
            # existing LossTracker subclasses overriding update() continue
            # to receive the accelerator argument.
            self.update(args.accelerator, args.model_outputs)

    def _on_optimizer_step_end(self, args: PipelineHookArgs) -> None:
        """Log and reset losses after a committed optimizer step."""
        if args.exception is not None:
            self._reset_window()
            return

        if not args.is_optimizer_step_committed:
            self._reset_window()
            return

        if args.global_step_id % self.cfg.step_log_freq == 0:
            try:
                loss_keys = sorted(self._cached_loss_sums)
                reduced_losses: dict[str, torch.Tensor] = {}
                if loss_keys:
                    local_loss_stats = torch.stack(
                        [
                            torch.stack(
                                [
                                    self._cached_loss_sums[k],
                                    self._cached_loss_sums[k].new_tensor(
                                        self._cached_loss_counts[k]
                                    ),
                                ]
                            )
                            for k in loss_keys
                        ]
                    )
                    global_loss_stats = args.accelerator.reduce(
                        local_loss_stats,
                        reduction="sum",
                    )
                    reduced_losses = {
                        k: global_loss_stats[idx, 0]
                        / global_loss_stats[idx, 1].clamp_min(1)
                        for idx, k in enumerate(loss_keys)
                    }

                # only log in main process
                if args.accelerator.is_main_process:
                    msg = "Epoch[{}/{}] Step[{}] GlobalStep[{}/{}]: ".format(
                        args.epoch_id,
                        args.max_epoch - 1
                        if args.max_epoch is not None
                        else "NA",
                        args.step_id,
                        args.global_step_id,
                        args.max_step if args.max_step is not None else "NA",
                    )
                    total_loss = 0.0
                    for k in loss_keys:
                        v = reduced_losses[k].item()
                        msg += f"{k}[{v:.4f}]\t"
                        total_loss += v
                        args.accelerator.log(
                            {f"Loss/{k}": v},
                            step=args.global_step_id,
                        )

                    if self.cfg.log_total_loss:
                        msg += f"total_loss[{total_loss:.4f}]"
                        args.accelerator.log(
                            {"Loss/Total_Loss": total_loss},
                            step=args.global_step_id,
                        )

                    logger.info(msg)
            finally:
                self._reset_window()


class LossTrackerConfig(PipelineHooksConfig[LossTracker]):
    """Configuration for committed optimizer-step model-output loss logging."""

    class_type: ClassType[LossTracker] = LossTracker

    step_log_freq: int = Field(ge=1, default=5)
    """The frequency in committed optimizer steps to log the loss."""

    log_total_loss: bool = False
    """If True, log the total loss as well."""

    moving_average: bool = True
    """How to retain selected model-output losses in the logging window.

    If True, accumulate a sum/count across all updates in the window. If
    False, retain only the latest value for each key with a count of one. The
    window size is controlled by ``step_log_freq`` committed optimizer steps.
    """


@deprecated(
    "LossMovingAverageTrackerConfig is deprecated, "
    "please use LossTrackerConfig instead."
)
class LossMovingAverageTrackerConfig(LossTrackerConfig):
    step_log_freq: int = 25
    log_total_loss: bool = True


@deprecated(
    "LossMovingAverageTracker is deprecated, please use LossTracker instead."
)
class LossMovingAverageTracker(LossTracker):
    def __init__(self, cfg: LossMovingAverageTrackerConfig):
        super().__init__(cfg)
