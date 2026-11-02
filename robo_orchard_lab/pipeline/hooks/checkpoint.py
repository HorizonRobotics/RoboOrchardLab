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
import os
import shutil
from typing import Any, Optional, TypeVar

from accelerate import Accelerator
from accelerate.utils import gather_object

from robo_orchard_lab.models.torch_model import TorchModelMixin
from robo_orchard_lab.pipeline.hooks.mixin import (
    ClassType,
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)
from robo_orchard_lab.utils.accelerate import (
    _capture_prepared_loader_recovery,
)
from robo_orchard_lab.utils.huggingface import (
    _checkpoint_capability_reason,
    _has_model_save_artifact,
    _should_write_like_accelerate_save,
    _supports_checkpoint_backend,
)

__all__ = [
    "SaveCheckpoint",
    "SaveCheckpointConfig",
    "SaveModel",
    "SaveModelConfig",
]


SaveHookType = TypeVar("SaveHookType", bound=PipelineHooks)

logger = logging.getLogger(__name__)


def _export_checkpoint_models(
    accelerator: Accelerator, directory: str
) -> None:
    """Export models and their configs alongside the native checkpoint."""
    should_write = _should_write_like_accelerate_save(accelerator)
    for idx, model in enumerate(accelerator._models):
        prefix = "model" if len(accelerator._models) == 1 else f"model_{idx}"
        target = os.path.join(directory, prefix)
        accelerator.save_model(model, target)
        config = os.path.join(directory, f"{prefix}.config.json")
        if (
            os.path.exists(config)
            and should_write
            and _has_model_save_artifact(target)
        ):
            shutil.copy(config, target)


class SaveHookConfigT(PipelineHooksConfig[SaveHookType]):
    save_root: str | None = None
    """The root directory where checkpoints are saved."""
    save_epoch_freq: Optional[int] = 1
    """Frequency of saving checkpoints based on epochs."""
    save_step_freq: Optional[int] = None
    """Checkpoint frequency in completed accumulation windows."""
    save_when_loop_end: bool = True
    """Whether to save the model at the end of the training loop."""
    save_model: bool = True
    """Whether to save the model as well."""

    def __post_init__(self) -> None:
        if self.save_epoch_freq is None and self.save_step_freq is None:
            raise ValueError(
                "Either `save_epoch_freq` or `save_step_freq` must be "
                "specified. Both cannot be None."
            )
        if self.save_epoch_freq is not None and self.save_epoch_freq < 1:
            raise ValueError(
                "save_epoch_freq = {} < 1 is not allowed".format(
                    self.save_epoch_freq
                )
            )
        if self.save_step_freq is not None and self.save_step_freq < 1:
            raise ValueError(
                "save_step_freq = {} < 1 is not allowed".format(
                    self.save_step_freq
                )
            )


class SaveCheckpoint(PipelineHooks):
    """Save Accelerate training state at configured training boundaries.

    Step-based saves run after completed accumulation windows. Epoch and
    loop saves use their own scopes and independent frequency triggers.

    Args:
        cfg (SaveCheckpointConfig): Configuration object containing
            parameters for checkpointing.
    """

    def __init__(
        self,
        cfg: SaveCheckpointConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_root = cfg.save_root
        self.save_epoch_freq = cfg.save_epoch_freq
        self.save_step_freq = cfg.save_step_freq
        self._is_checked = False

        self.register_hook(
            channel="on_optimizer_step",
            hook=HookContext.from_callable(after=self._on_step_end),
        )
        self.register_hook(
            channel="on_epoch",
            hook=HookContext.from_callable(after=self._on_epoch_end),
        )
        self.register_hook(
            channel="on_loop",
            hook=HookContext.from_callable(after=self._on_loop_end),
        )

    def _check(self, accelerator: Accelerator) -> None:
        if self._is_checked:
            return
        if (
            not accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is None
        ):
            raise ValueError(
                "save_root is required when `automatic_checkpoint_naming of Accelerator is False"  # noqa: E501
            )
        if (
            accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is not None
            and accelerator.is_main_process
        ):
            logger.warning(
                "Ignore customize save_root = {} since `automatic_checkpoint_naming` is True for Accelerator".format(  # noqa: E501
                    self.save_root
                )
            )
        self._is_checked = True

    def _save_state(
        self, accelerator: Accelerator, data_loader: Any | None = None
    ) -> str:
        """Gather recovery state and delegate best-effort saving to Accelerate.

        Checkpoint behavior is undefined after training, hook or save errors.
        The rank gather below carries normal recovery state and capability; it
        does not coordinate errors or make the filesystem save atomic.
        A prepared loader supplies its data state; standalone calls without
        one save the remaining checkpoint state without data recovery details.
        Recovery capability is evaluated for each save, so a temporary
        limitation does not permanently disable data-state capture.
        """
        from robo_orchard_lab.pipeline.training._progress import (
            TrainerProgressState,
        )

        self._check(accelerator)
        owners = [
            obj
            for obj in getattr(accelerator, "_custom_objects", ())
            if isinstance(obj, TrainerProgressState)
        ]
        if any(
            progress.micro_step.index_in_optimizer_step != 0
            for progress in owners
        ):
            raise RuntimeError("Cannot checkpoint an unfinished window.")
        if owners:
            supports_recovery = _supports_checkpoint_backend(accelerator)
            if supports_recovery and len(owners) != 1:
                raise RuntimeError(
                    "Trainer/data recovery requires one trainer "
                    "progress owner."
                )
            reason = _checkpoint_capability_reason(accelerator)
            outcomes = []
            if supports_recovery:
                progress = owners[0]
                if (
                    reason is None
                    and getattr(data_loader, "_lab_recovery", None) is None
                ):
                    reason = "prepared DataLoader is unavailable"
                local_state = None
                if reason is None:
                    if (
                        accelerator.step
                        % accelerator.gradient_accumulation_steps
                    ):
                        raise RuntimeError(
                            "Checkpoint accumulation phase is unsafe."
                        )
                    data_state = _capture_prepared_loader_recovery(
                        data_loader,
                        epoch=progress.epoch_id,
                        processed_batches=progress.micro_step.epoch_step_id,
                    )
                    local_state = {
                        "rank": accelerator.process_index,
                        **data_state,
                    }
                outcomes = gather_object(
                    [{"reason": reason, "state": local_state}]
                )
                reason = next(
                    (
                        item["reason"]
                        for item in outcomes
                        if item["reason"] is not None
                    ),
                    None,
                )
            for progress in owners:
                progress.rank_states = (
                    [item["state"] for item in outcomes]
                    if reason is None
                    else []
                )
            if reason is not None:
                logger.warning(
                    "Saving native Accelerate state without trainer/data "
                    "recovery: %s",
                    reason,
                )

        save_location = accelerator.save_state(self.save_root)
        if self.cfg.save_model:
            _export_checkpoint_models(accelerator, save_location)
        return save_location

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        """Callback when a completed-window scope ends.

        Saves a checkpoint at the end of a completed accumulation window if
        `save_step_freq` conditions are met.


        Args:
            args (PipelineHookArgs): Arguments containing the
                ``global_step_id`` and ``accelerator`` used for saving state.

        Logs:
            A message indicating the checkpoint location.
        """
        if args.exception is not None:
            return
        if not args.is_optimizer_step_boundary:
            return

        if (
            self.save_step_freq is not None
            and args.global_step_id % self.save_step_freq == 0
        ):
            save_location = self._save_state(
                args.accelerator, args._checkpoint_dataloader
            )
            if args.accelerator.is_main_process:
                logger.info(
                    "Save checkpoint at the end of step {} to {}".format(
                        args.global_step_id, save_location
                    )
                )

    @staticmethod
    def _skip_unfinished_window(args: PipelineHookArgs) -> bool:
        """Skip non-step saves while an accumulation window is unfinished."""
        if (
            args.micro_step is not None
            and args.micro_step.index_in_optimizer_step != 0
        ):
            logger.warning(
                "Skipping checkpoint: gradient accumulation is unfinished."
            )
            return True
        return False

    def _on_epoch_end(self, args: PipelineHookArgs) -> None:
        """Callback when epoch ends.

        Saves a checkpoint at the end of an epoch if `save_epoch_freq`
        conditions are met.

        Args:
            args (PipelineHookArgs): Arguments containing the ``epoch_id`` and
                ``accelerator`` used for saving state.

        Logs:
            A message indicating the checkpoint location.
        """
        if args.exception is not None:
            return

        if self._epoch_checkpoint_due(args.epoch_id) and not (
            self._skip_unfinished_window(args)
        ):
            defer = args._defer_epoch_checkpoint
            if defer is not None:
                defer(lambda: self._save_epoch_checkpoint(args))
            else:
                self._save_epoch_checkpoint(args)

    def _epoch_checkpoint_due(self, epoch: int) -> bool:
        return self.save_epoch_freq is not None and (
            (epoch + 1) % self.save_epoch_freq == 0
        )

    def _save_epoch_checkpoint(self, args: PipelineHookArgs) -> None:
        """Save at the trainer's epoch boundary, or immediately standalone."""
        save_location = self._save_state(
            args.accelerator, args._checkpoint_dataloader
        )
        if args.accelerator.is_main_process:
            logger.info(
                "Save checkpoint at the end of epoch {} to {}".format(
                    args.epoch_id, save_location
                )
            )

    def _on_loop_end(self, args: PipelineHookArgs) -> None:
        if (
            args.exception is not None
            or not self.cfg.save_when_loop_end
            or self._skip_unfinished_window(args)
        ):
            return

        save_location = self._save_state(
            args.accelerator, args._checkpoint_dataloader
        )
        if args.accelerator.is_main_process:
            logger.info(
                "Save checkpoint at the end of training to {}".format(
                    save_location
                )
            )


class SaveCheckpointConfig(SaveHookConfigT[SaveCheckpoint]):
    """Configuration class for SaveCheckpoint."""

    class_type: ClassType[SaveCheckpoint] = SaveCheckpoint


class SaveModel(PipelineHooks):
    """Save model weights at configured epoch or completed-window boundaries.

    This hook writes model artifacts via ``Accelerator.save_model``. It tracks
    the last saved global step to avoid duplicate saves when a step, epoch, and
    loop-end event refer to the same completed accumulation window.
    """

    cfg: SaveModelConfig

    def __init__(
        self,
        cfg: SaveModelConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_hook(
            channel="on_optimizer_step",
            hook=HookContext.from_callable(after=self._on_step_end),
        )
        self.register_hook(
            channel="on_epoch",
            hook=HookContext.from_callable(after=self._on_epoch_end),
        )

        self.register_hook(
            channel="on_loop",
            hook=HookContext.from_callable(after=self._on_loop_end),
        )
        # use to avoid duplicate saving at epoch end or loop end
        self._saved_last_step: int = -1

    def _save_model(
        self, args: PipelineHookArgs, step_id: int, epoch_id: int
    ) -> str:
        """Saves the model state using the provided accelerator.

        Args:
            args (PipelineHookArgs): The arguments containing the
                accelerator and other relevant information.

        Raises:
            ValueError: If the save root is not specified when
                automatic checkpoint naming is disabled.
        """
        if args.model is None:
            raise ValueError("model is None. Cannot save model.")
        self._saved_last_step = args.global_step_id
        accelerator = args.accelerator
        save_root = self.cfg.get_save_root(accelerator)
        save_root = os.path.join(
            save_root, f"model_epoch_{epoch_id}_step_{step_id}"
        )
        should_write_config = _should_write_like_accelerate_save(accelerator)
        for idx, model in enumerate(accelerator._models):
            save_directory = os.path.join(save_root, f"{idx}")
            accelerator.save_model(model, save_directory)
            if (
                should_write_config
                and isinstance(model, TorchModelMixin)
                and _has_model_save_artifact(save_directory)
            ):
                model_config_path = os.path.join(
                    save_directory, "model.config.json"
                )
                with open(model_config_path, "w") as f:
                    f.write(model.cfg.model_dump_json(indent=4))

        return save_root

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        """Save the model after matching completed accumulation windows."""
        if args.exception is not None:
            return
        if not args.is_optimizer_step_boundary:
            return

        if (
            self.cfg.save_step_freq is not None
            and args.global_step_id % self.cfg.save_step_freq == 0
            and self._saved_last_step != args.global_step_id
        ):
            args.accelerator.wait_for_everyone()
            save_location = self._save_model(
                args, step_id=args.step_id, epoch_id=args.epoch_id
            )
            args.accelerator.wait_for_everyone()
            if args.accelerator.is_main_process:
                logger.info(
                    "Save model at the end of step {} to {}".format(
                        args.global_step_id, save_location
                    )
                )

    def _on_epoch_end(self, args: PipelineHookArgs) -> None:
        """Save the model after matching epoch scopes."""
        if args.exception is not None:
            return

        if (
            self.cfg.save_epoch_freq is not None
            and (args.epoch_id + 1) % self.cfg.save_epoch_freq == 0
            and self._saved_last_step != args.global_step_id
        ):
            args.accelerator.wait_for_everyone()
            save_location = self._save_model(
                args, step_id=args.step_id, epoch_id=args.epoch_id
            )
            args.accelerator.wait_for_everyone()
            if args.accelerator.is_main_process:
                logger.info(
                    "Save model at the end of epoch {} of global step {} to {}".format(  # noqa: E501
                        args.epoch_id, args.global_step_id, save_location
                    )
                )

    def _on_loop_end(self, args: PipelineHookArgs) -> None:
        """Saves the model at the end of the training loop.

        This method ensures that the model is saved at the end of training,
        even if the last epoch or step did not meet the saving criteria
        defined by `save_epoch_freq` or `save_step_freq`.

        Args:
            args (PipelineHookArgs): The arguments containing the
                accelerator and other relevant information.

        Logs:
            A message indicating the checkpoint location.
        """
        if args.exception is not None:
            return

        if (
            self.cfg.save_when_loop_end is False
            or self._saved_last_step == args.global_step_id
        ):
            return

        args.accelerator.wait_for_everyone()

        save_location = self._save_model(
            args, step_id=args.step_id, epoch_id=args.epoch_id
        )
        args.accelerator.wait_for_everyone()
        if args.accelerator.is_main_process:
            logger.info(
                "Save model at the end of training global step {} to {}".format(  # noqa: E501
                    args.global_step_id, save_location
                )
            )


class SaveModelConfig(SaveHookConfigT[SaveModel]):
    """Configuration for model-only artifact saves."""

    class_type: ClassType[SaveModel] = SaveModel

    def get_save_root(self, accelerator: Accelerator) -> str:
        """Return the root directory for model artifacts.

        If `automatic_checkpoint_naming` is enabled in the Accelerator's
        project configuration, the save root is set to a default directory
        within the project's directory. Otherwise, it uses the provided
        `save_root` value.
        """
        if (
            not accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is None
        ):
            raise ValueError(
                "save_root is required when `automatic_checkpoint_naming of Accelerator is False"  # noqa: E501
            )
        if (
            accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is not None
            and accelerator.is_main_process
        ):
            logger.warning(
                "Ignore customize save_root = {} since `automatic_checkpoint_naming` is True for Accelerator".format(  # noqa: E501
                    self.save_root
                )
            )
        if accelerator.project_configuration.automatic_checkpoint_naming:
            return os.path.join(accelerator.project_dir, "saved_models")
        else:
            if self.save_root is None:
                raise ValueError(
                    "save_root must be set when automatic naming is off."
                )
            return self.save_root
