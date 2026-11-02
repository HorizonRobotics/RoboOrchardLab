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

import warnings
from collections.abc import (
    Callable,
    Iterable as IterableABC,
)
from contextlib import nullcontext
from typing import Any, Iterable

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from robo_orchard_core.utils.config import Config
from torch.utils.data import DataLoader

from robo_orchard_lab.dataset.robot._prefetch import (
    _close_dataloader_owner_resources,
)
from robo_orchard_lab.models.torch_model import TorchModelMixin
from robo_orchard_lab.pipeline.hooks.grad_clip import (
    GradientClippingHook,
    GradientClippingHookConfig,
    _clip_optimizer_gradients,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    MicroStepProgressState,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)
from robo_orchard_lab.pipeline.hooks.validation import ValidationHookConfig
from robo_orchard_lab.pipeline.training._deepspeed import (
    DeepSpeedStepSnapshot,
    configure_deepspeed_gradient_clipping,
    is_deepspeed_accelerator,
    observe_deepspeed_optimizer_step,
    prepare_deepspeed_optimizer_scheduler,
    read_deepspeed_step_snapshot,
)
from robo_orchard_lab.pipeline.training._progress import (
    TrainerProgressState,
)
from robo_orchard_lab.processing.step_processor import BatchStepProcessorMixin
from robo_orchard_lab.utils.accelerate import (
    PreparedDataLoaderIteration,
    _configure_prepared_loader_recovery,
    _load_prepared_loader_recovery,
    _set_prepared_loader_epoch,
    _warn_if_prepare_falls_back_to_slow_path,
    configure_data_loader_for_accelerate,
    iterate_prepared_loader,
)
from robo_orchard_lab.utils.huggingface import (
    _checkpoint_capability_reason,
    accelerator_load_state,
    get_accelerate_project_last_checkpoint_id,
)

logger = get_logger(__name__)


__all__ = [
    "HookBasedTrainer",
    "LRSchedulerFactory",
    "ResumeCheckpointConfig",
    "GradientClippingHookConfig",
    "ValidationHookConfig",
    "PipelineHookOrConfigType",
]


PipelineHookOrConfigType = PipelineHooksConfig | PipelineHooks
LRSchedulerFactory = Callable[
    [torch.optim.Optimizer],
    torch.optim.lr_scheduler.LRScheduler,
]


class ResumeCheckpointConfig(Config):
    """Load checkpoints with Lab training policy and continued save naming.

    Registered trainer progress enables recovery capability checks and a new
    accumulation window after loading. Call ``load_state`` without trainer
    progress for native loading and continued automatic save naming only.
    """

    resume_from: str
    """The directory containing the checkpoints."""
    cache_dir: str | None = None
    """The directory to cache the checkpoints if from a remote path."""

    safe_serialization: bool = True
    """Retained for existing callers; Accelerate detects the saved format."""

    def load_state(self, accelerator: Accelerator, **kwargs) -> None:
        """Restore native state, then normalize a registered trainer's phase.

        Capability checks run before loading when ``TrainerProgressState`` is
        registered. The accumulation phase is reset only after the complete
        native load succeeds, including custom objects and global RNG. The
        trainer applies prepared DataLoader recovery separately afterward.

        Args:
            accelerator: Accelerator containing the objects to restore.
            **kwargs: Forwarded to the native checkpoint loader.

        Raises:
            RuntimeError: Registered trainer/data recovery is unsupported by
                the current runtime configuration.
        """
        has_trainer_progress = any(
            isinstance(obj, TrainerProgressState)
            for obj in getattr(accelerator, "_custom_objects", ())
        )
        if has_trainer_progress:
            reason = _checkpoint_capability_reason(accelerator)
            if reason is not None:
                raise RuntimeError(reason)
        accelerator_load_state(
            accelerator=accelerator,
            input_dir=self.resume_from,
            cache_dir=self.cache_dir,
            safe_serialization=self.safe_serialization,
            **kwargs,
        )
        if has_trainer_progress:
            # A fresh phase also permits changing the accumulation size.
            accelerator.step = 0
        project = accelerator.project_configuration
        if project.automatic_checkpoint_naming and accelerator.project_dir:
            project.iteration = max(
                project.iteration,
                get_accelerate_project_last_checkpoint_id(
                    accelerator.project_dir
                )
                + 1,
            )


def _normalize_hook_inputs(
    hooks: (
        PipelineHookOrConfigType | Iterable[PipelineHookOrConfigType] | None
    ),
) -> list[PipelineHookOrConfigType]:
    """Return raw hook inputs without instantiating hook configs."""

    if hooks is None:
        return []
    if isinstance(hooks, (PipelineHooksConfig, PipelineHooks)):
        return [hooks]
    if not isinstance(hooks, IterableABC):
        raise TypeError(
            "Expected PipelineHooks, PipelineHooksConfig, an iterable of "
            "hooks/configs, or None."
        )
    return list(hooks)


def _split_gradient_clipping_config(
    *,
    hooks: (
        PipelineHookOrConfigType | Iterable[PipelineHookOrConfigType] | None
    ),
    grad_clip: GradientClippingHookConfig | None,
    is_deepspeed: bool,
) -> tuple[list[PipelineHookOrConfigType], GradientClippingHookConfig | None]:
    """Extract trainer-owned gradient clipping config from raw hooks.

    ``GradientClippingHookConfig`` can arrive either through the trainer's
    ``grad_clip`` argument or directly in the raw ``hooks`` input. It must be
    consumed before ``PipelineHooks.from_hooks(...)`` instantiates configs so
    DeepSpeed can receive equivalent clipping config before
    ``accelerator.prepare(...)``.
    """

    remaining_hooks: list[PipelineHookOrConfigType] = []
    effective_grad_clip = grad_clip
    saw_direct_grad_clip_hook = False
    for hook_or_config in _normalize_hook_inputs(hooks):
        if isinstance(hook_or_config, GradientClippingHookConfig):
            if saw_direct_grad_clip_hook:
                raise ValueError(
                    "Do not combine trainer-owned `grad_clip` or "
                    "GradientClippingHookConfig with an already constructed "
                    "GradientClippingHook."
                )
            if effective_grad_clip is not None:
                raise ValueError(
                    "Only one gradient clipping configuration can be "
                    "provided. Use either `grad_clip` or one "
                    "GradientClippingHookConfig in `hooks`, not both."
                )
            effective_grad_clip = hook_or_config
            continue

        if isinstance(hook_or_config, GradientClippingHook):
            if saw_direct_grad_clip_hook:
                raise ValueError(
                    "Only one gradient clipping hook can be provided. "
                    "Use `grad_clip` or one already constructed "
                    "GradientClippingHook, not multiple clipping hooks."
                )
            saw_direct_grad_clip_hook = True
            if is_deepspeed:
                raise ValueError(
                    "DeepSpeed gradient clipping must be configured from "
                    "GradientClippingHookConfig before "
                    "accelerator.prepare(...). "
                    "Do not pass an already constructed GradientClippingHook "
                    "when DeepSpeed is active."
                )
            if effective_grad_clip is not None:
                raise ValueError(
                    "Do not combine trainer-owned `grad_clip` or "
                    "GradientClippingHookConfig with an already constructed "
                    "GradientClippingHook."
                )

        remaining_hooks.append(hook_or_config)
    return remaining_hooks, effective_grad_clip


def _resolve_optimizer_scheduler_before_prepare(
    *,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer | DummyOptim,
    lr_scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | DummyScheduler
        | LRSchedulerFactory
    ),
) -> tuple[
    torch.optim.Optimizer | DummyOptim,
    torch.optim.lr_scheduler.LRScheduler | DummyScheduler,
]:
    """Normalize optimizer and scheduler inputs before one prepare call.

    Ordinary backends require real PyTorch runtime objects, so factories are
    materialized immediately and DeepSpeed placeholders are rejected.
    DeepSpeed-specific ownership and conversion are delegated to
    ``training._deepspeed``.
    """

    def reject_metric_scheduler(scheduler: object) -> None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError(
                "HookBasedTrainer does not support ReduceLROnPlateau: "
                "its scheduler is stepped without a validation metric."
            )

    reject_metric_scheduler(lr_scheduler)
    if is_deepspeed_accelerator(accelerator):
        return prepare_deepspeed_optimizer_scheduler(
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
    if isinstance(optimizer, DummyOptim):
        raise ValueError(
            "DummyOptim can be used only with a DeepSpeed accelerator."
        )
    if isinstance(lr_scheduler, DummyScheduler):
        raise ValueError(
            "DummyScheduler can be used only with a DeepSpeed accelerator."
        )
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
        return optimizer, lr_scheduler
    if callable(lr_scheduler):
        resolved_scheduler = lr_scheduler(optimizer)
        reject_metric_scheduler(resolved_scheduler)
        if not isinstance(
            resolved_scheduler,
            torch.optim.lr_scheduler.LRScheduler,
        ):
            raise TypeError(
                "LRSchedulerFactory must return a "
                "torch.optim.lr_scheduler.LRScheduler."
            )
        return optimizer, resolved_scheduler
    raise TypeError(
        "lr_scheduler must be an LRScheduler, DummyScheduler, or "
        "LRSchedulerFactory."
    )


class HookBasedTrainer:
    """An NN trainer class that uses hooks to manage the training process.

    The data loader, model, optimizer, and learning rate scheduler are
    prepared using the `Accelerator` instance, which provides
    distributed training capabilities. The `PipelineHooks` are used to
    manage the training process, allowing for custom hooks to be defined
    for various stages of the training loop.

    The whole training process with hooks is as follows:

    .. code-block:: text

        with on_loop:
            with on_epoch:
                for batch in dataloader:
                    with accelerator.accumulate(model):
                        with on_step:
                            with on_batch:
                                batch_processor(...)
                                ...
                            update micro-step progress

                        if optimizer boundary:
                            with on_optimizer_step:
                                run optimizer finalization
                                update optimizer-step progress
                        else:
                            run non-boundary optimizer finalization
            update epoch id


    Note:
        The trainer registers hook sets in this order at initialization:

        - ``ValidationHook``: This hook performs validation during training.
          It calls the evaluation callback at the configured frequency and is
          registered when ``validation`` is provided.

        - User-provided hooks.

        After each ``on_step`` scope exits, the trainer performs optimizer
        finalization. Boundary finalization is wrapped in
        ``on_optimizer_step`` so after-hooks can observe whether the update was
        committed through ``PipelineHookArgs.is_optimizer_step_committed``.
        This means user ``on_step`` after-hooks observe the pre-optimizer
        state, including parameters, gradients, and scheduler state. Hooks
        that need post-update state should register on ``on_optimizer_step``.
        When ``grad_clip`` is provided, ordinary torch/DDP clipping is a
        trainer-owned optimizer-finalization side effect that runs after user
        ``on_optimizer_step`` before-hooks and before ``optimizer.step``.
        DeepSpeed clipping is configured before ``accelerator.prepare(...)``
        through the DeepSpeed config instead of running local clipping hooks.

    Args:
        accelerator (Accelerator): The ``Accelerator`` instance managing
            distributed training.
        model (torch.nn.Module): The model to be trained.
        dataloader (DataLoader | Iterable): The data loader that feeds batches
            to the model during training.
        batch_processor (BatchStepProcessorMixin): The step processor
            responsible for processing batches and backpropagating the loss.
        optimizer (torch.optim.Optimizer | DummyOptim): The optimizer input.
            ``DummyOptim`` is accepted only when the DeepSpeed config owns
            optimizer construction. Automatic DeepSpeed conversion accepts
            only exact ``torch.optim.AdamW``; ordinary backends keep their
            code-owned optimizer.
        lr_scheduler (LRScheduler | DummyScheduler | LRSchedulerFactory): The
            scheduler input. Factories receive the optimizer that owns runtime
            parameter groups: immediately on ordinary backends, or from
            DeepSpeed after it constructs the actual optimizer. Ordinary
            schedulers are stepped during training without a validation
            metric; ``ReduceLROnPlateau`` is not supported. The trainer
            cannot validate a DeepSpeed factory's deferred result before
            the engine creates it.
        hooks (PipelineHooks | Iterable[PipelineHooks] | None, optional): The
            hooks used during training. These hooks can customize various
            stages of the training process. Default is None.
        max_step (int | None, optional): The maximum number of completed
            accumulation windows, including AMP skips. Either ``max_step`` or
            ``max_epoch`` must be specified. Default is None.
        max_epoch (int | None, optional): The maximum number of epochs for
            training. Either ``max_step`` or ``max_epoch`` must be specified.
            Default is None.
        grad_clip (GradientClippingHookConfig | None, optional): The
            trainer-owned gradient clipping configuration. Use this argument
            instead of passing ``GradientClippingHook`` as a user hook,
            especially under DeepSpeed where clipping must be lowered into
            the DeepSpeed config before ``accelerator.prepare(...)``. Default
            is None.
        validation (ValidationHookConfig | None, optional): The validation
            configuration. If not specified, no validation is performed.
            Default is None.
        resume_from (ResumeCheckpointConfig | None, optional): The
            configuration for resuming from checkpoints. If not specified,
            training starts from scratch. Default is None.
    """

    hooks: PipelineHooks
    """All hooks to be used during training."""
    accelerator: Accelerator
    """The `Accelerator` instance managing distributed training."""

    dataloader: DataLoader
    """The data loader for feeding batches to the model during training."""
    model: torch.nn.Module
    """The model to be trained."""
    optimizer: AcceleratedOptimizer
    """The optimizer after being prepared by the `Accelerator`."""
    lr_scheduler: AcceleratedScheduler
    """The learning rate scheduler after being prepared by the
    ``Accelerator``."""
    _grad_clip: GradientClippingHookConfig | None
    """Trainer-owned gradient clipping config for ordinary torch/DDP."""

    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        dataloader: DataLoader | Iterable,
        batch_processor: BatchStepProcessorMixin,
        optimizer: torch.optim.Optimizer | DummyOptim,
        lr_scheduler: (
            torch.optim.lr_scheduler.LRScheduler
            | DummyScheduler
            | LRSchedulerFactory
        ),
        hooks: (
            PipelineHookOrConfigType
            | Iterable[PipelineHookOrConfigType]
            | None
        ) = None,
        max_step: int | None = None,
        max_epoch: int | None = None,
        grad_clip: GradientClippingHookConfig | None = None,
        validation: ValidationHookConfig | None = None,
        resume_from: ResumeCheckpointConfig | None = None,
    ):
        if max_step is None and max_epoch is None:
            raise ValueError(
                "Either `max_step` or `max_epoch` must be specified."
            )
        if max_step is not None and max_step < 1:
            raise ValueError(
                "max_step = {} < 1 is not allowed".format(max_step)
            )
        if max_epoch is not None and max_epoch < 1:
            raise ValueError(
                "max_epoch = {} < 1 is not allowed".format(max_epoch)
            )
        if hooks is None:
            hooks = []
        self.accelerator = accelerator
        is_deepspeed = is_deepspeed_accelerator(accelerator)
        raw_user_hooks, grad_clip = _split_gradient_clipping_config(
            hooks=hooks,
            grad_clip=grad_clip,
            is_deepspeed=is_deepspeed,
        )
        if is_deepspeed and grad_clip is not None:
            configure_deepspeed_gradient_clipping(
                accelerator=accelerator,
                grad_clip=grad_clip,
            )
            grad_clip = None
        self._grad_clip = grad_clip
        user_hooks = PipelineHooks.from_hooks(raw_user_hooks)
        self.max_step = max_step
        self.max_epoch = max_epoch

        if isinstance(model, TorchModelMixin):
            model.accelerate_model_id = len(accelerator._models)
            model.accelerator_register_all_hooks(accelerator)

        optimizer, lr_scheduler = _resolve_optimizer_scheduler_before_prepare(
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        # Normalize iterable-dataset settings before a single
        # accelerator.prepare(...) call across all training objects.
        should_check_slow_path = configure_data_loader_for_accelerate(
            accelerator=accelerator, data_loader=dataloader
        )
        source_dataloader = dataloader
        self.dataloader, self.model, self.optimizer, self.lr_scheduler = (
            accelerator.prepare(
                dataloader,
                model,
                optimizer,
                lr_scheduler,
            )
        )
        _configure_prepared_loader_recovery(
            accelerator, self.dataloader, source_dataloader
        )
        if should_check_slow_path:
            _warn_if_prepare_falls_back_to_slow_path(self.dataloader)

        self.dataloader: DataLoader = self.dataloader
        self.model: torch.nn.Module = self.model
        self.optimizer: AcceleratedOptimizer = self.optimizer
        self.lr_scheduler: AcceleratedScheduler = self.lr_scheduler

        self.trainer_progress_state = TrainerProgressState()
        accelerator.register_for_checkpointing(self.trainer_progress_state)

        self.batch_processor = batch_processor

        self.hooks = PipelineHooks()
        if validation is not None:
            self.hooks += validation()

        # register user hooks
        self.hooks += user_hooks
        self._start_epoch = 0
        self._start_step = 0

        if resume_from is not None:
            logger.info(f"Resume from: {resume_from}", main_process_only=True)
            resume_from.load_state(accelerator=self.accelerator)
            if self.trainer_progress_state.micro_step.index_in_optimizer_step:
                warnings.warn(
                    "Checkpoint contains an unfinished accumulation window; "
                    "its gradients were not restored. Starting a new window.",
                    UserWarning,
                )
                self.trainer_progress_state.reset_optimizer_step_window()
            _load_prepared_loader_recovery(
                accelerator,
                self.dataloader,
                self.trainer_progress_state.rank_states,
                epoch=self.trainer_progress_state.epoch_id,
                processed_batches=(
                    self.trainer_progress_state.micro_step.epoch_step_id
                ),
            )
            logger.warning(
                "LossTracker and MetricTracker aggregation state is not "
                "restored from checkpoints; statistics spanning a resume "
                "may contain only post-resume samples until their next reset.",
                main_process_only=True,
            )
            self._start_epoch = self.trainer_progress_state.epoch_id
            self._start_step = self.trainer_progress_state.step_id

    def _get_hook_args(
        self,
        *,
        current_micro_step: MicroStepProgressState | None = None,
    ) -> PipelineHookArgs:
        """Build hook args from trainer-owned runtime and progress state.

        Creates and returns a ``PipelineHookArgs`` object with stable
        trainer-owned runtime objects and the current committed progress
        snapshot. When ``current_micro_step`` is provided, the args also
        include the pending micro-step runtime state for one ``on_step`` event.

        Args:
            current_micro_step (MicroStepProgressState | None): Preview state
                for the current dataloader micro step, if this event belongs
                to the step loop.

        Returns:
            PipelineHookArgs: An object containing the current training state
            and runtime objects.
        """
        hookargs = PipelineHookArgs(
            accelerator=self.accelerator,
            max_step=self.max_step,
            max_epoch=self.max_epoch,
            dataloader=self.dataloader,
            _trainer_dataloader=self.dataloader,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            start_epoch=self._start_epoch,
            start_step=self._start_step,
        )
        self.trainer_progress_state.sync_pipeline_hook_arg(hookargs)
        if current_micro_step is not None:
            hookargs.micro_step = current_micro_step
            hookargs.is_optimizer_step_boundary = (
                self.accelerator.sync_gradients
            )
        return hookargs

    def _run_optimizer_step(
        self,
        hook_args: PipelineHookArgs,
        *,
        is_optimizer_step_boundary: bool,
        micro_steps: int,
        deepspeed_step_before: DeepSpeedStepSnapshot | None = None,
    ) -> bool:
        """Finalize one micro step's optimizer progress.

        Ordinary torch/DDP backends call optimizer, scheduler, and zero-grad
        wrappers here. DeepSpeed backends only observe engine counters because
        ``accelerator.backward`` already owns the actual engine step.
        Optimizer-boundary and window-size decisions come from trainer-owned
        arguments, not from mutable ``PipelineHookArgs`` fields.

        Returns:
            bool: ``True`` when the current optimizer boundary committed a
            model update.

        Raises:
            RuntimeError: If the hook args do not carry micro-step progress,
            or if DeepSpeed finalization is missing its pre-step snapshot.
        """
        if hook_args.exception is not None:
            return False
        if hook_args.micro_step is None:
            raise RuntimeError(
                "HookBasedTrainer optimizer step requires "
                "PipelineHookArgs.micro_step."
            )
        if is_deepspeed_accelerator(self.accelerator):
            if deepspeed_step_before is None:
                raise RuntimeError(
                    "DeepSpeed optimizer-step observation requires a "
                    "pre-step snapshot from HookBasedTrainer._run_batch_step."
                )
            committed = observe_deepspeed_optimizer_step(
                accelerator=self.accelerator,
                is_optimizer_step_boundary=is_optimizer_step_boundary,
                step_before=deepspeed_step_before,
            )
        else:
            if is_optimizer_step_boundary and self._grad_clip is not None:
                _clip_optimizer_gradients(
                    accelerator=self.accelerator,
                    optimizer=self.optimizer,
                    clip_mode=self._grad_clip.clip_mode,
                    clip_value=self._grad_clip.clip_value,
                    max_norm=self._grad_clip.max_norm,
                    norm_type=self._grad_clip.norm_type,
                )
            self.optimizer.step()
            committed = (
                is_optimizer_step_boundary
                and not self.accelerator.optimizer_step_was_skipped
            )
            scheduler_handles_optimizer_step_gating = (
                isinstance(self.lr_scheduler, AcceleratedScheduler)
                and self.lr_scheduler.step_with_optimizer
            )
            if scheduler_handles_optimizer_step_gating or committed:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        if is_optimizer_step_boundary:
            self.trainer_progress_state.commit_optimizer_step(
                micro_steps=micro_steps
            )
            self.trainer_progress_state.sync_pipeline_hook_arg(hook_args)
        return committed

    def _run_batch_step(
        self, batch: Any, *, data_iteration: PreparedDataLoaderIteration
    ) -> None:
        """Run one batch through step and batch hooks.

        The batch processor owns forward/backward work. The trainer commits
        micro-step progress before step after-hooks run, then handles
        optimizer side effects or DeepSpeed step observation after leaving
        the on_step scope.
        """
        current_micro_step = (
            self.trainer_progress_state.preview_next_micro_step()
        )
        is_optimizer_step_boundary = self.accelerator.sync_gradients
        micro_steps = current_micro_step.index_in_optimizer_step
        deepspeed_step_before = read_deepspeed_step_snapshot(self.accelerator)
        with self.hooks.begin(
            "on_step",
            self._get_hook_args(
                current_micro_step=current_micro_step,
            ),
        ) as on_step_hook_args:
            with self.hooks.begin(
                "on_batch",
                on_step_hook_args.copy_with_updates(batch=batch),
            ) as on_batch_hook_args:
                self.batch_processor(
                    pipeline_hooks=self.hooks,
                    on_batch_hook_args=on_batch_hook_args,
                    model=self.model,
                )
                on_step_hook_args.model_outputs = (
                    on_batch_hook_args.model_outputs
                )
            self.trainer_progress_state.commit_micro_step(current_micro_step)
            self.trainer_progress_state.sync_pipeline_hook_arg(
                on_step_hook_args
            )
        self.trainer_progress_state.sync_pipeline_hook_arg(on_step_hook_args)
        on_step_hook_args.is_optimizer_step_boundary = (
            is_optimizer_step_boundary
        )
        on_step_hook_args.is_optimizer_step_committed = False
        optimizer_scope = (
            self.hooks.begin(
                "on_optimizer_step",
                on_step_hook_args,
            )
            if is_optimizer_step_boundary
            else nullcontext(on_step_hook_args)
        )
        with optimizer_scope as optimizer_hook_args:
            committed = self._run_optimizer_step(
                optimizer_hook_args,
                is_optimizer_step_boundary=is_optimizer_step_boundary,
                micro_steps=micro_steps,
                deepspeed_step_before=deepspeed_step_before,
            )
            if is_optimizer_step_boundary:
                data_iteration.commit()
            optimizer_hook_args.is_optimizer_step_committed = committed
            self.trainer_progress_state.sync_pipeline_hook_arg(
                optimizer_hook_args
            )

    def _run_epoch_steps(
        self,
        on_epoch_hook_args: PipelineHookArgs,
    ) -> bool:
        """Run one epoch's dataloader iterator and always close it.

        Returns:
            bool: Whether the training loop should end after this epoch.
        """
        end_loop_flag = False
        with iterate_prepared_loader(
            self.accelerator,
            self.dataloader,
            epoch=self.trainer_progress_state.epoch_id,
        ) as iteration:
            for batch in iteration:
                with self.accelerator.accumulate(self.model):
                    self._run_batch_step(batch=batch, data_iteration=iteration)
                self.trainer_progress_state.sync_pipeline_hook_arg(
                    on_epoch_hook_args
                )
                end_loop_flag = self.trainer_progress_state.is_training_end(
                    max_step=self.max_step,
                    max_epoch=self.max_epoch,
                )
                if iteration.should_stop(local_stop=end_loop_flag):
                    break
        return end_loop_flag

    def _run_epoch(self) -> bool:
        """Finish epoch hooks and data transition before queued saves."""
        epoch_saves: list[Callable[[], None]] = []
        epoch_args = self._get_hook_args()
        epoch_args._defer_epoch_checkpoint = epoch_saves.append
        with self.hooks.begin("on_epoch", epoch_args) as hook_args:
            end_loop = self._run_epoch_steps(hook_args)
        if not end_loop:
            self.trainer_progress_state.update_epoch()
            _set_prepared_loader_epoch(
                self.dataloader,
                self.trainer_progress_state.epoch_id,
            )
        for save in epoch_saves:
            save()
        return end_loop

    def __call__(self):
        logger.info(
            "\n" + "=" * 50 + "BEGIN TRAINING" + "=" * 50,
            main_process_only=True,
        )
        logger.info(
            f"Start training loop from epoch {self._start_epoch} "
            f"and step {self._start_step}",
            main_process_only=True,
        )
        end_loop_flag = self.trainer_progress_state.is_training_end(
            max_step=self.max_step, max_epoch=self.max_epoch
        )
        if end_loop_flag:
            self.accelerator.set_trigger()
        end_loop_flag = self.accelerator.check_trigger()
        self.model.train()

        primary_exc: BaseException | None = None
        try:
            with self.hooks.begin(
                "on_loop", self._get_hook_args()
            ) as on_loop_hook_args:
                while not end_loop_flag:
                    # Uneven rank inputs may make gradient accumulation
                    # synchronize on different batches and hang. Callers
                    # must configure distributed sampling accordingly.
                    end_loop_flag = self._run_epoch()
                    self.trainer_progress_state.sync_pipeline_hook_arg(
                        on_loop_hook_args
                    )
                    if self.trainer_progress_state.is_training_end(
                        max_step=self.max_step, max_epoch=self.max_epoch
                    ):
                        end_loop_flag = True
                        self.accelerator.set_trigger()
                    if self.accelerator.check_trigger():
                        end_loop_flag = True
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            _close_dataloader_owner_resources(
                self.dataloader,
                primary_exc=primary_exc,
            )

        logger.info(
            "\n" + "=" * 50 + "FINISH TRAINING" + "=" * 50,
            main_process_only=True,
        )
