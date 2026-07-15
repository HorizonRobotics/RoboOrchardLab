# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedType,
    DummyOptim,
    DummyScheduler,
)

from robo_orchard_lab.pipeline.hooks.grad_clip import (
    GradientClippingHookConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs
from robo_orchard_lab.pipeline.training._progress import TrainerProgressState

__all__ = [
    "DeepSpeedStepSnapshot",
    "commit_deepspeed_optimizer_progress",
    "configure_deepspeed_gradient_clipping",
    "is_deepspeed_accelerator",
    "prepare_deepspeed_optimizer_scheduler",
    "read_deepspeed_step_snapshot",
]


logger = get_logger(__name__)

_MISSING_GRADIENT_CLIPPING = object()
_DEEPSPEED_OPTIMIZER_TYPE = "AdamW"
_UNSUPPORTED_DEEPSPEED_OPTIMIZER_FLAGS = (
    "amsgrad",
    "foreach",
    "maximize",
    "capturable",
    "differentiable",
    "fused",
)
_DEEPSPEED_ADAMW_PARAM_KEYS = (
    "lr",
    "weight_decay",
    "betas",
    "eps",
)
_LRSchedulerFactory = Callable[
    [torch.optim.Optimizer],
    torch.optim.lr_scheduler.LRScheduler,
]


@dataclass(frozen=True)
class DeepSpeedStepSnapshot:
    """DeepSpeed engine step counters observed around one micro step."""

    global_steps: int
    """DeepSpeed global optimizer-step attempts observed on the engine."""
    skipped_steps: int
    """DeepSpeed skipped optimizer-step attempts observed on the engine."""


def is_deepspeed_accelerator(accelerator: Accelerator) -> bool:
    """Return whether the accelerator is running a DeepSpeed backend."""

    return accelerator.distributed_type == DistributedType.DEEPSPEED


def prepare_deepspeed_optimizer_scheduler(
    *,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer | DummyOptim,
    lr_scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | DummyScheduler
        | _LRSchedulerFactory
    ),
) -> tuple[
    DummyOptim,
    DummyScheduler,
]:
    """Resolve optimizer and scheduler inputs for DeepSpeed prepare.

    For a real AdamW input, Python owns the optimizer recipe while this
    boundary lowers it into a runtime-only DeepSpeed optimizer section and a
    dummy pair. Static DeepSpeed profiles therefore need only describe engine
    topology. For an explicit ``DummyOptim`` input, the caller-owned runtime
    config must already describe the optimizer.

    A scheduler factory is delayed until DeepSpeed has constructed the actual
    optimizer. The returned dummy pair ensures both objects are stepped,
    saved, and loaded by the DeepSpeed engine rather than the trainer.

    Args:
        accelerator (Accelerator): DeepSpeed-backed accelerator whose mutable
            config defines runtime ownership.
        optimizer (torch.optim.Optimizer | DummyOptim): Optimizer input from
            the caller.
        lr_scheduler (LRScheduler | DummyScheduler | Callable): Concrete
            scheduler, DeepSpeed placeholder, or actual-optimizer factory.

    Returns:
        tuple[DummyOptim, DummyScheduler]: Engine-owned placeholders that can
        safely be passed to the same ``accelerator.prepare(...)``.

    Raises:
        TypeError: If automatic lowering cannot preserve optimizer
            semantics or an input has an unsupported runtime type.
        ValueError: If runtime inputs and DeepSpeed config declare conflicting
            owners, or a Python-owned scheduler would fall outside the
            DeepSpeed engine lifecycle.
    """

    _, deepspeed_config = _deepspeed_plugin_and_config(accelerator)
    config_owns_optimizer = "optimizer" in deepspeed_config
    config_owns_scheduler = "scheduler" in deepspeed_config

    if isinstance(optimizer, DummyOptim):
        if not config_owns_optimizer:
            raise ValueError(
                "An explicit DummyOptim requires an optimizer section in the "
                "DeepSpeed config. Pass a real supported optimizer to let "
                "HookBasedTrainer materialize the runtime config."
            )
        if isinstance(lr_scheduler, DummyScheduler):
            _validate_dummy_scheduler_ownership(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config_owns_scheduler=config_owns_scheduler,
            )
            return optimizer, lr_scheduler
        if callable(lr_scheduler):
            if config_owns_scheduler:
                raise ValueError(
                    "DeepSpeed config must not define a scheduler when "
                    "lr_scheduler is an LRSchedulerFactory. Choose one "
                    "scheduler source of truth."
                )
            return optimizer, DummyScheduler(
                optimizer=optimizer,
                lr_scheduler_callable=lr_scheduler,
            )
        raise ValueError(
            "DummyOptim must be paired with DummyScheduler or an "
            "LRSchedulerFactory. A concrete scheduler would be bound to a "
            "different optimizer."
        )

    if config_owns_optimizer:
        raise ValueError(
            "A real optimizer is the optimizer recipe source of truth, so "
            "the DeepSpeed config must not also define optimizer. Remove the "
            "optimizer section, or pass an explicit DummyOptim to select a "
            "config-owned optimizer."
        )
    if config_owns_scheduler:
        raise ValueError(
            "A real optimizer plus Python scheduler factory cannot be "
            "combined with a DeepSpeed config-owned scheduler. Remove the "
            "scheduler section, or pass an explicit DummyOptim and "
            "DummyScheduler for a fully config-owned pair."
        )
    if not callable(lr_scheduler):
        raise ValueError(
            "A real optimizer under DeepSpeed requires an "
            "LRSchedulerFactory so the scheduler binds the actual optimizer."
        )
    dummy_optimizer = _lower_adamw_optimizer_for_deepspeed(
        optimizer=optimizer,
        deepspeed_config=deepspeed_config,
    )
    return dummy_optimizer, DummyScheduler(
        optimizer=dummy_optimizer,
        lr_scheduler_callable=lr_scheduler,
    )


def read_deepspeed_step_snapshot(
    accelerator: Accelerator,
) -> DeepSpeedStepSnapshot | None:
    """Read DeepSpeed step counters when the accelerator owns an engine.

    Returns:
        DeepSpeedStepSnapshot | None: Counter snapshot for DeepSpeed
        accelerators, or ``None`` when the accelerator uses another backend.

    Raises:
        RuntimeError: If DeepSpeed is active but the engine wrapper has not
        been prepared.
    """

    if not is_deepspeed_accelerator(accelerator):
        return None

    engine_wrapper = accelerator.deepspeed_engine_wrapped
    if engine_wrapper is None:
        raise RuntimeError(
            "DeepSpeed accelerator is active, but no prepared DeepSpeed "
            "engine wrapper is available."
        )

    engine = engine_wrapper.engine
    return DeepSpeedStepSnapshot(
        global_steps=int(engine.global_steps),
        skipped_steps=int(engine.skipped_steps),
    )


def configure_deepspeed_gradient_clipping(
    *,
    accelerator: Accelerator,
    grad_clip: GradientClippingHookConfig,
) -> None:
    """Inject trainer-owned norm clipping into DeepSpeed config.

    ``gradient_clipping='auto'`` is an Accelerate/HF placeholder, not a
    DeepSpeed runtime clipping value. When trainer ``grad_clip`` is provided,
    that logical config becomes the clipping source of truth and must replace
    ``'auto'`` before ``accelerator.prepare(...)``.
    """

    if grad_clip.clip_mode != "norm":
        raise ValueError(
            "DeepSpeed gradient clipping supports only norm clipping from "
            "GradientClippingHookConfig. Configure value clipping outside "
            "DeepSpeed or disable trainer grad_clip."
        )
    if grad_clip.norm_type != 2.0:
        raise ValueError(
            "DeepSpeed gradient clipping maps only the default L2 norm "
            "semantics. norm_type must be 2.0 when DeepSpeed is active."
        )
    if grad_clip.max_norm is None:
        raise ValueError(
            "DeepSpeed gradient clipping requires max_norm when "
            "clip_mode='norm'."
        )

    deepspeed_plugin, deepspeed_config = _deepspeed_plugin_and_config(
        accelerator
    )
    current_value = deepspeed_config.get(
        "gradient_clipping", _MISSING_GRADIENT_CLIPPING
    )
    if current_value is _MISSING_GRADIENT_CLIPPING:
        deepspeed_config["gradient_clipping"] = grad_clip.max_norm
        deepspeed_plugin.gradient_clipping = grad_clip.max_norm
        return
    if current_value == "auto":
        logger.warning(
            "DeepSpeed config has gradient_clipping='auto' while trainer "
            "grad_clip is provided; overriding it with grad_clip.max_norm=%s "
            "before accelerator.prepare(...).",
            grad_clip.max_norm,
            main_process_only=True,
        )
        deepspeed_config["gradient_clipping"] = grad_clip.max_norm
        deepspeed_plugin.gradient_clipping = grad_clip.max_norm
        return

    try:
        current_float = float(current_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DeepSpeed config field `gradient_clipping` must be numeric, "
            "'auto', or omitted when trainer grad_clip is provided."
        ) from exc
    if current_float != float(grad_clip.max_norm):
        raise ValueError(
            "Conflicting gradient clipping configuration: DeepSpeed config "
            f"has gradient_clipping={current_value!r}, but trainer "
            f"grad_clip.max_norm={grad_clip.max_norm!r}."
        )
    deepspeed_config["gradient_clipping"] = grad_clip.max_norm
    deepspeed_plugin.gradient_clipping = grad_clip.max_norm


def commit_deepspeed_optimizer_progress(
    *,
    accelerator: Accelerator,
    progress_state: TrainerProgressState,
    hook_args: PipelineHookArgs,
    is_optimizer_step_boundary: bool,
    micro_steps: int,
    step_before: DeepSpeedStepSnapshot,
) -> bool:
    """Commit trainer progress from DeepSpeed engine step counters.

    Accelerate calls ``DeepSpeedEngine.step()`` from ``accelerator.backward``
    at sync-gradient boundaries. The trainer must therefore observe DeepSpeed
    counters instead of calling optimizer, scheduler, or zero-grad methods.

    Returns:
        bool: ``True`` when the observed boundary committed a model update,
        otherwise ``False`` for non-boundary micro steps or skipped optimizer
        boundaries.

    Raises:
        RuntimeError: If DeepSpeed counters move in a way that cannot be
        mapped to one HookBasedTrainer micro step.
    """

    if hook_args.micro_step is None:
        raise RuntimeError(
            "DeepSpeed optimizer-step observation requires "
            "PipelineHookArgs.micro_step."
        )

    step_after = read_deepspeed_step_snapshot(accelerator)
    if step_after is None:
        raise RuntimeError(
            "DeepSpeed optimizer-step observation requires a prepared "
            "DeepSpeed engine."
        )

    global_step_delta = step_after.global_steps - step_before.global_steps
    skipped_step_delta = step_after.skipped_steps - step_before.skipped_steps
    if global_step_delta < 0 or skipped_step_delta < 0:
        raise RuntimeError(
            "DeepSpeed optimizer-step counters moved backwards during "
            "one HookBasedTrainer micro step."
        )
    if skipped_step_delta > global_step_delta:
        raise RuntimeError(
            "DeepSpeed skipped-step counter advanced without a matching "
            "global-step counter advance."
        )
    if global_step_delta > 1:
        raise RuntimeError(
            "DeepSpeed advanced more than one optimizer step during one "
            "HookBasedTrainer micro step."
        )

    if not is_optimizer_step_boundary:
        if global_step_delta != 0 or skipped_step_delta != 0:
            raise RuntimeError(
                "DeepSpeed advanced optimizer progress on a non-boundary "
                "HookBasedTrainer micro step."
            )
        return False

    if global_step_delta == 0:
        raise RuntimeError(
            "DeepSpeed did not advance optimizer progress on an "
            "Accelerate optimizer-step boundary. Ensure the batch "
            "processor calls accelerator.backward(loss)."
        )

    if skipped_step_delta:
        progress_state.reset_optimizer_step_window()
        committed = False
    else:
        progress_state.commit_optimizer_step(micro_steps=micro_steps)
        committed = True
    progress_state.sync_pipeline_hook_arg(hook_args)
    return committed


def _validate_dummy_scheduler_ownership(
    *,
    optimizer: DummyOptim,
    lr_scheduler: DummyScheduler,
    config_owns_scheduler: bool,
) -> None:
    """Validate one explicitly constructed DeepSpeed dummy pair."""

    if lr_scheduler.optimizer is not optimizer:
        raise ValueError(
            "DummyScheduler.optimizer must be the same DummyOptim passed to "
            "HookBasedTrainer."
        )
    if config_owns_scheduler:
        if lr_scheduler.lr_scheduler_callable is not None:
            raise ValueError(
                "DummyScheduler must not define lr_scheduler_callable when "
                "the DeepSpeed config owns the scheduler."
            )
        return
    if lr_scheduler.lr_scheduler_callable is None:
        raise ValueError(
            "DummyScheduler requires lr_scheduler_callable when the "
            "DeepSpeed config does not define a scheduler."
        )


def _lower_adamw_optimizer_for_deepspeed(
    *,
    optimizer: torch.optim.Optimizer,
    deepspeed_config: MutableMapping[str, Any],
) -> DummyOptim:
    """Lower one supported PyTorch AdamW recipe into DeepSpeed inputs."""

    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError(
            "DeepSpeed optimizer lowering supports only exact "
            "torch.optim.AdamW. Pass an explicit DummyOptim/DummyScheduler "
            "pair for custom or unsupported optimizers."
        )

    defaults = optimizer.defaults
    _reject_enabled_deepspeed_optimizer_flags(
        optimizer_options=defaults,
        location="optimizer defaults",
    )
    if optimizer.state:
        raise ValueError(
            "DeepSpeed optimizer lowering requires a pristine AdamW without "
            "optimizer state. Restore optimizer and scheduler state through "
            "the prepared Accelerator/DeepSpeed checkpoint instead."
        )
    if (
        "decoupled_weight_decay" in defaults
        and defaults["decoupled_weight_decay"] is not True
    ):
        raise TypeError(
            "torch.optim.AdamW must use decoupled_weight_decay=True for "
            "DeepSpeed optimizer lowering."
        )

    normalized_groups: list[dict[str, Any]] = []
    for group_idx, param_group in enumerate(optimizer.param_groups):
        _reject_enabled_deepspeed_optimizer_flags(
            optimizer_options=param_group,
            location=f"param group {group_idx}",
        )
        if (
            "decoupled_weight_decay" in param_group
            and param_group["decoupled_weight_decay"] is not True
        ):
            raise TypeError(
                "DeepSpeed optimizer lowering requires param group "
                f"{group_idx} decoupled_weight_decay=True."
            )
        normalized_groups.append(
            {
                "params": param_group["params"],
                **{
                    key: param_group[key]
                    for key in _DEEPSPEED_ADAMW_PARAM_KEYS
                },
            }
        )

    deepspeed_config["optimizer"] = {
        "type": _DEEPSPEED_OPTIMIZER_TYPE,
        "params": {
            "lr": defaults["lr"],
            "weight_decay": defaults["weight_decay"],
            "betas": list(defaults["betas"]),
            "eps": defaults["eps"],
        },
    }
    return DummyOptim(
        params=normalized_groups,
        lr=defaults["lr"],
        weight_decay=defaults["weight_decay"],
    )


def _reject_enabled_deepspeed_optimizer_flags(
    *,
    optimizer_options: Mapping[str, Any],
    location: str,
) -> None:
    """Reject enabled PyTorch-only optimizer behavior before lowering."""

    unsupported_enabled = [
        key
        for key in _UNSUPPORTED_DEEPSPEED_OPTIMIZER_FLAGS
        if optimizer_options.get(key)
    ]
    if unsupported_enabled:
        raise TypeError(
            "DeepSpeed optimizer lowering does not support enabled PyTorch "
            f"optimizer options in {location}: "
            f"{sorted(unsupported_enabled)}."
        )


def _deepspeed_plugin_and_config(
    accelerator: Accelerator,
) -> tuple[Any, MutableMapping[str, Any]]:
    """Return the DeepSpeed plugin and mutable pre-prepare config.

    Raises:
        RuntimeError: If the accelerator does not expose a mutable DeepSpeed
            plugin config before ``accelerator.prepare(...)``.
    """

    deepspeed_plugin = getattr(accelerator, "deepspeed_plugin", None)
    if deepspeed_plugin is None:
        state = getattr(accelerator, "state", None)
        deepspeed_plugin = getattr(state, "deepspeed_plugin", None)
    deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", None)
    if not isinstance(deepspeed_config, MutableMapping):
        raise RuntimeError(
            "DeepSpeed preparation requires a mutable "
            "accelerator.deepspeed_plugin.deepspeed_config before "
            "accelerator.prepare(...)."
        )
    return deepspeed_plugin, deepspeed_config
