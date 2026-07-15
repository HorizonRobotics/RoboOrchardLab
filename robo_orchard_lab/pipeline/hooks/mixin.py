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
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeAlias,
    runtime_checkable,
)

import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType,
)
from robo_orchard_core.utils.hook import (
    HookContext,
    HookContextChannel,
    RemoveableHandle,
)
from robo_orchard_core.utils.string import add_indentation
from typing_extensions import Self, TypeVar

__all__ = [
    "ClassType",
    "HookContext",
    "PipelineHooks",
    "PipelineHookArgs",
    "MicroStepProgressState",
    "PipelineHookChanelType",
    "PipelineHooksConfig",
    "ModelOutput",
    "ModelOutputHasLossKeys",
]


@runtime_checkable
class ModelOutput(Protocol):
    """A protocol representing a model output that contains key-value pairs."""

    def __getitem__(self, key: str) -> Any: ...

    def keys(self) -> Iterable[Any]: ...

    def items(self) -> tuple[Any, Any]: ...

    def values(self) -> Iterable[Any]: ...


@runtime_checkable
class ModelOutputHasLossKeys(ModelOutput, Protocol):
    def loss_keys(self) -> Iterable[str]: ...


@dataclass
class MicroStepProgressState:
    """Progress for dataloader-batch micro steps in trainer loops.

    The top-level ``step_id`` and ``global_step_id`` fields in
    ``PipelineHookArgs`` are optimizer-step counters. This state carries the
    batch-level counters used by ``on_step`` and ``on_batch`` events when
    gradient accumulation is enabled.
    """

    epoch_step_id: int = 0
    """Number of micro steps completed in the current epoch."""
    global_step_id: int = 0
    """Number of micro steps completed across all epochs."""
    index_in_optimizer_step: int = 0
    """Number of micro steps accumulated in the current optimizer window."""
    last_optimizer_step_size: int = 0
    """Number of micro steps used by the last committed optimizer step."""


@dataclass
class PipelineHookArgs:
    """Mutable event context passed through pipeline hook scopes.

    Hook handlers use this object to inspect trainer-owned runtime objects and
    the current progress snapshot. ``step_id`` and ``global_step_id`` are
    committed optimizer-step counters; hooks that need dataloader-batch
    progress should read ``micro_step`` when it is provided by a trainer loop.
    """

    accelerator: Accelerator
    epoch_id: int = 0
    step_id: int = 0
    """Current optimizer-step id within the epoch."""
    global_step_id: int = 0
    """Current global optimizer-step id."""
    micro_step: Optional[MicroStepProgressState] = None
    """Current micro-step progress, when created by a trainer loop."""
    is_optimizer_step_boundary: bool = True
    """Whether this micro step is an optimizer-finalization boundary."""
    is_optimizer_step_committed: bool = False
    """Whether the optimizer boundary committed a model update.

    This field is written by ``HookBasedTrainer`` inside the
    ``on_optimizer_step`` scope after optimizer finalization. It is false for
    ordinary micro-step events, non-boundary micro steps, and skipped
    optimizer boundaries.
    """
    max_epoch: Optional[int] = None
    max_step: Optional[int] = None
    start_epoch: int = 0
    start_step: int = 0
    dataloader: Optional[Iterable] = None
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[AcceleratedOptimizer] = None
    lr_scheduler: Optional[AcceleratedScheduler] = None
    batch: Optional[Any] = None
    model_outputs: Optional[ModelOutput] = None
    exception: BaseException | None = None
    """The exception raised by the hook context body, if one occurred.

    After hooks can use this field to skip side effects such as optimizer
    steps, checkpoint saves, or distributed collectives when the wrapped body
    is already failing. The original exception is re-raised by the context
    manager.
    """
    reduced_backward_loss: Optional[torch.Tensor] = None
    """The detached backward loss reduced across processes, if applicable.

    When the batch processor returns a loss for backward computation, this
    field stores the detached loss after backward has run and after the value
    is reduced across distributed processes. This value is distinct from
    model-output loss entries, which may include auxiliary or diagnostic
    losses.
    """

    @property
    def reduce_loss(self) -> Optional[torch.Tensor]:
        """Deprecated alias for :attr:`reduced_backward_loss`.

        Returns:
            Optional[torch.Tensor]: The detached reduced backward loss, if it
            was computed for the current micro step.
        """

        warnings.warn(
            "PipelineHookArgs.reduce_loss is deprecated and read-only; "
            "use PipelineHookArgs.reduced_backward_loss instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.reduced_backward_loss

    def copy_with_updates(self, **kwargs) -> PipelineHookArgs:
        """Create a copy of the current instance with updated attributes.

        This method allows you to create a new instance of the class with
        modified attributes while keeping the original instance unchanged.

        Args:
            **kwargs: Keyword arguments representing the attributes to be
                updated. The keys should match the attribute names of the
                class.

        Returns:
            PipelineHookArgs: A new instance of the class with updated
                attributes.
        """

        instance = self.__class__(**self.__dict__)
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute")
        return instance


PipelineHookChanelType: TypeAlias = Literal[
    "on_loop",  # the whole training loop pipeline
    "on_epoch",  # in one epoch pipeline
    "on_step",  # in one dataloader-step pipeline.
    "on_batch",  # in one batch pipeline
    "on_model_forward",  # only in model forward pipeline
    "on_model_backward",  # only in model backward pipeline
    "on_optimizer_step",  # around optimizer-step finalization
]


class PipelineHooks(ClassInitFromConfigMixin):
    """Registry for context-manager hooks used by trainer pipeline stages.

    Each ``on_*`` channel represents a scoped event. ``begin(...)`` enters the
    channel and runs registered before/after callbacks around the wrapped
    body. If the body raises, the exception is written to
    ``PipelineHookArgs.exception`` before it is re-raised so after-hooks can
    avoid unsafe side effects.
    """

    def __init__(self):
        self.hooks: dict[
            PipelineHookChanelType, HookContextChannel[PipelineHookArgs]
        ] = {}
        for c in PipelineHookChanelType.__args__:
            self.hooks[c] = HookContextChannel[PipelineHookArgs](c)

    @contextmanager
    def begin(self, channel: PipelineHookChanelType, arg: PipelineHookArgs):
        """Enter a hook channel context.

        Args:
            channel (PipelineHookChanelType): The scoped hook channel to enter.
            arg (PipelineHookArgs): Mutable event context shared by callbacks
                and the wrapped body.

        Yields:
            PipelineHookArgs: The same event context object after registered
            before-hooks have run.
        """
        with self.hooks[channel].begin(arg) as ctx:
            try:
                yield ctx
            except BaseException as exc:
                arg.exception = exc
                raise

    def register_hook(
        self,
        channel: PipelineHookChanelType,
        hook: HookContext[PipelineHookArgs],
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a hook context handler.

        Args:
            channel (PipelineHookChanelType): The channel to register the hook.
            hook (HookContext[PipelineHookArgs]): The hook context handler
                to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hook.
        """
        return self.hooks[channel].register(hook)

    def register_pipeline_hooks(
        self,
        hooks: PipelineHooks,
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a set of pipeline hooks.

        Args:
            hooks (PipelineHooks[T]): The pipeline hooks to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hooks.
        """
        handles: list[RemoveableHandle] = []
        for channel, hook in hooks.hooks.items():
            handles.append(self.hooks[channel].register_hook_channel(hook))

        def remove():
            for handle in handles:
                handle()

        return RemoveableHandle(remove)

    def __iadd__(self, other: PipelineHooks) -> Self:
        """Add another set of pipeline hooks to the current instance.

        Args:
            other (PipelineHooks): The other set of pipeline hooks to add.

        Returns:
            PipelineHooks: The updated instance with the added hooks.
        """
        self.register_pipeline_hooks(other)
        return self

    def unregister_all(self):
        """Unregister all hook context handlers."""
        for channel in self.hooks.values():
            channel.unregister_all()

    @classmethod
    def from_hooks(
        cls: Type[Self],
        hooks: (
            Self
            | PipelineHooksConfig
            | Iterable[Self | PipelineHooksConfig]
            | None
        ),
    ) -> Self:
        """Create a new instance of the class from a list of hooks.

        Args:
            hooks (Self | Iterable[Self] | None): A list of hooks to register.

        Returns:
            Self: A new instance of the class with the registered hooks.
        """

        if hooks is None:
            return cls()

        if isinstance(hooks, (PipelineHooksConfig, PipelineHooks)):
            hooks_or_cfg_: list[PipelineHooks | PipelineHooksConfig] = [hooks]
        else:
            hooks_or_cfg_ = hooks  # type: ignore

        input_hooks: list[PipelineHooks] = []
        for hook_or_cfg in hooks_or_cfg_:
            if isinstance(hook_or_cfg, PipelineHooksConfig):
                hook = hook_or_cfg()
            elif isinstance(hook_or_cfg, PipelineHooks):
                hook = hook_or_cfg
            else:
                raise TypeError(
                    f"Expected PipelineHooks or PipelineHooksConfig, "
                    f"but got {type(hook_or_cfg)}"
                )
            input_hooks.append(hook)

        ret = cls()
        for hook in input_hooks:
            ret += hook
        return ret

    def __repr__(self) -> str:
        hook_str = "{"
        for k, v in self.hooks.items():
            if len(v) == 0:
                continue
            hook_str += "\n" + add_indentation(f"{k}: {v}, ", indent=2)
        if hook_str != "{":
            hook_str += "\n"
        hook_str += "}"
        content = f"hooks={hook_str}"
        ret = (
            f"<{self.__class__.__module__}.{self.__class__.__name__}(\n"
            + add_indentation(content, indent=2, first_line_indent=True)
            + ")>"
        )
        return ret


PipelineHooksType_co = TypeVar(
    "PipelineHooksType_co",
    bound=PipelineHooks,
    covariant=True,
)


class PipelineHooksConfig(ClassConfig[PipelineHooksType_co]):
    pass
