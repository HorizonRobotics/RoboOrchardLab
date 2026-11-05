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

"""Source-selection policies for worker-local iterable dataset mixtures."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from robo_orchard_core.utils.config import ClassConfig, ClassType

__all__ = [
    "MixtureSamplingView",
    "MixtureSamplingStrategy",
    "MixtureSamplingStrategyConfig",
    "DefaultMixtureSamplingStrategy",
    "DefaultMixtureSamplingStrategyConfig",
]


@dataclass(frozen=True, slots=True)
class MixtureSamplingView:
    """Read source metadata and row progress without owning readers.

    Physical IDs index ``targets`` and ``logical_indices``. Logical IDs index
    ``logical_names``. Counts are output rows after worker sharding,
    resampling, and dataset-side drop-last. ``remaining_targets`` and
    ``segment_remaining`` are read-only live mappings owned by the framework;
    copy their values when retaining a historical observation. Candidate
    tuples and admission metadata refresh when the resident pool changes.

    ``residence_rows`` describes the next residency of pending sources without
    exposing a reader, index plan, or mutable cursor. Strategies may only
    select IDs from the candidates for the current decision.
    """

    pending_ids: tuple[int, ...]
    active_ids: tuple[int, ...]
    logical_indices: tuple[int, ...]
    targets: tuple[int, ...]
    remaining_targets: Mapping[int, int]
    residence_rows: Mapping[int, int]
    segment_remaining: Mapping[int, int]
    batch_size: int
    logical_names: tuple[str | None, ...]
    initial_residency_slots: int = 0
    """Initial finite reader slots; zero after filling or without offload."""


class MixtureSamplingStrategy(ABC):
    """Choose mixture sources within framework-owned quotas and lifecycles.

    Subclass all three methods and select the runtime through a
    :class:`MixtureSamplingStrategyConfig` ``class_type``. The dataset creates
    a fresh runtime per worker and complete iteration, using
    ``config(generator=generator, view=view)``. Only the config is retained on
    the dataset or serialized. Strategies never open readers or update row
    counters; the framework validates their selected physical IDs.

    Args:
        cfg (MixtureSamplingStrategyConfig): Declarative runtime configuration.
        generator (torch.Generator | np.random.Generator): Existing mixture
            RNG, already scoped by the framework to epoch, rank and worker.
            Sharing this stream preserves ordering with row sampling.
        view (MixtureSamplingView): Initial read-only metadata and progress.
    """

    InitFromConfig = True

    def __init__(
        self,
        cfg: MixtureSamplingStrategyConfig,
        *,
        generator: torch.Generator | np.random.Generator,
        view: MixtureSamplingView,
    ) -> None:
        self.cfg = cfg
        self._generator = generator

    @abstractmethod
    def choose_admission(self, view: MixtureSamplingView) -> int:
        """Return one pending physical ID to occupy a free reader slot."""

    @abstractmethod
    def choose_active(self, view: MixtureSamplingView) -> int:
        """Return one active physical ID for its next sample or inner batch."""

    @abstractmethod
    def residency_completed(self, item_id: int, *, requeue: bool) -> None:
        """Observe successful EOF, owned cleanup and progress commit.

        ``requeue=True`` means this unfinished source will become pending
        again. ``False`` means its worker-local output is complete. Failure
        and early close never produce this notification.
        """

    def initial_residency_rows(
        self, item_id: int, *, rows: int, batch_size: int
    ) -> int:
        """Return the raw first-segment budget for an initial resident.

        Called only for positive-output sources in the first shuffled offload
        pool, before allocation or reader creation. ``rows`` includes the
        original cap/tail merge. Retain it unchanged, or return a positive
        smaller whole-inner-batch budget. If ``rows < batch_size``, retain it.
        Validation, output rounding, slicing and EOF belong to the framework.
        """
        return rows


class MixtureSamplingStrategyConfig(ClassConfig[MixtureSamplingStrategy]):
    """Serialize a source-selection runtime and its replay capability.

    Subclasses can add policy fields using the standard ``ClassConfig``
    convention. Set ``supports_epoch_replay`` only when recreating the runtime
    from this config and the framework RNG reconstructs every source decision;
    the default ``False`` leaves custom recovery capability unknown.
    """

    class_type: ClassType[MixtureSamplingStrategy]
    supports_epoch_replay: bool = False


class DefaultMixtureSamplingStrategy(MixtureSamplingStrategy):
    """Use live remaining rows and stagger initial reader residencies.

    Bootstrap chooses a quota-weighted logical anchor, then uniform uncovered
    groups. Normal admission uses each eligible group's remaining rows divided
    by its pending mean residency; physical members share that admission by
    remaining rows. Active service keeps its existing logical/physical weights.
    Only shuffled offload prefetches initial budget uniforms, before admission.
    """

    def __init__(
        self,
        cfg: MixtureSamplingStrategyConfig,
        *,
        generator: torch.Generator | np.random.Generator,
        view: MixtureSamplingView,
    ) -> None:
        super().__init__(cfg, generator=generator, view=view)
        slots = view.initial_residency_slots
        if isinstance(generator, np.random.Generator):
            self._initial_uniforms = (
                generator.random(slots).tolist() if slots else []
            )
        elif isinstance(generator, torch.Generator):
            self._initial_uniforms = (
                torch.rand(
                    slots, dtype=torch.float64, generator=generator
                ).tolist()
                if slots
                else []
            )
        else:
            raise ValueError(
                "Generator must be either a torch.Generator "
                "or a numpy.random.Generator."
            )
        self._initial_uniform_index = 0
        self._logical_indices = np.asarray(
            view.logical_indices, dtype=np.int64
        )
        targets = np.zeros(len(view.logical_names), dtype=np.int64)
        np.add.at(
            targets,
            self._logical_indices,
            np.asarray(view.targets, dtype=np.int64),
        )
        total = int(targets.sum())
        self._logical_probabilities = (
            targets / total
            if total
            else np.zeros(len(targets), dtype=np.float64)
        )
        self._anchor_selected = False
        self._bootstrap_ended = False
        self._weights = np.empty(0, dtype=np.float64)
        self._weight_units: tuple[int, ...] = ()
        self._weights_dirty = True

    def choose_admission(self, view: MixtureSamplingView) -> int:
        """Choose a logical group, then a member by remaining rows."""
        self._weights_dirty = True
        members: dict[int, list[int]] = {}
        for idx in view.pending_ids:
            if view.residence_rows[idx] == 0:
                return idx
            if view.remaining_targets[idx] > 0:
                members.setdefault(view.logical_indices[idx], []).append(idx)
        if not members:
            raise ValueError("No positive pending physical source.")
        active_groups = {
            view.logical_indices[idx]
            for idx in view.active_ids
            if view.segment_remaining[idx] > 0
            and view.remaining_targets[idx] > 0
        }
        uncovered = sorted(
            group for group in members if group not in active_groups
        )
        eligible = uncovered or sorted(members)
        if not self._bootstrap_ended and not self._anchor_selected:
            eligible = sorted(members)
            mass = self._logical_probabilities[eligible]
        elif not self._bootstrap_ended and uncovered:
            mass = np.ones(len(eligible), dtype=np.float64)
        else:
            remaining = np.zeros(len(view.logical_names), dtype=np.int64)
            np.add.at(
                remaining,
                self._logical_indices,
                np.fromiter(
                    (
                        view.remaining_targets[idx]
                        for idx in range(len(view.targets))
                    ),
                    dtype=np.int64,
                    count=len(view.targets),
                ),
            )
            means = np.asarray(
                [
                    sum(
                        view.remaining_targets[idx] * view.residence_rows[idx]
                        for idx in members[group]
                    )
                    / sum(
                        view.remaining_targets[idx] for idx in members[group]
                    )
                    for group in eligible
                ],
                dtype=np.float64,
            )
            mass = (remaining[eligible] / int(remaining.sum())) / means
        probabilities = np.asarray(mass, dtype=np.float64) / np.sum(mass)
        if not np.all(np.isfinite(probabilities)) or not np.all(
            probabilities > 0
        ):
            raise ValueError(
                "Eligible admission probabilities must be finite and positive."
            )
        group = eligible[self._sample_index(probabilities)]
        candidates = members[group]
        remaining = np.asarray(
            [view.remaining_targets[idx] for idx in candidates],
            dtype=np.float64,
        )
        selected = candidates[self._sample_index(remaining / remaining.sum())]
        self._anchor_selected = True
        return selected

    def initial_residency_rows(
        self, item_id: int, *, rows: int, batch_size: int
    ) -> int:
        """Use the next initial uniform without drawing during materialization.

        A shortened budget contains whole inner batches. Sources shorter than
        one batch retain their original raw budget. Later admissions never
        consume unused uniforms from an incompletely filled initial pool.
        """
        if not self._initial_uniforms:
            return rows
        if self._initial_uniform_index >= len(self._initial_uniforms):
            raise RuntimeError("Initial residency budget draws are exhausted.")
        uniform = self._initial_uniforms[self._initial_uniform_index]
        self._initial_uniform_index += 1
        if rows < batch_size:
            return rows
        return min(
            rows, batch_size * (1 + int(uniform * (rows // batch_size)))
        )

    def choose_active(self, view: MixtureSamplingView) -> int:
        """Draw active service using the original weight-refresh boundaries."""
        self._bootstrap_ended = True
        units = tuple(
            min(view.batch_size, view.segment_remaining[idx])
            for idx in view.active_ids
        )
        if self._weights_dirty or units != self._weight_units:
            logical_remaining = [0] * len(view.logical_names)
            for idx, remaining in view.remaining_targets.items():
                logical_remaining[view.logical_indices[idx]] += remaining
            active_remaining: dict[int, int] = {}
            for idx in view.active_ids:
                logical_idx = view.logical_indices[idx]
                active_remaining[logical_idx] = (
                    active_remaining.get(logical_idx, 0)
                    + view.remaining_targets[idx]
                )
            self._weights = np.array(
                [
                    logical_remaining[view.logical_indices[idx]]
                    * view.remaining_targets[idx]
                    / (active_remaining[view.logical_indices[idx]] * unit)
                    for idx, unit in zip(view.active_ids, units, strict=True)
                ],
                dtype=np.float64,
            )
            self._weights /= self._weights.sum()
            self._weight_units = units
            self._weights_dirty = False
        return view.active_ids[self._sample_index(self._weights)]

    def residency_completed(self, item_id: int, *, requeue: bool) -> None:
        """Invalidate active weights after successful EOF and owned cleanup."""
        self._weights_dirty = True

    def _sample_index(self, weights: np.ndarray) -> int:
        """Consume the shared mixture RNG for one categorical choice."""
        if isinstance(self._generator, np.random.Generator):
            return int(self._generator.choice(len(weights), p=weights))
        if isinstance(self._generator, torch.Generator):
            return int(
                torch.multinomial(
                    torch.from_numpy(weights), 1, generator=self._generator
                ).item()
            )
        raise ValueError(
            "Generator must be either a torch.Generator "
            "or a numpy.random.Generator."
        )


class DefaultMixtureSamplingStrategyConfig(MixtureSamplingStrategyConfig):
    """Use remaining-row admission and initial jitter with epoch replay."""

    class_type: ClassType[DefaultMixtureSamplingStrategy] = (
        DefaultMixtureSamplingStrategy
    )
    supports_epoch_replay: bool = True
