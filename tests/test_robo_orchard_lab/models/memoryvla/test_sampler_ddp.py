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

"""What the episode sampler does once there is more than one rank.

Every recorded run of this port was single-process, and accelerate does not
wrap a batch_sampler at one process -- so the shard-of-a-shard below was
reachable for as long as the sampler has existed without any evidence able to
show it. Measured at two ranks before the fix: each rank consumed 24 of 96
frames, the union covered 48, and rank 0 read episode 0 as [0,1,2,3] then
[8,9,10,11].

These tests reproduce accelerate's ``BatchSamplerShard`` with the one line of
its behaviour that matters (``batches[process_index::num_processes]``) rather
than importing it, so they state the contract this sampler is written
against instead of asking accelerate whether it agrees with itself.

CPU only, no dataset, no GPU, no process group.
"""

import pytest

from robo_orchard_lab.models.memoryvla.sampler import (
    MemoryVLAEpisodeStreamBatchSampler,
    _assert_shard_composes,
)

FRAMES = 12
BATCH = 4


class EpisodeDataset:
    """N episodes of equal length; global index in, (lmdb, ep, step) out."""

    def __init__(self, n_episodes, frames=FRAMES):
        self.n_episodes = n_episodes
        self.frames = frames

    def __len__(self):
        return self.n_episodes * self.frames

    def _get_indices(self, i):
        return 0, i // self.frames, i % self.frames


def build(n_episodes, world, rank, frames=FRAMES):
    return MemoryVLAEpisodeStreamBatchSampler(
        EpisodeDataset(n_episodes, frames),
        BATCH,
        drop_last=True,
        num_replicas=world,
        rank=rank,
    )


def shard(batches, process_index, num_processes):
    """Reproduce the one line of BatchSamplerShard that matters here.

    accelerate/data_loader.py, with split_batches=False.
    """
    return batches[process_index::num_processes]


def consumed(n_episodes, world, frames=FRAMES):
    """Global indices each rank ends up reading, after the outer shard."""
    out = []
    for rank in range(world):
        s = build(n_episodes, world, rank, frames)
        batches = shard(list(s), rank, world)
        out.append(batches)
    return out


# -- the shard-of-a-shard ----------------------------------------------------
def test_ranks_together_cover_every_frame():
    per_rank = consumed(8, world=2)
    seen = {i for r in per_rank for b in r for i in b}
    assert seen == set(range(8 * FRAMES))


def test_ranks_never_read_the_same_frame():
    a, b = consumed(8, world=2)
    assert {i for x in a for i in x}.isdisjoint({i for x in b for i in x})


def test_episode_stream_has_no_holes_after_the_outer_shard():
    """The regression that had no test: rank 0 read [0..3] then [8..11]."""
    for rank_batches in consumed(8, world=2):
        by_episode = {}
        for batch in rank_batches:
            by_episode.setdefault(batch[0] // FRAMES, []).extend(batch)
        for episode, frames in by_episode.items():
            base = episode * FRAMES
            assert frames == list(range(base, base + len(frames))), (
                f"episode {episode} came through as {frames}"
            )


def test_an_episode_never_straddles_two_ranks():
    a, b = consumed(8, world=2)
    eps_a = {x[0] // FRAMES for x in a}
    eps_b = {x[0] // FRAMES for x in b}
    assert eps_a.isdisjoint(eps_b)


# -- single process must be untouched ----------------------------------------
def test_one_rank_is_bit_for_bit_the_old_behaviour():
    s = build(8, world=1, rank=0)
    batches = list(s)
    assert len(batches) == 8 * (FRAMES // BATCH)
    assert {i for b in batches for i in b} == set(range(8 * FRAMES))


def test_one_rank_emits_each_batch_once():
    s = build(8, world=1, rank=0)
    batches = list(s)
    assert len(batches) == len(s)
    assert len({tuple(b) for b in batches}) == len(batches)


# -- rank evenness -----------------------------------------------------------
def test_every_rank_yields_the_same_number_of_batches():
    """hook_based_trainer.py:412 warns uneven counts may hang the loop."""
    for world in (2, 3, 4):
        counts = {len(x) for x in consumed(9, world=world)}
        assert len(counts) == 1, f"world={world} gave {counts}"


def test_len_matches_what_is_yielded():
    for world in (1, 2, 3):
        for rank in range(world):
            s = build(9, world, rank)
            assert len(list(s)) == len(s)


def test_uneven_shards_truncate_rather_than_duplicate():
    """9 episodes over 2 ranks: 5 and 4. Nothing may be read twice."""
    for rank_batches in consumed(9, world=2):
        flat = [i for b in rank_batches for i in b]
        assert len(flat) == len(set(flat))


def test_a_rank_with_no_batches_is_refused():
    """Silently training one rank on nothing is the failure to avoid."""
    with pytest.raises(ValueError, match="0 batches"):
        build(3, world=4, rank=0)


# -- the guard that keeps the trade honest ------------------------------------
class FakeShard:
    def __init__(self, num_processes, split_batches=False):
        self.num_processes = num_processes
        self.split_batches = split_batches


FakeShard.__name__ = "BatchSamplerShard"


def test_guard_passes_when_the_shard_matches():
    s = build(8, world=2, rank=0)
    _assert_shard_composes([FakeShard(2), s])


def test_guard_is_silent_at_one_rank_with_no_shard():
    s = build(8, world=1, rank=0)
    _assert_shard_composes([s])


def test_guard_rejects_a_missing_shard():
    s = build(8, world=2, rank=0)
    with pytest.raises(RuntimeError, match="BatchSamplerShard"):
        _assert_shard_composes([s])


def test_guard_rejects_a_shard_striding_by_something_else():
    s = build(8, world=2, rank=0)
    with pytest.raises(RuntimeError, match="num_processes"):
        _assert_shard_composes([FakeShard(4), s])


def test_guard_rejects_split_batches():
    s = build(8, world=2, rank=0)
    with pytest.raises(RuntimeError, match="split_batches"):
        _assert_shard_composes([FakeShard(2, split_batches=True), s])
