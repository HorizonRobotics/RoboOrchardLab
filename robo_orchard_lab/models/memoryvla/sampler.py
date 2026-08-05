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

"""Episode-ordered batch sampler for MemoryVLA's ``stream`` memory.

The host's default ``DistributedBatchFlagSampler`` draws a full random
permutation (dataset_wrapper.py:133). That is fine for a single-frame model
and useless for a memory bank: with shuffled frames the bank is keyed by an
episode it sees once and then never again, so it retrieves nothing and the
method silently degenerates to an identity-ish transform.

This sampler instead shuffles *episodes* and walks each one forward in time,
emitting ``batch_size`` consecutive frames of a single episode per batch --
the input both bank modes assume. ``stream`` carries the bank across batches
and ``group`` wipes it at every training call, so they differ in how far
memory reaches, not in whether they need episode-ordered batches. Neither
works without them.

It wraps rather than modifies the host sampler: the host's own sampler is
untouched, and this one is selected from config.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler

__all__ = [
    "MemoryVLAEpisodeStreamBatchSampler",
    "assert_episode_stream_wired",
]

logger = logging.getLogger(__name__)


def _episode_spans(dataset) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` global index spans, one per episode.

    Uses ``_get_indices``, which every LMDB manipulation dataset here exposes
    and which maps a global index to ``(lmdb_index, episode_index,
    step_index)`` without touching image data
    (robodojo_lmdb_dataset.py:152). Frames of one episode are contiguous and
    step-ordered in the global index -- verified on swap_T, where indices
    0,1,2 map to steps 0,1,2 of episode 0.
    """
    subsets = getattr(dataset, "datasets", None)
    if subsets is None:
        subsets, offsets = [dataset], [0]
    else:
        offsets, acc = [], 0
        for d in subsets:
            offsets.append(acc)
            acc += len(d)

    spans: list[tuple[int, int]] = []
    for sub, offset in zip(subsets, offsets):
        get = getattr(sub, "_get_indices", None)
        if get is None:
            raise TypeError(
                f"{type(sub).__name__} has no `_get_indices`, so episode "
                "boundaries cannot be derived. MemoryVLA's memory needs "
                "episode-ordered batches in either bank mode, so use a "
                "dataset that exposes it. Switching dataloader_type does not "
                "help: `group` needs the same ordering, it just stops "
                "carrying memory between batches."
            )
        n = len(sub)
        prev_key = None
        start = 0
        for i in range(n):
            lmdb_i, ep_i, _step = get(i)
            key = (lmdb_i, ep_i)
            if prev_key is None:
                prev_key = key
            elif key != prev_key:
                spans.append((offset + start, offset + i))
                start = i
                prev_key = key
        if n:
            spans.append((offset + start, offset + n))
    return spans


class MemoryVLAEpisodeStreamBatchSampler(Sampler[list[int]]):
    """Yield batches of consecutive frames drawn from a single episode.

    Two things here exist only because of what happens downstream, and both
    are no-ops at ``num_replicas == 1``:

    **Every batch is emitted num_replicas times.** ``accelerator.prepare()``
    wraps whatever is in ``batch_sampler`` in ``BatchSamplerShard``
    (accelerate/data_loader.py:1252), unconditionally and with no opt-out,
    and that wrapper keeps ``batches[process_index::num_processes]``. This
    sampler has already sharded -- by episode, which is the whole point,
    since a batch-level shard cuts episodes in half -- so the two compose
    into a shard of a shard: measured at two ranks, each rank saw 24 of 96
    frames, the union covered 48, and rank 0's stream through episode 0 ran
    ``[0,1,2,3]`` then ``[8,9,10,11]``, a hole where ``[4,5,6,7]`` should be.
    At 8 GPUs that is 1/64 of the data, and nothing about it is visible in
    the loss. Emitting each batch N times makes the downstream stride an
    identity. ``assert_episode_stream_wired`` refuses to start if that
    wrapper turns out not to be there, because then this would hand every
    batch out N times for real.

    **Batch counts are equalised across ranks.** Sharding by episode gives
    ranks unequal totals (measured: 41007 vs 40999 on the RoboDojo set). The
    host trainer carries a standing TODO about exactly that
    (hook_based_trainer.py:412: "If the dataloader has a different number of
    batches, the training loop may hang or produce unexpected results"), and
    100k steps over ~41k batches per rank crosses two epoch boundaries, so
    this sampler must not be the thing that walks into it. Every rank can
    compute every other rank's total from the same span list, so the common
    minimum is taken locally -- no collective, no backend or device
    assumptions at construction time. The cost is the tail of the
    longer-shard ranks: at 600 episodes over 16 ranks, at most one episode
    each, re-drawn every epoch because the episode order is reshuffled.

    Args:
        data_source: the training dataset.
        batch_size: frames per batch. All of them come from one episode.
        drop_last: drop an episode's trailing partial batch. Default True,
            matching the host dataloader, and it also keeps every batch a
            uniform size for the bank.
        seed: base seed for the episode shuffle.
        num_replicas / rank: distributed sharding, by episode rather than by
            frame, so an episode never straddles two ranks.
        allow_partial_episode_batches: keep the trailing partial batch. Off by
            default; turning it on makes batch sizes ragged.
    """

    #: How many times ``__iter__`` hands out each batch, == num_replicas.
    #: A class default rather than only an instance attribute so that a
    #: half-built stub (``object.__new__``, as several tests use) reads as
    #: single-process instead of raising. ``__init__`` always overrides it,
    #: so a real instance can never fall back to this by accident.
    _emit_repeat: int = 1

    def __init__(
        self,
        data_source,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        allow_partial_episode_batches: bool = False,
    ):
        super().__init__(data_source)
        if num_replicas is None or rank is None:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas, rank = 1, 0

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last and not allow_partial_episode_batches
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self._epoch = 0

        spans = _episode_spans(data_source)
        # shard by episode; an episode must not straddle ranks
        self.spans = spans[rank::num_replicas]
        # Every rank sees the same `spans`, so every rank can price every
        # other rank's shard without talking to it. See the class docstring
        # for why they have to end up equal.
        per_rank = [
            sum(self._batches_in(e - s) for s, e in spans[r::num_replicas])
            for r in range(num_replicas)
        ]
        self._num_batches_local = per_rank[rank]
        self._num_batches = min(per_rank)
        # accelerate's BatchSamplerShard will take every num_replicas-th
        # batch back out; see the class docstring.
        self._emit_repeat = num_replicas
        logger.info(
            "MemoryVLAEpisodeStreamBatchSampler: %d episodes total, %d on "
            "rank %d/%d, %d batches of %d (own shard yields %d, truncated to "
            "the per-rank minimum so DDP cannot deadlock at an epoch "
            "boundary; each emitted %dx for BatchSamplerShard to undo)",
            len(spans), len(self.spans), rank, num_replicas,
            self._num_batches, batch_size, self._num_batches_local,
            self._emit_repeat,
        )
        if not self.spans:
            raise ValueError(
                "no episodes on this rank -- the dataset yielded no episode "
                "spans, so stream memory would never accumulate."
            )
        dropped = self._num_batches_local - self._num_batches
        if dropped and dropped > 0.05 * self._num_batches_local:
            logger.warning(
                "MemoryVLAEpisodeStreamBatchSampler: equalising rank batch "
                "counts drops %d of this rank's %d batches per epoch (%.1f%%) "
                "-- %d episodes do not divide %d ways evenly enough. Per-rank "
                "totals are %s. Either use fewer ranks or accept the loss; it "
                "falls on a different set of episodes each epoch because the "
                "episode order is reshuffled.",
                dropped, self._num_batches_local,
                100.0 * dropped / self._num_batches_local,
                len(spans), num_replicas, per_rank,
            )
        if self._num_batches == 0:
            raise ValueError(
                "some rank's episode shard yields 0 batches of {} ({} per "
                "rank across {} rank(s)), so that rank would train on "
                "nothing while the others ran. Either the dataset is too "
                "small to shard this many ways, or batch_size exceeds the "
                "shortest episode.".format(batch_size, per_rank, num_replicas)
            )

    def _batches_in(self, n: int) -> int:
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def reset(self) -> None:
        self._epoch += 1

    def __len__(self) -> int:
        return self._num_batches * self._emit_repeat

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        order = rng.permutation(len(self.spans))
        emitted = 0
        for j in order:
            start, end = self.spans[j]
            # forward in time within the episode -- never shuffled
            for b in range(start, end, self.batch_size):
                batch = list(range(b, min(b + self.batch_size, end)))
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if emitted >= self._num_batches:
                    self._epoch += 1
                    return
                for _ in range(self._emit_repeat):
                    yield batch
                emitted += 1
        self._epoch += 1


def _sampler_chain(dataloader) -> list:
    """Every sampler object the loader delegates to, outermost first.

    ``accelerator.prepare()`` can replace ``batch_sampler`` with a
    ``BatchSamplerShard`` wrapper (accelerate/data_loader.py:239), so the
    sampler that was constructed is not necessarily the one being iterated.
    Only the unwrapped chain answers "what is training actually reading".
    """
    chain = []
    cur = getattr(dataloader, "batch_sampler", None)
    if cur is None:
        cur = getattr(dataloader, "sampler", None)
    for _ in range(8):
        if cur is None:
            break
        chain.append(cur)
        nxt = getattr(cur, "batch_sampler", None)
        if nxt is None:
            nxt = getattr(cur, "sampler", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return chain


def _effective_batch_size(chain, config: dict):
    """How many samples the memory module will be handed per call.

    Read off the sampler that produces the batches, not off the config key:
    ``prepare()`` may re-wrap, and the object is what runs. By the time this
    is called the episode sampler is guaranteed to be in the chain -- the
    check above raises otherwise -- so it is the first preference. The
    fallbacks exist so that a future wrapper cannot quietly turn this check
    into a no-op; if all of them miss, the caller says so out loud rather
    than skipping in silence.

    Returns (batch_size, where_it_came_from), or (None, None).
    """
    for pref in (True, False):
        for s in chain:
            if pref and not isinstance(s, MemoryVLAEpisodeStreamBatchSampler):
                continue
            bs = getattr(s, "batch_size", None)
            if isinstance(bs, int) and not isinstance(bs, bool):
                return bs, "{}.batch_size".format(type(s).__name__)
    bs = config.get("batch_size")
    if isinstance(bs, int) and not isinstance(bs, bool):
        return bs, "config['batch_size']"
    return None, None


def _assert_shard_composes(chain) -> None:
    """Check the downstream shard is exactly the one the sampler compensates.

    The sampler emits every batch ``num_replicas`` times because
    ``BatchSamplerShard`` keeps ``batches[process_index::num_processes]``,
    turning the pair into an identity. That trade only holds if the wrapper
    is there and is striding by the same N. If it is absent, every batch is
    handed out N times for real; if N disagrees, the stream is silently
    resampled. Both are invisible in the loss, so both raise.

    Single-process runs are unaffected: accelerate still wraps, but at N=1
    the repeat and the stride are each an identity, and a missing wrapper is
    equally harmless -- so the check only bites above one rank.
    """
    ours = next(
        (
            s
            for s in chain
            if isinstance(s, MemoryVLAEpisodeStreamBatchSampler)
        ),
        None,
    )
    if ours is None or ours._emit_repeat <= 1:
        return

    outer = [s for s in chain if type(s).__name__ == "BatchSamplerShard"]
    names = [type(s).__name__ for s in chain]
    if len(outer) != 1:
        raise RuntimeError(
            "MemoryVLAEpisodeStreamBatchSampler emits every batch {}x so that "
            "accelerate's BatchSamplerShard can stride it back to 1x, but the "
            "post-prepare() chain is {} -- {} BatchSamplerShard in it. "
            "Without exactly one, training would see every batch {} times "
            "over, with no error and a normal-looking loss.".format(
                ours._emit_repeat,
                names,
                "no" if not outer else "{} of them".format(len(outer)),
                ours._emit_repeat,
            )
        )

    shard = outer[0]
    n = getattr(shard, "num_processes", None)
    if n != ours._emit_repeat or getattr(shard, "split_batches", False):
        raise RuntimeError(
            "MemoryVLAEpisodeStreamBatchSampler sharded for {} replicas and "
            "emits each batch that many times, but the BatchSamplerShard "
            "above it has num_processes={!r}, split_batches={!r}. The two "
            "must be the same plain stride or the episode stream is resampled "
            "without a word.".format(
                ours._emit_repeat,
                n,
                getattr(shard, "split_batches", None),
            )
        )


def assert_episode_stream_wired(config: dict, dataloader) -> None:
    """Raise unless episode-ordered batches are what the trainer will iterate.

    `episode_stream_sampler` shipped as a config key that nothing read: the
    switch was on, the sampler existed, and training still ran the host's
    random-permutation sampler, under which the memory banks are an exact
    identity and 7.47M parameters never receive a gradient -- with no error,
    no warning and a normal-looking loss. This makes that state impossible to
    reach silently.

    The criterion is the object, not the keys. An earlier version asked
    whether ``dataloader_type`` and ``episode_stream_sampler`` agreed, on the
    premise that the sampler was only meaningful for ``stream``. That premise
    is false -- ``dataloader_type`` selects how far memory reaches, and both
    values need the same episode-ordered input -- and it left exactly one
    unguarded cell: ``group`` with the sampler off passed every check while
    reproducing the original failure verbatim. Asking what is actually in the
    sampler chain covers every combination of keys, including ones nobody has
    invented yet.

    A second, independent question is asked once the chain is satisfied: can
    this configuration hold memory at all? Under ``group`` the answer is no
    when ``min(group_size, batch_size) == 1``, and that is decidable from the
    config plus the batch size, with no forward required. It used to be left
    to the consumer-side watchdog, which needs K forwards before it can rule
    -- so any run shorter than K passed silently while degenerating.

    Call after the trainer exists, so the check sees the post-``prepare()``
    dataloader rather than the one that was handed in.
    """
    mv = config.get("memoryvla") or {}
    if not mv.get("enable", False):
        return

    stream_sampler = mv.get("episode_stream_sampler", False)
    dl_type = mv.get("dataloader_type", "stream")
    group_size = mv.get("group_size", 16)

    if config.get("dataset_sample_weights"):
        raise RuntimeError(
            "memoryvla.enable=True cannot be combined with "
            "dataset_sample_weights={!r}: the memory needs episode-ordered "
            "batches, MemoryVLAEpisodeStreamBatchSampler is what produces "
            "them, and it takes no such parameter -- so the weights would be "
            "dropped without a word. Drop the weights, or leave the memory "
            "off for this run.".format(config.get("dataset_sample_weights"))
        )

    # dataloader is None on the --eval_only path, where there is no training
    # sampler to check at all.
    if dataloader is None:
        return

    chain = _sampler_chain(dataloader)
    if not any(
        isinstance(s, MemoryVLAEpisodeStreamBatchSampler) for s in chain
    ):
        raise RuntimeError(
            "memoryvla.enable=True (dataloader_type={!r}, "
            "episode_stream_sampler: {!r}) but the trainer is iterating {} -- "
            "MemoryVLAEpisodeStreamBatchSampler is not in the chain. "
            "Episode-ordered batches are what lets the memory bank "
            "accumulate; without them every retrieval finds an empty history, "
            "every fusion degenerates to an exact identity, and the module "
            "trains nothing while the loss looks perfectly normal. "
            "Both bank modes need those batches -- `stream` carries memory "
            "across batches, `group` only within one -- so set "
            "memoryvla.episode_stream_sampler=True. Turning it off is never "
            "the fix; it is how this state is reached.".format(
                dl_type,
                stream_sampler,
                [type(s).__name__ for s in chain] or "no sampler at all",
            )
        )

    _assert_shard_composes(chain)

    # The sampler is wired. That still leaves configurations in which memory
    # is impossible by construction, and `group` is where they live: it calls
    # bank.clear() at the top of every training call (memory_bank.py:361) and
    # clears the previous group every `group_size` samples (memory_bank.py:374;
    # with episode-ordered batches that previous group is the SAME episode).
    # So memory reaches exactly min(group_size, batch_size) samples, and at 1
    # no sample ever has a predecessor to read from.
    #
    # Decided here rather than left to the consumer-side watchdog because it is
    # decidable here: it follows from two config values and the batch size,
    # with no forward required. The watchdog needs K forwards before it can
    # rule, so a run shorter than K -- 4 to 8 steps, which is this project's
    # own smoke length -- got no protection at all. A consequence check is a
    # backstop for what cannot be decided statically, not the only line.
    if dl_type == "group":
        batch_size, bs_source = _effective_batch_size(chain, config)
        if batch_size is None:
            logger.warning(
                "MemoryVLAMemory: could not read an effective batch size from "
                "the sampler chain %s or from config['batch_size'], so the "
                "`group` memory-span check is being skipped. It is the check "
                "that rejects min(group_size, batch_size) == 1, under which "
                "the memory degenerates to an exact identity. The "
                "consumer-side check in wrapper.py still covers this on the "
                "first training forward.",
                [type(s).__name__ for s in chain],
            )
        elif min(group_size, batch_size) <= 1:
            raise RuntimeError(
                "memoryvla.enable=True with dataloader_type='group', but this "
                "configuration cannot hold any memory at all. `group` clears "
                "the bank at the top of every training call and again every "
                "group_size samples within the batch, so its memory reaches "
                "min(group_size, batch_size) = min({}, {}) = {} sample(s). At "
                "1, no sample ever has a predecessor to read: every retrieval "
                "finds an empty history, every fusion reduces to an exact "
                "identity, and 7.47M parameters receive no gradient while the "
                "loss looks perfectly normal.\n"
                "Observed: dataloader_type='group', group_size={}, "
                "batch_size={} (read from {}), episode_stream_sampler={!r}.\n"
                "Two ways out, both effective here:\n"
                "  * dataloader_type='stream' -- it carries the bank across "
                "calls, so batch_size=1 is a perfectly good configuration "
                "there; this is also the episode-level memory the paper "
                "describes.\n"
                "  * keep 'group' but use batch_size >= 2 AND group_size >= 2 "
                "-- memory then spans min(group_size, batch_size) frames "
                "inside each batch, and nothing beyond it.\n"
                "The episode sampler is NOT the problem here: it is wired, it "
                "is in the chain, and the batches it produces are "
                "episode-contiguous. Changing episode_stream_sampler will not "
                "affect this.".format(
                    group_size,
                    batch_size,
                    min(group_size, batch_size),
                    group_size,
                    batch_size,
                    bs_source,
                    stream_sampler,
                )
            )
