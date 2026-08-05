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

"""Glue between HoloBrain's feature tensors and MemoryVLA's memory banks.

Sits in ``HoloBrain_Qwen2_5_VL._forward`` between ``_vlm_outputs_handler`` and
the decoder. Shapes in equal shapes out, so the decoder, the spatial enhancer
and every loss below are untouched.

Two streams, mapped onto what this host actually has:

* perceptual -- ``feature_maps[0]`` ``[B, cams, C, h, w]``, flattened to
  ``[B, cams*h*w, C]``. Note these are post-VLM features, whereas MemoryVLA
  reads pre-LLM vision-backbone patches. Equivalent in role, not in content.
* cognitive -- the last valid token of ``text_dict["embedded"]``, giving
  ``[B, 1, C]``, matching MemoryVLA's single cognition token. Picking one
  token also sidesteps padding: ``CrossTransformerBlock`` takes no attention
  mask, so handing it padded history would silently corrupt retrieval.

``BottleneckSE`` is ported and reference-checked but deliberately NOT wired in
here. MemoryVLA uses it to squeeze ~2176-dim DINO+SigLIP features down to 256
for a separate DiT stream; this host's features are already ``embed_dims`` and
must stay that width to preserve the decoder contract. Compressing here would
break that contract, so it stays available and verified rather than wired.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch
from torch import nn

from robo_orchard_lab.models.memoryvla.memory_bank import (
    CogMemBank,
    PerMemBank,
)

__all__ = ["MemoryVLAMemory"]

# The identity probe below fails under this. 1e-5 sits far above float32
# rounding (the measured degenerate gap was 1-2 ULP, ~6e-08) and far below
# a working bank (measured ~1.2 with the episode sampler).
IDENTITY_TOL = 1e-5

logger = logging.getLogger(__name__)


class MemoryVLAMemory(nn.Module):
    """Perceptual + cognitive memory over HoloBrain's VLM features.

    Args:
        token_size: channel width of the features, i.e. ``decoder.embed_dims``.
        use_perceptual: run the perceptual bank over ``feature_maps[0]``.
        use_cognitive: run the cognitive bank over the last valid text token.
        dataloader_type: ``stream`` keeps the bank across calls and only drops
            an episode when it changes -- the episode-level memory the paper
            describes. ``group`` wipes the bank every training call, so memory
            never spans a batch. ``stream`` unless you are reproducing the
            original default, and it requires an episode-ordered batch
            sampler (see ``MemoryVLAEpisodeStreamBatchSampler``).
        group_size: only meaningful for ``group``.
        mem_length: entries kept per episode before consolidation kicks in.
        retrieval_layers: number of cross-attention blocks.
        use_timestep_pe: add a sinusoidal embedding of the frame index to the
            retrieved history. Requires ``step_index`` in the batch.
        fusion_type: ``gate`` or ``add``.
        consolidate_type: ``tome`` (merge the most similar adjacent pair) or
            ``fifo``.
        update_fused: store the fused feature instead of the raw one.
        episode_id_key: batch key holding a per-sample episode identifier.
            ``uuid`` is globally unique here, so it needs no disambiguation.
        timestep_key: batch key holding the frame index within the episode.
    """

    #: Training forwards to watch before ruling on bank liveness. Must be >= 2:
    #: at batch_size 1 in `stream` mode a bank only reaches 2 on the second
    #: forward, so 1 would fail a healthy run. 8 leaves margin and still stops
    #: long before a real training run has spent anything.
    #:
    #: Placed after the docstring, not before it: an assignment ahead of the
    #: string demotes the string to a plain expression statement and leaves
    #: __doc__ as None, which autoapi then renders as an empty class page. ruff
    #: ignores D101 here and does not flag string expressions as B018, so no
    #: gate catches it -- only reading the source does.
    BANK_LIVENESS_FORWARDS = 8

    def __init__(
        self,
        token_size: int,
        use_perceptual: bool = True,
        use_cognitive: bool = True,
        dataloader_type: str = "stream",
        group_size: int = 16,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = False,
        episode_id_key: str = "uuid",
        timestep_key: str = "step_index",
    ):
        super().__init__()
        if not (use_perceptual or use_cognitive):
            raise ValueError(
                "MemoryVLAMemory was built with both streams disabled. "
                "Leave `memoryvla=None` in the model config instead -- the "
                "module must not be constructed when it is off, otherwise it "
                "consumes global RNG during init and perturbs the run."
            )
        self.token_size = token_size
        self.use_perceptual = use_perceptual
        self.use_cognitive = use_cognitive
        self.use_timestep_pe = use_timestep_pe
        self.episode_id_key = episode_id_key
        self.timestep_key = timestep_key
        self.dataloader_type = dataloader_type
        # kept because the guards below reason about how far memory can
        # possibly reach, and that is min(group_size, batch_size) under
        # `group` and capped by mem_length in either mode
        self.group_size = group_size
        self.mem_length = mem_length
        # guardrail state; plain attributes, so nothing enters state_dict
        self._episode_check_done = False
        self._identity_check_done = False
        self._bank_liveness_checked = False
        self._train_forwards = 0
        self._max_bank_len_seen = 0
        self._batch_sizes_seen: set[int] = set()
        self._distinct_in_batch_seen: set[int] = set()

        bank_kwargs = dict(
            dataloader_type=dataloader_type,
            group_size=group_size,
            token_size=token_size,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            fusion_type=fusion_type,
            consolidate_type=consolidate_type,
            update_fused=update_fused,
        )
        self.per_mem_bank = PerMemBank(**bank_kwargs) if use_perceptual else None
        self.cog_mem_bank = CogMemBank(**bank_kwargs) if use_cognitive else None

        # eval-time episode tracking; training-time clearing is the banks' own
        self._last_episode_ids: Optional[tuple] = None
        #: Episode identity at inference. Training keys the bank by the
        #: dataset's ``uuid``; the deployed input has no such field
        #: (``MultiArmManipulationInput`` does not carry one and the processor
        #: never adds one), so at inference the episode boundary is whatever
        #: ``reset()`` says it is. Bumping this counter there gives the banks a
        #: key with the same two properties uuid has: constant within an
        #: episode, distinct across episodes.
        self._eval_episode = 0
        self._eval_forwards = 0
        self._eval_history_reads = 0
        self._last_timestep_seen: Optional[int] = None

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        """Drop all memory. Call at an episode boundary during inference.

        The banks only manage episodes while ``self.training`` is set
        (memory_vla.py:267), so nothing clears them at inference time unless
        the caller does it.

        This is also where an inference episode *ends*, which makes it the
        only place a consumer-side check can run at inference: the three
        training-time guards all return early when ``self.training`` is unset,
        so before this an evaluation run had no guard of any kind. A run whose
        memory never retrieved anything looks exactly like a healthy one from
        the outside -- the same failure P0-1 was about, on the one path that
        was still uncovered.
        """
        self._report_eval_episode()
        for bank in (self.per_mem_bank, self.cog_mem_bank):
            if bank is not None:
                bank.reset()
        self._last_episode_ids = None
        self._eval_episode += 1
        self._eval_forwards = 0
        self._eval_history_reads = 0
        self._last_timestep_seen = None

    def _report_eval_episode(self) -> None:
        """Say whether the episode that just ended used its memory at all."""
        if self._eval_forwards == 0:
            return
        if self._eval_history_reads == 0:
            logger.warning(
                "MemoryVLAMemory: inference episode %d ran %d forward(s) and "
                "never once retrieved history, so every fusion in it reduced "
                "to an identity and the memory contributed nothing. Either "
                "the episode was one frame long, or `%s` is not advancing "
                "across calls, or reset() is being called more often than "
                "once per episode. memory_stats() carries the counters.",
                self._eval_episode,
                self._eval_forwards,
                self.timestep_key,
            )
        else:
            logger.info(
                "MemoryVLAMemory: inference episode %d ended after %d "
                "forward(s), %d of which retrieved history.",
                self._eval_episode,
                self._eval_forwards,
                self._eval_history_reads,
            )

    def memory_stats(self) -> dict:
        """Counters an eval harness can assert on. Cheap, no side effects."""
        return {
            "eval_episode": self._eval_episode,
            "eval_forwards": self._eval_forwards,
            "eval_history_reads": self._eval_history_reads,
            "bank_lengths": {
                name: sorted(len(v) for v in bank.bank.values())
                for name, bank in (
                    ("per_mem_bank", self.per_mem_bank),
                    ("cog_mem_bank", self.cog_mem_bank),
                )
                if bank is not None and hasattr(bank, "bank")
            },
        }

    def _check_eval_episode_boundary(self, timesteps: Optional[list]) -> None:
        """Raise if a new inference episode started without ``reset()``.

        At inference the episode identity comes from ``reset()``, so a caller
        that never calls it hands every episode the same key and the bank
        retrieves across episode boundaries -- reading one task's history
        while acting in the next. Nothing downstream notices: the shapes are
        right, the loss is not computed, and the only symptom is a score.

        ``step_index`` is what makes this decidable rather than a matter of
        trust. It counts frames within an episode, so it only ever goes
        backwards when a new episode has begun; if that happens with no
        ``reset()`` in between, the caller is not doing what this module
        needs. Raising is the point -- a warning here would be read after the
        evaluation had already produced numbers.
        """
        if timesteps is None:
            return
        t = min(timesteps)
        prev = self._last_timestep_seen
        if prev is not None and t < prev:
            raise RuntimeError(
                "MemoryVLAMemory: `{}` went backwards at inference ({} -> {}) "
                "with no reset() in between, so a new episode began while the "
                "memory bank still held the previous episode's history. Every "
                "retrieval from here on reads across an episode boundary.\n"
                "The evaluation loop must call reset() on the policy at each "
                "episode start; HoloBrain's model-level reset() forwards to "
                "this module. If your loop does call it, then the object it "
                "resets is not the model being run -- check that the policy "
                "resets `self.model`/`self.pipeline` and not a copy.\n"
                "Observed: eval_episode={}, forwards in it={}, of which "
                "retrieved history={}.".format(
                    self.timestep_key,
                    prev,
                    t,
                    self._eval_episode,
                    self._eval_forwards,
                    self._eval_history_reads,
                )
            )
        self._last_timestep_seen = t

    def _autoreset_for_eval(self, episode_ids: Sequence[Any]) -> None:
        """Drop episodes that are no longer in play, during inference only."""
        current = tuple(dict.fromkeys(episode_ids))
        if self._last_episode_ids is not None:
            for eid in self._last_episode_ids:
                if eid not in current:
                    for bank in (self.per_mem_bank, self.cog_mem_bank):
                        if bank is not None:
                            bank.clear_episode(eid)
        self._last_episode_ids = current

    # -- batch field extraction ------------------------------------------
    def _episode_ids(self, inputs: dict, batch_size: int) -> list:
        ids = inputs.get(self.episode_id_key)
        if ids is None:
            if not self.training:
                # The deployed input genuinely has no episode id, and telling
                # the reader to edit a dataset config -- which is what this
                # used to say on both paths -- is an instruction that cannot
                # be carried out at inference: there is no dataset. At
                # inference the episode boundary is defined by reset(), so the
                # counter it bumps is the identity. Constant within an
                # episode, distinct across episodes: the two properties uuid
                # supplies during training.
                return [f"eval-episode-{self._eval_episode}"] * batch_size
            raise KeyError(
                f"MemoryVLAMemory needs `{self.episode_id_key}` in the batch "
                "to key its memory by episode, but the training batch does "
                "not carry it. Add it to the ItemSelection whitelist in the "
                "dataset config. (At inference this is not an error: the "
                "episode identity comes from reset() instead.)"
            )
        if torch.is_tensor(ids):
            ids = ids.tolist()
        ids = list(ids)
        if len(ids) != batch_size:
            raise ValueError(
                f"`{self.episode_id_key}` has {len(ids)} entries for a batch "
                f"of {batch_size}."
            )
        return ids

    def _timesteps(self, inputs: dict, batch_size: int) -> Optional[list]:
        if not self.use_timestep_pe:
            return None
        ts = inputs.get(self.timestep_key)
        if ts is None:
            raise KeyError(
                f"`use_timestep_pe=True` needs `{self.timestep_key}` in the "
                "batch. RoboDojoLmdbDataset produces it, but ItemSelection "
                "drops it unless the memoryvla switch is on -- check the "
                "dataset config."
            )
        # dtype is inconsistent upstream: python int on the first episode,
        # np.int64 later, which decides whether collate_batch_dict makes a
        # tensor or leaves a list. Accept both.
        if torch.is_tensor(ts):
            ts = ts.tolist()
        # At inference there is no collate at all: deploy_policy calls
        # `self.model(self.processor.pre_process(data))` directly, and
        # pre_process sets step_index to a scalar (processor.py:158). A scalar
        # is not iterable, so the comprehension below used to raise TypeError
        # on the first evaluation forward.
        if not isinstance(ts, (list, tuple)):
            ts = [ts] * batch_size
        ts = [int(x) for x in ts]
        if len(ts) != batch_size:
            raise ValueError(
                f"`{self.timestep_key}` has {len(ts)} entries for a batch of "
                f"{batch_size}."
            )
        return ts

    # -- forward ----------------------------------------------------------
    def forward(self, feature_maps, text_dict, inputs):
        fm0 = feature_maps[0]
        batch_size = fm0.shape[0]

        episode_ids = self._episode_ids(inputs, batch_size)
        timesteps = self._timesteps(inputs, batch_size)
        if not self.training:
            self._check_eval_episode_boundary(timesteps)
            self._autoreset_for_eval(episode_ids)
            self._eval_forwards += 1
            if self._history_will_be_read(episode_ids):
                self._eval_history_reads += 1

        self._check_episode_stream(episode_ids, batch_size)
        probe = (
            self.training
            and not self._identity_check_done
            and self._history_will_be_read(episode_ids)
        )
        per_in = (
            fm0.detach().clone() if probe and self.use_perceptual else None
        )
        cog_in = (
            text_dict["embedded"].detach().clone()
            if probe and self.use_cognitive
            else None
        )

        if self.use_perceptual:
            feature_maps = list(feature_maps)
            feature_maps[0] = self._forward_perceptual(
                fm0, episode_ids, timesteps
            )

        if self.use_cognitive:
            text_dict = dict(text_dict)
            text_dict["embedded"] = self._forward_cognitive(
                text_dict["embedded"],
                text_dict.get("text_token_mask"),
                episode_ids,
                timesteps,
            )

        if probe:
            self._assert_not_identity(
                per_in,
                feature_maps[0] if self.use_perceptual else None,
                cog_in,
                text_dict["embedded"] if self.use_cognitive else None,
            )

        # after the banks have written, not before: `group` clears at the top
        # of process_batch (memory_bank.py:361), so sampling on entry reads
        # the previous batch's residue.
        self._check_bank_liveness(episode_ids, batch_size)

        return feature_maps, text_dict

    # -- guardrails --------------------------------------------------------
    def _group_span(self, batch_size: int) -> int:
        """How many samples ``group`` memory can reach inside one call.

        ``group`` clears the bank at the top of every training call and clears
        the previous group every ``group_size`` samples, and with
        episode-ordered batches that previous group is the same episode -- so
        the reach is ``min(group_size, batch_size)``. At 1 nothing can ever be
        retrieved, whatever the batch looks like.
        """
        return min(self.group_size, batch_size)

    def _check_episode_stream(self, episode_ids, batch_size) -> None:
        """Log the first batch's episode spread; raise if memory is impossible.

        Once per run, not per batch: a warning on every batch is a warning
        nobody reads. Training only -- at inference, batches legitimately span
        many episodes.

        Two independent reasons to stop, in this order:

        1. the configuration cannot hold memory at all (``group`` with
           ``min(group_size, batch_size) == 1``).
           ``assert_episode_stream_wired`` decides the same thing at
           assembly time, before any data is loaded;
           this is the backstop for entry points that never call it -- it is
           called from exactly one place in ``train.py``, and a second training
           entry point exists.
        2. the batch that arrived cannot support memory: one sample per
           episode means no sample can ever read history, whatever the bank
           mode is called. This check used to return early unless
           ``dataloader_type`` was ``"stream"``, which silenced it in the one
           configuration that needed it most. A legitimate ``group`` layout
           puts ``group_size`` frames of an episode side by side, so it never
           trips this.
        """
        if self._episode_check_done or not self.training:
            return
        self._episode_check_done = True
        distinct = len(set(episode_ids))
        logger.info(
            "MemoryVLAMemory[%s]: first training batch holds %d distinct "
            "episode(s) across %d samples (group_size=%d, mem_length=%d).",
            self.dataloader_type,
            distinct,
            batch_size,
            self.group_size,
            self.mem_length,
        )
        group_span = self._group_span(batch_size)
        if self.dataloader_type == "group" and group_span <= 1:
            raise RuntimeError(
                "memoryvla is on with dataloader_type='group', but this "
                "configuration cannot hold any memory at all. `group` clears "
                "the bank at the top of every training call and again every "
                "group_size samples within the batch, so its memory reaches "
                "min(group_size, batch_size) = min({}, {}) = {} sample(s). At "
                "1, no sample ever has a predecessor to read: every retrieval "
                "finds an empty history, every fusion reduces to an exact "
                "identity, and 7.47M parameters receive no gradient while the "
                "loss looks perfectly normal.\n"
                "Observed on the first training batch: batch_size={}, "
                "distinct episodes in it={}, group_size={}, mem_length={}.\n"
                "Two ways out, both effective here:\n"
                "  * dataloader_type='stream' -- it carries the bank across "
                "calls, so batch_size=1 is a perfectly good configuration "
                "there.\n"
                "  * keep 'group' but use batch_size >= 2 AND group_size >= 2."
                "\n"
                "The episode sampler is NOT the problem here; changing "
                "episode_stream_sampler will not affect this. If this run "
                "reached a forward at all, the assembly-time check that "
                "rejects the same thing before training starts "
                "(assert_episode_stream_wired) was never called on this "
                "path.".format(
                    self.group_size,
                    batch_size,
                    group_span,
                    batch_size,
                    distinct,
                    self.group_size,
                    self.mem_length,
                )
            )
        if batch_size > 1 and distinct == batch_size:
            raise RuntimeError(
                "memoryvla is on (dataloader_type={!r}) but the first "
                "training batch has {} samples from {} different episodes. "
                "Each sample is then the only frame of its episode the bank "
                "ever sees, so there is no history to retrieve and every "
                "fusion reduces to an exact identity. Use an episode-ordered "
                "batch sampler (memoryvla.episode_stream_sampler=True); every "
                "bank mode needs one.".format(
                    self.dataloader_type, batch_size, distinct
                )
            )

    def _check_bank_liveness(self, episode_ids, batch_size) -> None:
        """Raise if, after K training forwards, no bank ever held >1 entry.

        The consumer-side counterpart to ``assert_episode_stream_wired``. That
        one asks whether the right sampler got wired and whether the
        configuration could hold memory at all; this one asks whether the
        batches actually arriving carry episode history -- something neither
        the config nor the sampler's identity can answer.

        Why bank length and not the identity probe: the probe only arms once
        history exists, and "history never exists" is precisely the failure
        mode. P0-1 and P1-B both had that shape -- every batch a different
        episode, every bank one entry long, every fusion an exact identity --
        and no probe ever armed to say so.

        What K is for, and what it is not for. K is a *time* gate: nothing is
        decided before the K-th forward, so a run shorter than K gets nothing
        from this check. That was the whole of P1-C -- `group` at batch 1 is a
        configuration in which memory is impossible, and a 4-step smoke run
        sailed through it in silence. Configurations whose failure is decidable
        without running are now rejected before training starts, by the
        assembly-time check and by ``_check_episode_stream`` on the first
        forward. What is left for K is the class that genuinely cannot be
        decided statically: batches that are supposed to be episode-contiguous
        and are not. K must be >= 2 (at batch 1 in `stream` a healthy bank only
        reaches 2 on the second forward) and is 8 for margin -- an episode of a
        single frame, or a run of very short episodes, can delay growth past 2.

        This raise deliberately does not assert which of two causes it is
        looking at, because bank length cannot tell them apart: "the batches
        are broken" and "this configuration cannot hold memory" produce the
        same observation. The previous version asserted the first and
        recommended episode_stream_sampler=True -- which, in the configuration
        that actually reached it, was already True.
        """
        if self._bank_liveness_checked or not self.training:
            return
        self._batch_sizes_seen.add(batch_size)
        self._distinct_in_batch_seen.add(len(set(episode_ids)))
        for bank in (self.per_mem_bank, self.cog_mem_bank):
            if bank is None:
                continue
            for entries in bank.bank.values():
                if len(entries) > self._max_bank_len_seen:
                    self._max_bank_len_seen = len(entries)
        self._train_forwards += 1
        if self._train_forwards < self.BANK_LIVENESS_FORWARDS:
            return
        self._bank_liveness_checked = True
        logger.info(
            "MemoryVLAMemory bank liveness after %d training forwards: "
            "longest episode history seen = %d (batch sizes seen %s, distinct "
            "episodes per batch seen %s).",
            self._train_forwards,
            self._max_bank_len_seen,
            sorted(self._batch_sizes_seen),
            sorted(self._distinct_in_batch_seen),
        )
        if self._max_bank_len_seen > 1:
            return
        if self.mem_length <= 1:
            # Not a failure, and raising here would be a false positive: with
            # mem_length=1 consolidation trims the bank back to one entry after
            # every write, so the length can never exceed 1 -- while that one
            # entry is a real merged history and IS retrieved. Bank length is
            # simply blind in this configuration, so say so instead of ruling.
            logger.warning(
                "MemoryVLAMemory: no bank exceeded 1 entry in %d training "
                "forwards, but mem_length=%d caps bank length at 1, so this "
                "criterion cannot tell a working bank from a dead one here "
                "and is standing down rather than failing the run. Memory may "
                "well be working: a single consolidated entry is still a real "
                "history and is still retrieved. Use mem_length >= 2 to get "
                "this guard back.",
                self._train_forwards,
                self.mem_length,
            )
            return
        raise RuntimeError(
            "MemoryVLAMemory ran {} training forwards and no episode's memory "
            "ever grew past a single entry, so every retrieval found an empty "
            "history, every fusion reduced to an exact identity, 7.47M "
            "parameters received no gradient, and the loss stayed normal. "
            "Nothing else raises in this state, which is the entire reason "
            "this check exists.\n"
            "Observed: dataloader_type={!r}, group_size={}, mem_length={}, "
            "batch sizes seen={}, distinct episodes per batch seen={}, "
            "longest bank={}.\n"
            "Two different things produce this and they need different fixes. "
            "Bank length cannot tell them apart, so both are listed rather "
            "than one being asserted:\n"
            "  (a) the batches are not episode-contiguous. If "
            "memoryvla.episode_stream_sampler is off, turn it on -- every "
            "bank mode needs episode-ordered batches. If it is already on, "
            "then MemoryVLAEpisodeStreamBatchSampler's episode spans do not "
            "match this dataset: _episode_spans (sampler.py) assumes one "
            "episode's frames are contiguous in the global index.\n"
            "  (b) the configuration cannot hold memory at all -- under "
            "dataloader_type='group' that is min(group_size, batch_size) == "
            "1. assert_episode_stream_wired rejects that before training "
            "starts, so reaching here that way means it was never called "
            "on this path.".format(
                self._train_forwards,
                self.dataloader_type,
                self.group_size,
                self.mem_length,
                sorted(self._batch_sizes_seen),
                sorted(self._distinct_in_batch_seen),
                self._max_bank_len_seen,
            )
        )

    def _history_will_be_read(self, episode_ids) -> bool:
        """Would at least one sample in this batch retrieve real history?

        Two ways it can: an episode already carries entries from an earlier
        batch, or two samples here share an episode -- ``process_batch`` walks
        the batch in order and writes as it goes, so the later one reads what
        the earlier one just stored.

        Needed because at the very first frame of an episode the bank IS empty
        and the identity bypass is the correct answer. Probing before history
        exists would fail a run that is behaving exactly as designed.
        """
        if len(set(episode_ids)) < len(episode_ids):
            return True
        if self.dataloader_type == "group":
            # `group` runs bank.clear() at the top of every training call
            # (memory_bank.py:361), so whatever is in the bank right now is
            # about to be discarded. Counting it would arm the probe for a
            # forward that then legitimately has no history to read, failing a
            # run that behaves exactly as documented. Within-batch repeats
            # above still count: those are written and read inside this call.
            return False
        for bank in (self.per_mem_bank, self.cog_mem_bank):
            if bank is None:
                continue
            if any(len(bank.bank.get(eid, [])) > 0 for eid in episode_ids):
                return True
        return False

    def _assert_not_identity(self, per_in, per_out, cog_in, cog_out) -> None:
        """Raise if the module returned its input back, to within noise.

        Generic on purpose. The specific bug was `s*w + (1-s)*w == w`, but any
        cause of "the switch is on and the method is algebraically a no-op"
        lands here, which is the class of failure that produces no symptom.
        """
        self._identity_check_done = True
        gaps = {}
        if per_in is not None and per_out is not None:
            gaps["perceptual"] = float((per_out.detach() - per_in).abs().max())
        if cog_in is not None and cog_out is not None:
            gaps["cognitive"] = float((cog_out.detach() - cog_in).abs().max())
        logger.info(
            "MemoryVLAMemory identity probe on the first forward that reads "
            "history: %s (tolerance %g)",
            {k: f"{v:.6e}" for k, v in gaps.items()},
            IDENTITY_TOL,
        )
        # min, not max: the two banks share episode ids and are written in
        # lockstep, so both always have history at the same time. Requiring
        # only one of them to be non-degenerate would let a single dead stream
        # hide behind a live one.
        if gaps and min(gaps.values()) <= IDENTITY_TOL:
            raise RuntimeError(
                "MemoryVLAMemory is a numerical identity: history was "
                "available and it changed its input by {} (tolerance {:g}). "
                "The module is switched on and consuming parameters, but "
                "mathematically it is doing nothing.".format(
                    {k: f"{v:.6e}" for k, v in gaps.items()}, IDENTITY_TOL
                )
            )

    def _forward_perceptual(self, fm, episode_ids, timesteps):
        # [B, cams, C, h, w] -> [B, cams*h*w, C]
        b, cams, c, h, w = fm.shape
        tokens = fm.permute(0, 1, 3, 4, 2).reshape(b, cams * h * w, c)
        fused = self.per_mem_bank.process_batch(
            tokens, episode_ids, timesteps
        )
        # and back, so the decoder sees the layout it expects
        return (
            fused.reshape(b, cams, h, w, c)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

    def _forward_cognitive(self, embedded, mask, episode_ids, timesteps):
        b, length, c = embedded.shape
        idx = self._last_valid_index(mask, b, length, embedded.device)
        gather_idx = idx.view(b, 1, 1).expand(b, 1, c)

        cog = embedded.gather(1, gather_idx)  # [B, 1, C]
        fused = self.cog_mem_bank.process_batch(cog, episode_ids, timesteps)
        return embedded.scatter(1, gather_idx, fused.to(embedded.dtype))

    @staticmethod
    def _last_valid_index(mask, batch_size, length, device):
        """Index of the last True per row; falls back to the last position.

        ``text_token_mask`` is True for valid tokens -- non-image and non-pad
        (structure.py:344-352). The tokenizer pads on the left
        (structure.py:183) so the final position is normally valid anyway, but
        this does not lean on that.
        """
        if mask is None:
            return torch.full(
                (batch_size,), length - 1, dtype=torch.long, device=device
            )
        valid = mask.to(torch.bool)
        positions = torch.arange(length, device=valid.device)
        # -1 where invalid, so argmax over the position picks the last True
        masked = torch.where(
            valid, positions.unsqueeze(0).expand_as(valid),
            torch.full_like(positions.unsqueeze(0).expand_as(valid), -1),
        )
        idx = masked.max(dim=1).values
        # a row with no valid token at all: fall back to the last position
        idx = torch.where(idx < 0, torch.full_like(idx, length - 1), idx)
        return idx.to(device)
