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
        # guardrail state; plain attributes, so nothing enters state_dict
        self._episode_check_done = False
        self._identity_check_done = False

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

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        """Drop all memory. Call at an episode boundary during inference.

        The banks only manage episodes while ``self.training`` is set
        (memory_vla.py:267), so nothing clears them at inference time unless
        the caller does it.
        """
        for bank in (self.per_mem_bank, self.cog_mem_bank):
            if bank is not None:
                bank.reset()
        self._last_episode_ids = None

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
            raise KeyError(
                f"MemoryVLAMemory needs `{self.episode_id_key}` in the batch "
                "to key its memory by episode, but the batch does not carry "
                "it. Add it to the ItemSelection whitelist in the dataset "
                "config."
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
            self._autoreset_for_eval(episode_ids)

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

        return feature_maps, text_dict

    # -- guardrails --------------------------------------------------------
    def _check_episode_stream(self, episode_ids, batch_size) -> None:
        """Log the first batch's episode spread; raise if all-distinct.

        Once per run, not per batch: a warning on every batch is a warning
        nobody reads. Training only -- at inference, batches legitimately span
        many episodes.
        """
        if self._episode_check_done or not self.training:
            return
        if self.dataloader_type != "stream":
            return
        self._episode_check_done = True
        distinct = len(set(episode_ids))
        logger.info(
            "MemoryVLAMemory[stream]: first training batch holds %d distinct "
            "episode(s) across %d samples.",
            distinct,
            batch_size,
        )
        if batch_size > 1 and distinct == batch_size:
            raise RuntimeError(
                "memoryvla.dataloader_type='stream' but the first training "
                "batch has {} samples from {} different episodes. Each sample "
                "is then the only frame of its episode the bank ever sees, so "
                "there is no history to retrieve and every fusion reduces to "
                "an exact identity. Use an episode-ordered batch sampler "
                "(memoryvla.episode_stream_sampler=True).".format(
                    batch_size, distinct
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
