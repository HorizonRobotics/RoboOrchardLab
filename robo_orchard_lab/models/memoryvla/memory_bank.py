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
#
# ---------------------------------------------------------------------------
# This file ports the perceptual/cognitive memory bank of MemoryVLA.
#
#   MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models
#   for Robotic Manipulation, arXiv:2508.19236v2
#   https://github.com/shihao1895/MemoryVLA @ 0eef5c3, MIT License
#
# The classes below are copied from `vla/memory_vla.py` with their arithmetic
# unchanged, so that they can be checked against reference values captured
# from the original implementation. Every class carries a [port:memoryvla]
# marker naming the exact source lines. The single deliberate deviation is
# documented on BottleneckSE.
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import os

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

__all__ = [
    "TimestepEmbedder",
    "CrossTransformerBlock",
    "BottleneckSE",
    "GateFusion",
    "CogMemBank",
    "PerMemBank",
]


# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L30-L68
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t.to(next(self.mlp.parameters()).device)
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size
        ).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L71-L102
class CrossTransformerBlock(nn.Module):
    """Cross-attention + FFN block used for memory retrieval.

    Note: the original takes no attention mask, so every key/value position is
    attended to. Callers must not hand it padded history.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        query: torch.Tensor,  # (B, N, D)
        k: torch.Tensor,  # (B, M, D)
        v: torch.Tensor,  # (B, M, D)
    ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

        # residual + LN
        x = self.attn_norm(query + attn_out)

        # FFN + LN
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L105-L136
class BottleneckSE(nn.Module):
    """Squeeze-and-excitation bottleneck that compresses the channel dim.

    DEVIATION FROM THE ORIGINAL. The original recovers the spatial grid with
    ``_h = _w = int(math.sqrt(_n))`` and asserts ``_h * _h == _n``
    (memory_vla.py:126-128), i.e. it only accepts a square token grid. This
    host produces an 8x11 grid (256x352 images at patch size 32), so that
    assert would fire on every real batch.

    The fix is to take the grid explicitly. When ``hw`` is omitted the square
    inference is used verbatim, which keeps the original numerics reachable --
    that path is what the reference values in ``ref/bottleneck_se.npz`` pin.
    """

    def __init__(self, C_in, C_mid, C_out):  # noqa: N803
        super().__init__()
        self.C_in = C_in
        self.C_mid = C_mid
        self.C_out = C_out

        self.reduce = nn.Conv2d(C_in, C_mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, C_mid // 16, 1),
            nn.ReLU(),
            nn.Conv2d(C_mid // 16, C_mid, 1),
            nn.Sigmoid(),
        )

        self.expand = nn.Conv2d(C_mid, C_out, 1, bias=False)

    def forward(
        self, x: torch.Tensor, hw: Optional[tuple[int, int]] = None
    ) -> torch.Tensor:
        _b, _n, _c = x.shape
        if hw is None:
            _h = _w = int(math.sqrt(_n))
            assert _h * _h == _n, "Input feature has no spatial structure"
        else:
            _h, _w = hw
            assert _h * _w == _n, (
                f"grid {_h}x{_w} does not match {_n} tokens"
            )

        x = x.reshape(_b, _h, _w, _c).permute(0, 3, 1, 2)  # (B, C_in, H, W)
        z = self.act(self.reduce(x))
        w = self.excite(z)

        final = self.expand(z * w)
        final = final.reshape(_b, self.C_out, _n).permute(0, 2, 1)
        return final


# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L139-L155
class GateFusion(nn.Module):
    """Gated fusion of the working memory and the retrieved episode memory.

    Note: the initialisation is NOT identity. proj is seeded from
    ``normal(0, 1e-3)``, so sigmoid(~0) ~= 0.5 and the module starts out
    averaging its two inputs. Turning the memory on therefore changes the
    numbers immediately; off-equivalence has to come from not building this
    module at all, not from a zero init.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.proj(torch.cat([x1, x2], dim=-1)))
        fused = scale * x1 + (1 - scale) * x2
        return fused


# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L158-L332
class CogMemBank(nn.Module):
    """Episode-keyed memory bank: consolidate, retrieve, fuse.

    Stateful across forward calls. ``self.bank`` maps an episode id to a list
    of ``(timestep, feature)`` pairs, held detached, so history contributes no
    gradient.

    ``dataloader_type`` decides how far the memory reaches, and the two values
    differ far more than their names suggest:

    * ``stream`` -- the bank survives between calls and is only dropped when
      the episode changes. This is the episode-level memory the paper
      describes, and the mode this host uses.
    * ``group`` -- ``self.bank.clear()`` runs at the top of every training
      call, so memory never spans a batch and ``mem_length`` never binds when
      ``group_size <= mem_length``. Kept for fidelity with the original
      default, not recommended here.
    """

    def __init__(
        self,
        dataloader_type: str,
        group_size: int,
        token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = False,
    ):
        super().__init__()
        assert dataloader_type in ("stream", "group")
        assert fusion_type in ("gate", "add")
        assert consolidate_type in ("fifo", "tome")

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.retrieval_blocks = nn.ModuleList(
            [
                CrossTransformerBlock(self.token_size)
                for _ in range(self.retrieval_layers)
            ]
        )

        if self.fusion_type == "gate":
            self.gate_fusion_blocks = GateFusion(self.token_size)

        # Ablate the fusion at run time, keeping the module built and its
        # weights loadable. Setting fusion_type="add" in the package config
        # cannot express this: "add" skips GateFusion, the checkpoint's four
        # gate_fusion_blocks tensors become unexpected keys, and
        # structure.load_state_dict asserts on those under strict=True. So the
        # question "does the learned gate earn its parameters" is unanswerable
        # from the config alone.
        #
        # Read here rather than per forward, so a typo fails at startup
        # instead of silently leaving the gate in place for a whole cell.
        self._fusion_mode = (
            os.environ.get("HOLOBRAIN_FUSION_MODE", "").strip().lower()
            or self.fusion_type
        )
        if self._fusion_mode not in ("gate", "add"):
            raise ValueError(
                "HOLOBRAIN_FUSION_MODE must be gate or add, got "
                f"{self._fusion_mode!r}"
            )
        if self._fusion_mode == "gate" and self.fusion_type != "gate":
            raise ValueError(
                "HOLOBRAIN_FUSION_MODE=gate needs a package built with "
                f"fusion_type=gate, but this one is {self.fusion_type!r}: "
                "there is no gate to use."
            )

        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(
                self.token_size,
                frequency_embedding_size=self.token_size // 4,
            )
        else:
            self.timestep_encoder = None

        self.reset()

    def reset(self):
        # bank[episode_id] = [(timestep, feat[N,D]), ...]
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id):
        bank = self.bank.get(episode_id, [])
        T = len(bank)  # noqa: N806
        if T < 2:
            return

        feats = [feat for (_, feat) in bank]

        sims = []
        for i in range(T - 1):
            f1 = (
                feats[i].flatten(1)
                if feats[i].dim() > 1
                else feats[i].unsqueeze(0)
            )
            f2 = (
                feats[i + 1].flatten(1)
                if feats[i + 1].dim() > 1
                else feats[i + 1].unsqueeze(0)
            )
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())

        idx_max = int(torch.tensor(sims).argmax().item())

        timestep_i, feat_i = bank[idx_max]
        _timestep_j, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)

        bank[idx_max] = (timestep_i, fused_feat.detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(
        self,
        episode_id,
        feat: torch.Tensor,
        timestep: Optional[torch.Tensor],
    ):
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        self.bank[episode_id].append((timestep, feat.detach().clone()))

        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][
                    -self.mem_length :
                ]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError

    def process_batch(
        self,
        tokens: torch.Tensor,  # [B, N, D_role]
        episode_ids: np.ndarray,
        timesteps: np.ndarray,
    ) -> torch.Tensor:
        assert episode_ids is not None, (
            "episode_ids must be provided during training"
        )

        if self.use_timestep_pe:
            assert timesteps is not None, (
                "timesteps must be provided during training"
            )

        B, N, D = tokens.shape  # noqa: N806
        outputs = []

        if self.training:
            if self.dataloader_type == "group":
                self.bank.clear()
                self.eid_stream = None
            elif self.dataloader_type == "stream":
                first_eid = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid

        for i in range(B):
            # 1) episode management
            eid = episode_ids[i]
            if self.training:
                if self.dataloader_type == "group":
                    if i > 0 and i % self.group_size == 0:
                        prev_group_eid = episode_ids[i - self.group_size]
                        self.clear_episode(prev_group_eid)
                if self.dataloader_type == "stream":
                    if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                        self.clear_episode(episode_ids[i - 1])
                        self.eid_stream = episode_ids[i]

            # 2) memory retrieval
            working_mem = tokens[i].unsqueeze(0)  # (1, N, D)

            hist = self.bank.get(eid, [])
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]
                episode_mem = (
                    torch.stack(hist_feats, dim=0)
                    .reshape(-1, D)
                    .unsqueeze(0)
                )  # (1, T*N, D)

                if self.use_timestep_pe:
                    hist_timesteps = [t for t, _ in hist]
                    hist_timesteps = torch.tensor(hist_timesteps).to(
                        working_mem.device
                    )
                    pe = self.timestep_encoder(hist_timesteps).unsqueeze(0)
                    pe = pe.repeat_interleave(N, dim=1)  # (1, T*N, D)
                else:
                    pe = torch.zeros_like(episode_mem)

                query = working_mem
                for block in self.retrieval_blocks:
                    query = block(query, episode_mem + pe, episode_mem)

                retrieved_episode_mem = query
            else:
                # without history: working memory as episode memory
                retrieved_episode_mem = working_mem  # (1, N, D)

            # 3) memory adaptive fusion -- self._fusion_mode, not
            # self.fusion_type, so the run-time ablation can drop the gate
            # while its weights stay loaded.
            if self._fusion_mode == "add":
                fused_feats = (working_mem + retrieved_episode_mem) * 0.5
            elif self._fusion_mode == "gate":
                fused_feats = self.gate_fusion_blocks(
                    working_mem, retrieved_episode_mem
                )

            outputs.append(fused_feats)

            # 4) memory consolidate
            timestep_i = timesteps[i] if self.use_timestep_pe else None

            if self.update_fused:
                self._memory_consolidate(
                    eid, fused_feats.squeeze(0), timestep_i
                )
            else:
                self._memory_consolidate(eid, tokens[i], timestep_i)

        return torch.cat(outputs, dim=0)  # [B, N, D_role]


# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L335-L357
class PerMemBank(CogMemBank):
    """Perceptual memory bank.

    In the original this is a subclass whose ``__init__`` forwards every
    argument unchanged, i.e. it is behaviourally identical to CogMemBank and
    exists only to give the perceptual stream its own parameters. Kept as a
    distinct class for exactly that reason -- two banks must not share weights.
    """
