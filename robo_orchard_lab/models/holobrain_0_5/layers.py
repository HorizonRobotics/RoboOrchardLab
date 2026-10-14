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

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from robo_orchard_lab.models.layers.transformer_layers import FFN
from robo_orchard_lab.utils.build import build

__all__ = [
    "ActionFFN",
    "AdaRMSNorm",
    "RotaryEmbedding",
    "ScalarEmbedder",
    "RotaryAttention",
    "JointGraphAttention",
    "TemporalJointGraphAttention",
    "MultiModalAttention",
    "UpsampleHead",
    "linear_act_ln",
]


def _init_projection(linear, init="default", std=None):
    if init in (None, "default"):
        return
    if init == "zero":
        constant_(linear.weight, 0.0)
    elif init == "normal":
        if std is None:
            raise ValueError("`std` must be set when init is 'normal'.")
        nn.init.normal_(linear.weight, mean=0.0, std=std)
    else:
        raise ValueError(f"Unsupported projection init: {init}")
    if linear.bias is not None:
        constant_(linear.bias, 0.0)


def linear_act_ln(
    embed_dims,
    in_loops,
    out_loops,
    input_dims=None,
    act_cfg=None,
):
    """Build linear, activation and normalization blocks."""
    if act_cfg is None:
        act_cfg = dict(type=nn.ReLU, inplace=True)

    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(build(act_cfg))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class ScalarEmbedder(nn.Module):
    """Sinusoidal embedding of a scalar.

    Frequencies span ``[scale / max_period, scale]``, so ``scale`` sets the
    finest resolvable input difference (``1 / scale``) and ``max_period`` the
    number of resolvable levels across ``input_range``. Passing
    ``input_range`` derives ``scale`` from those two quantities; leaving it
    ``None`` keeps ``scale = 1``, which implicitly assumes the input is
    O(1)-quantized over a range of ``max_period`` (true for diffusion
    timesteps, false for physical units like m/s).

    Args:
        hidden_size (int): Output embedding dimension.
        frequency_embedding_size (int): Number of Fourier features.
        max_period (int): Ratio between the highest and lowest frequency,
            i.e. resolvable levels across ``input_range``.
        input_range (float, optional): Typical span of the input, in input
            units.
    """

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        max_period=10000,
        input_range=None,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        self.freqs: torch.Tensor
        self.register_buffer("freqs", freqs, persistent=False)
        self.input_range = input_range
        self.scale = 1.0 if input_range is None else max_period / input_range

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # `freqs` is derived from max_period/input_range. Older checkpoints
        # persisted it; dropping the key keeps them loadable under
        # strict=True without letting a stale band overwrite the new one.
        state_dict.pop(prefix + "freqs", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, t):
        t_freq = self.freqs * t[..., None] * self.scale
        t_freq = torch.cat([torch.cos(t_freq), torch.sin(t_freq)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class RotaryEmbedding(nn.Module):
    """Generate rotary position encodings for attention queries and keys."""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64)
                .float()
                .to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        freqs = (
            torch.arange(max_position_embeddings)[:, None].to(inv_freq)
            @ inv_freq[None]
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        # x: [bs,h,n,c] or [bs,n,c]
        # position_ids: [bs,n]
        position_ids = position_ids.to(torch.int32)
        cos = self.cos[position_ids]
        sin = self.sin[position_ids]
        while x.dim() > cos.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)  # b 1 n c
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos_h = cos[..., :half]
        sin_h = sin[..., :half]
        x = torch.cat(
            [x1 * cos_h - x2 * sin_h, x2 * cos_h + x1 * sin_h], dim=-1
        )
        return x


class RotaryAttention(nn.Module):
    """Apply rotary attention with optional inference key/value caching."""

    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        max_position_embeddings=128,
        output_proj_init="default",
        output_proj_std=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, all_head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)
        self.proj = nn.Linear(all_head_dim, embed_dims)
        self.position_encoder = RotaryEmbedding(
            head_dim, max_position_embeddings=max_position_embeddings
        )
        self._kv_cache: Optional[tuple] = None
        self._use_cache: bool = False
        self.output_proj_init = output_proj_init
        self.output_proj_std = output_proj_std
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            constant_(self.v_proj.bias, 0.0)
        _init_projection(
            self.proj, self.output_proj_init, self.output_proj_std
        )

    def clear_cache(self):
        self._kv_cache = None

    def do_cache(self, enable: bool):
        self._use_cache = enable

    def project_kv(
        self,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project keys and values, applying RoPE to K.

        Reads and writes no layer state; the result is what a caller
        passes to :meth:`attend` as ``k`` / ``v``.

        Args:
            key: Key input, ``(B, M, C)``.
            value: Value input, ``(B, M, C)``. Defaults to ``key``.
            key_pos: RoPE positions for K, ``(B, M)``. None applies no
                rotation, which is the image branch's convention.

        Returns:
            ``(k, v)``, both ``(B, num_heads, M, head_dim)``. ``k``
            carries RoPE, so :meth:`attend` must not re-apply it.
        """
        B, M, _ = key.shape  # noqa: N806
        if value is None:
            value = key
        k = (
            self.k_proj(key)
            .reshape(B, M, self.num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(value)
            .reshape(B, M, self.num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        if key_pos is not None:
            k = self.position_encoder(k, key_pos)
        return (k, v)

    def attend(
        self,
        query,
        k,
        v,
        query_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        identity=None,
    ):
        """Attend projected queries against given K and V.

        The attention core every read path shares: q-projection,
        optional RoPE on Q, mask build, SDPA and the output projection.
        ``k`` / ``v`` are post-projection tensors as returned by
        :meth:`project_kv` or held by a caller's cache, so their key
        length may exceed the current input's.

        Args:
            query: ``(B, N, C)``.
            k: ``(B, num_heads, M, head_dim)``, RoPE already applied.
            v: ``(B, num_heads, M, head_dim)``.
            query_pos: RoPE positions for Q, ``(B, N)``. None applies
                no rotation.
            attn_mask: Boolean mask, True blocks the key. 2-D
                ``(N, M)``, 3-D ``(B, N, M)`` or ``(B*num_heads, N,
                M)``; the head axis is added here.
            key_padding_mask: ``(B, M)``, True marks a padded key.
            identity: Residual base; defaults to ``query``.

        Returns:
            ``(B, N, C)``.
        """
        if identity is None:
            identity = query

        B, N, C = query.shape  # noqa: N806

        q = (
            self.q_proj(query)
            .reshape(B, N, self.num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        if query_pos is not None:
            q = self.position_encoder(q, query_pos)

        M = k.shape[2]  # noqa: N806

        # Build additive float mask for SDPA (True → -inf)
        float_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 3 and attn_mask.shape[0] == B:
                attn_mask = attn_mask.unsqueeze(1)
            float_mask = q.new_zeros(B, 1, N, M).masked_fill(
                attn_mask, float("-inf")
            )
        if key_padding_mask is not None:
            kpm = q.new_zeros(B, 1, 1, M).masked_fill(
                key_padding_mask[:, None, None], float("-inf")
            )
            float_mask = float_mask + kpm if float_mask is not None else kpm

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=float_mask, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x + identity
        return x

    def project_kv_cached(self, key, value=None, key_pos=None):
        """Project K/V, consulting and filling this layer's cache.

        The cache holds one static key set: when ``do_cache`` is on and
        a projection is already stored, it is returned unchanged;
        otherwise the input is projected and, with caching on, stored.
        ``do_cache`` off always projects fresh.

        Args:
            key: Key input, ``(B, M, C)``.
            value: Value input; defaults to ``key``.
            key_pos: RoPE positions for K.

        Returns:
            The cached or freshly projected ``(k, v)``.
        """
        if self._use_cache and self._kv_cache is not None:
            # Reuse cached K/V (img/text features are constant across
            # all denoising timesteps).
            return self._kv_cache
        k, v = self.project_kv(key, value, key_pos)
        if self._use_cache:
            self._kv_cache = (k, v)
        return (k, v)

    def forward(
        self,
        query,
        key,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        pre_scale=True,
        key_padding_mask=None,
        identity=None,
        **kwargs,
    ):
        if identity is None:
            identity = query

        k, v = self.project_kv_cached(key, value, key_pos)
        return self.attend(
            query,
            k,
            v,
            query_pos=query_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            identity=identity,
        )


class JointGraphAttention(nn.Module):
    """Attend over joints using relative kinematic graph distances."""

    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        output_proj_init="default",
        output_proj_std=None,
        joint_relative_range=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, all_head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)

        self.proj = nn.Linear(all_head_dim, embed_dims)
        if joint_relative_range is None:
            self.position_encoder = ScalarEmbedder(embed_dims)
        else:
            self.position_encoder = ScalarEmbedder(
                embed_dims,
                max_period=joint_relative_range,
                input_range=joint_relative_range,
            )
        self.output_proj_init = output_proj_init
        self.output_proj_std = output_proj_std
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            constant_(self.v_proj.bias, 0.0)
        _init_projection(
            self.proj, self.output_proj_init, self.output_proj_std
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        identity=None,
        **kwargs,
    ):
        if identity is None:
            identity = query
        B, N, C = query.shape  # noqa: N806
        M = key.shape[1]  # noqa: N806
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(key)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # b,h,n,c
        k = k.reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)  # b,h,m,c

        query_pos = self.position_encoder(query_pos.flatten()).reshape(
            -1, N, M, C
        )  # bs, n, m, c
        query_pos = query_pos.unflatten(-1, (self.num_heads, -1)).permute(
            0, 3, 1, 2, 4
        )
        if B != query_pos.shape[0]:
            query_pos = query_pos.tile(B // query_pos.shape[0], 1, 1, 1, 1)

        q = q[:, :, :, None] * query_pos

        attn = (q * k.unsqueeze(2)).sum(-1) * self.scale
        # Equivalent Implementation
        # attn = torch.einsum("bhnmc,bhmc->bhnm", q, k) * self.scale

        attn = attn.softmax(dim=-1).type_as(v)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = x + identity
        return x


class TemporalJointGraphAttention(nn.Module):
    """Attention carrying a joint-distance gate and a temporal RoPE.

    Query and key are ``(B, N, T, C)`` -- ``N``/``M`` joints by ``T``
    time steps. ``joint_distance`` is ``(B, N, M)``, one value per
    (query joint, key joint) pair, and multiplies into the scores in
    channel space. ``temporal_attn_mask`` is ``True`` where a query may
    not read a key; see :meth:`_broadcast_attn_mask` for the ranks it
    accepts.

    The layer-owned cache (``do_cache``) and a caller's own rollout
    memory are separate: the methods here (:meth:`project_kv`,
    :meth:`encode_joint_pe`, :meth:`attend`) read no cache state, so a
    caller holding its own K/V composes them freely without touching
    the cache.
    """

    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        max_position_embeddings=None,
        output_proj_init="default",
        output_proj_std=None,
        joint_relative_range=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, all_head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dims, all_head_dim, bias=qkv_bias)
        self.proj = nn.Linear(all_head_dim, embed_dims)

        if joint_relative_range is None:
            self.joint_pos_encoder = ScalarEmbedder(embed_dims)
        else:
            self.joint_pos_encoder = ScalarEmbedder(
                embed_dims,
                max_period=joint_relative_range,
                input_range=joint_relative_range,
            )
        self.temporal_position_encoder = RotaryEmbedding(
            head_dim, max_position_embeddings=max_position_embeddings
        )
        self._joint_dist_cache: Optional[torch.Tensor] = None
        self._use_cache: bool = False
        self.output_proj_init = output_proj_init
        self.output_proj_std = output_proj_std
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            constant_(self.v_proj.bias, 0.0)
        _init_projection(
            self.proj, self.output_proj_init, self.output_proj_std
        )

    def clear_cache(self):
        self._joint_dist_cache = None

    def do_cache(self, enable: bool):
        self._use_cache = enable

    @staticmethod
    def _broadcast_attn_mask(
        mask: torch.Tensor,
        B: int,  # noqa: N803
        N: int,  # noqa: N803
        T_q: int,  # noqa: N803
        M: int,  # noqa: N803
        T_k: int,  # noqa: N803
    ) -> torch.Tensor:
        """Reshape an attention mask to broadcast against the scores.

        Scores are ``(B, H, N, T_q, M, T_k)`` and every accepted rank
        broadcasts over the head axis. ``True`` means *block*. Four
        ranks, in increasing specificity:

        * ``(T_q, T_k)`` -- temporal only, shared by all joints and
          samples.
        * ``(B, T_q, T_k)`` -- temporal only, per sample.
        * ``(N, T_q, M, T_k)`` -- full joint x time, shared by the batch.
        * ``(B, N, T_q, M, T_k)`` -- full joint x time, per sample.

        The two joint-aware ranks let a caller block specific
        ``(query_joint, key_joint)`` pairs. A ``(T_q, T_k)`` mask cannot
        express that -- it forces every query joint to see the same key
        set, which is wrong for a layout whose legal keys differ per row.

        The leading batch axis may be ``1`` to broadcast across samples.
        """
        rank = mask.dim()
        if rank == 2:
            expected = (T_q, T_k)
            out = mask[None, None, None, :, None, :]
        elif rank == 3:
            expected = (mask.shape[0], T_q, T_k)
            out = mask[:, None, None, :, None, :]
        elif rank == 4:
            expected = (N, T_q, M, T_k)
            out = mask[None, None, :, :, :, :]
        elif rank == 5:
            expected = (mask.shape[0], N, T_q, M, T_k)
            out = mask[:, None, :, :, :, :]
        else:
            raise ValueError(
                "TemporalJointGraphAttention: temporal_attn_mask must be "
                "rank 2 (T_q, T_k), 3 (B, T_q, T_k), 4 (N, T_q, M, T_k) "
                f"or 5 (B, N, T_q, M, T_k); got rank {rank} with shape "
                f"{tuple(mask.shape)}."
            )
        assert tuple(mask.shape) == expected, (
            "TemporalJointGraphAttention: temporal_attn_mask shape "
            f"{tuple(mask.shape)} does not match the attention scores "
            f"(B={B}, N={N}, T_q={T_q}, M={M}, T_k={T_k}); expected "
            f"{expected} for a rank-{rank} mask."
        )
        if rank in (3, 5):
            assert mask.shape[0] in (1, B), (
                "TemporalJointGraphAttention: temporal_attn_mask batch "
                f"size {mask.shape[0]} is neither 1 nor B={B}."
            )
        return out

    def encode_joint_pe(
        self,
        joint_distance: torch.Tensor,
        B: int,  # noqa: N803
        N: int,  # noqa: N803
        M: int,  # noqa: N803
        C: int,  # noqa: N803
        columns_per_joint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode ``(B, N, M)`` joint distances to a broadcastable gate.

        Returns ``(B, H, N, 1, M, head_dim)`` -- the singleton is the
        query-time axis, so the gate multiplies every ``t_q`` of a given
        query joint identically.

        ``columns_per_joint`` widens the encoded gate along ``M`` by
        repeating each of the ``M`` inputs the given number of times,
        giving ``M_out = columns_per_joint.sum()``. It exists for key axes
        that flatten a *ragged* joint x time product into one axis: such
        an axis has one column per (joint, step) pair but only as many
        distinct joint values as there are joints. Repeating after the
        encoder keeps the :class:`ScalarEmbedder` MLP on the small
        distinct set rather than the widened one.
        """
        joint_pe = self.joint_pos_encoder(joint_distance.flatten()).reshape(
            B, N, M, C
        )
        if columns_per_joint is not None:
            assert columns_per_joint.shape == (M,), (
                "columns_per_joint must carry one column count per key joint; "
                f"got shape {tuple(columns_per_joint.shape)} for M={M}."
            )
            joint_pe = joint_pe.repeat_interleave(columns_per_joint, dim=2)
        joint_pe = joint_pe.unflatten(-1, (self.num_heads, -1)).permute(
            0, 3, 1, 2, 4
        )  # (b, h, n, m, c)
        return joint_pe.unsqueeze(3)  # (b, h, n, 1, m, c)

    @staticmethod
    def _fold_joint_pe(
        q: torch.Tensor,
        k: torch.Tensor,
        joint_pe: torch.Tensor,
        fold: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multiply the joint gate into the query or into the key.

        The gate is bilinear in channel space --
        ``score = sum_c q[n,q,c] * g[n,m,c] * k[m,k,c]`` -- so either
        operand gives the same scores. Which is cheaper depends on the
        geometry: a query-side fold materialises
        ``(B, H, N, T_q, M, hd)`` and a key-side fold
        ``(B, H, N, M, T_k, hd)``, so the winner is whichever of ``T_q``
        and ``T_k`` is smaller.

        Query-side suits a key axis that is a full joint x time
        rectangle. Key-side suits keys that are wide in joints but
        short in time -- the shape a flattened, ragged joint x time axis
        takes, where every column is its own joint and ``T_k`` collapses
        to 1.

        Args:
            q: ``(B, H, N, T_q, hd)`` projected queries.
            k: ``(B, H, M, T_k, hd)`` projected keys.
            joint_pe: ``(B, H, N, 1, M, hd)`` gate.
            fold: ``"query"`` or ``"key"``.

        Returns:
            ``(q_out, k_out)`` for the scoring einsum. Exactly one
            carries the gate, and it gains an axis.
        """
        if fold == "query":
            # (b,h,n,t_q,1,c) * (b,h,n,1,m,c) -> (b,h,n,t_q,m,c)
            return q.unsqueeze(4) * joint_pe, k
        if fold == "key":
            # (b,h,n,m,1,c) * (b,h,1,m,t_k,c) -> (b,h,n,m,t_k,c)
            gate = joint_pe.permute(0, 1, 2, 4, 3, 5)
            return q, gate * k.unsqueeze(2)
        raise ValueError(
            "TemporalJointGraphAttention: fold_joint_pe must be 'query' or "
            f"'key', got {fold!r}."
        )

    def project_kv(
        self,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        temporal_pos_k: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project keys and values, applying temporal RoPE to K.

        Reads and writes no layer state. ``v`` keeps the joint axis
        folded in, M-major, matching the layout a key axis flattened
        joint-major produces.

        Args:
            key: Key input ``(B, M, T_k, C)``.
            value: Value input; defaults to ``key``.
            temporal_pos_k: RoPE positions for K.

        Returns:
            ``(k, v)``: ``k`` is ``(B, num_heads, M, T_k, head_dim)``
            with RoPE applied; ``v`` is ``(B, num_heads, M * T_k,
            head_dim)``. A caller that needs the joint gate calls
            :meth:`encode_joint_pe` alongside.
        """
        B, M, T_k, _ = key.shape  # noqa: N806
        if value is None:
            value = key
        k = (
            self.k_proj(key)
            .reshape(B, M, T_k, self.num_heads, -1)
            .permute(0, 3, 1, 2, 4)
        )
        v = (
            self.v_proj(value)
            .reshape(B, M * T_k, self.num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        if temporal_pos_k is not None:
            k = self.temporal_position_encoder(k, temporal_pos_k)
        return (k, v)

    def attend(
        self,
        query,
        k,
        v,
        joint_pe=None,
        temporal_pos_q=None,
        temporal_attn_mask=None,
        identity=None,
        fold_joint_pe="query",
    ):
        """Attend queries against given K, V and joint gate.

        The attention core every read path shares: q-projection,
        optional RoPE on Q, gate fold, mask build, softmax, the value
        mix and the output projection. ``k`` / ``v`` / ``joint_pe`` are
        post-projection tensors as returned by :meth:`project_kv` and
        :meth:`encode_joint_pe` or held by a caller's cache.

        Args:
            query: ``(B, N, T_q, C)``.
            k: ``(B, num_heads, M, T_k, head_dim)``, RoPE applied.
            v: ``(B, num_heads, M * T_k, head_dim)``.
            joint_pe: ``(B, num_heads, N, 1, M, head_dim)`` gate. None
                attends ungated.
            temporal_pos_q: RoPE positions for Q.
            temporal_attn_mask: True blocks the key; see
                :meth:`_broadcast_attn_mask` for accepted ranks.
            identity: Residual base; defaults to ``query``.
            fold_joint_pe: Which operand the gate multiplies into;
                see :meth:`_fold_joint_pe`.

        Returns:
            ``(B, N, T_q, C)``.
        """
        if identity is None:
            identity = query

        B, N, T_q, C = query.shape  # noqa: N806

        q = (
            self.q_proj(query)
            .reshape(B, N, T_q, self.num_heads, -1)
            .permute(0, 3, 1, 2, 4)
        )  # (b, h, n, t_q, c)
        if temporal_pos_q is not None:
            q = self.temporal_position_encoder(q, temporal_pos_q)

        M, T_k = k.shape[2], k.shape[3]  # noqa: N806

        q_gated, k_gated = self._fold_joint_pe(q, k, joint_pe, fold_joint_pe)

        # (b, h, n, t_q, m, t_k)
        if fold_joint_pe == "query":
            attn = (
                torch.einsum("bhnqmc,bhmkc->bhnqmk", q_gated, k_gated)
                * self.scale
            )
        else:
            attn = (
                torch.einsum("bhnqc,bhnmkc->bhnqmk", q_gated, k_gated)
                * self.scale
            )

        if temporal_attn_mask is not None:
            attn = torch.where(
                self._broadcast_attn_mask(
                    temporal_attn_mask, B, N, T_q, M, T_k
                ),
                float("-inf"),
                attn,
            )
        attn = attn.reshape(B, self.num_heads, T_q * N, T_k * M)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)  # (b, n*t_q, h, c)
        x = x.reshape(B, N, T_q, C)

        x = self.proj(x)
        x = x + identity
        return x

    def encode_joint_pe_cached(self, joint_distance, query, channels):
        """Encode the joint gate, consulting and filling this layer's cache.

        The cache holds one gate: when ``do_cache`` is on and a gate is
        already stored, it is returned unchanged; otherwise the input
        distances are encoded and, with caching on, stored. ``do_cache``
        off always encodes fresh. K/V are never cached here -- in
        diffusion denoising they depend on the changing noisy action.

        Args:
            joint_distance: ``(B, N, M)`` distances.
            query: The forward's query tensor; only its leading shape
                is read.
            channels: Embedding width ``C``.

        Returns:
            The cached or freshly encoded gate.
        """
        if self._use_cache and self._joint_dist_cache is not None:
            return self._joint_dist_cache
        B, N = query.shape[:2]  # noqa: N806
        M = joint_distance.numel() // (B * N)  # noqa: N806
        joint_pe = self.encode_joint_pe(
            joint_distance,
            B,
            N,
            M,
            channels,
        )
        if self._use_cache:
            self._joint_dist_cache = joint_pe
        return joint_pe

    def forward(
        self,
        query,
        key=None,
        value=None,
        joint_distance=None,
        temporal_pos_q=None,
        temporal_pos_k=None,
        temporal_attn_mask=None,
        identity=None,
        **kwargs,
    ):
        if identity is None:
            identity = query
        C = query.shape[3]  # noqa: N806

        joint_pe = self.encode_joint_pe_cached(joint_distance, query, C)
        k, v = self.project_kv(key, value, temporal_pos_k)

        return self.attend(
            query,
            k,
            v,
            joint_pe=joint_pe,
            temporal_pos_q=temporal_pos_q,
            temporal_attn_mask=temporal_attn_mask,
            identity=identity,
        )


class MultiModalAttention(nn.Module):
    """Fuse visual, instruction and temporal joint action features."""

    def __init__(
        self,
        img_cross_attn,
        text_cross_attn,
        temp_joint_attn,
        embed_dims,
        num_heads=8,
        state_drop_rate=0.2,
        scale=3,
        parallel_attn=False,
    ):
        super().__init__()
        self.img_cross_attn = build(img_cross_attn)
        self.text_cross_attn = build(text_cross_attn)
        self.temp_joint_attn = build(temp_joint_attn)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.router = nn.Linear(embed_dims, 3 * num_heads)
        self.state_drop_rate = state_drop_rate
        self.scale = scale
        self.parallel_attn = parallel_attn
        self._branch_streams = {}

    def clear_cache(self):
        self.img_cross_attn.clear_cache()
        self.text_cross_attn.clear_cache()
        self.temp_joint_attn.clear_cache()

    def do_cache(self, enable: bool):
        self.img_cross_attn.do_cache(enable)
        self.text_cross_attn.do_cache(enable)
        self.temp_joint_attn.do_cache(enable)

    def _get_branch_streams(self, device):
        device_idx = torch.device(device).index
        streams = self._branch_streams.get(device_idx)
        if streams is None:
            streams = tuple(torch.cuda.Stream(device=device) for _ in range(3))
            self._branch_streams[device_idx] = streams
        return streams

    def project_kv(
        self,
        state_feature: torch.Tensor,
        temporal_pos_k: Optional[torch.Tensor],
        text_feature: Optional[torch.Tensor],
        tca_key_pos: Optional[torch.Tensor],
        img_feature: torch.Tensor,
        ica_key_pos: Optional[torch.Tensor] = None,
    ) -> dict:
        """Project K/V for all three sub-attentions.

        Reads and writes no layer state. The returned dict holds one
        entry per branch, each the branch layer's own
        ``project_kv`` output:

        - ``"img"``: ``(k, v)`` for the image branch.
        - ``"text"``: the same for text, or ``None`` when the batch
          carries no text.
        - ``"temp_joint"``: ``(k, v)`` for the state branch, its gate
          available separately through
          :meth:`TemporalJointGraphAttention.encode_joint_pe`.

        Args:
            state_feature: Key input ``(B, M, T_k, C)`` for the
                temporal-joint branch.
            temporal_pos_k: RoPE positions for the state K.
            text_feature: Text key input ``(B, L_text, C)``, or None.
            tca_key_pos: RoPE positions for text K, or None.
            img_feature: Image key input ``(B, L_img, C)``.
            ica_key_pos: RoPE positions for image K, or None.

        Returns:
            The per-branch K/V dict.
        """
        return {
            "img": self.img_cross_attn.project_kv(
                key=img_feature, key_pos=ica_key_pos
            ),
            "text": (
                None
                if text_feature is None
                else self.text_cross_attn.project_kv(
                    key=text_feature, key_pos=tca_key_pos
                )
            ),
            "temp_joint": self.temp_joint_attn.project_kv(
                key=state_feature,
                temporal_pos_k=temporal_pos_k,
            ),
        }

    def attend_branches(
        self,
        query,
        joint_pe,
        temp_joint_kv,
        img_kv,
        text_kv=None,
        temporal_pos_q=None,
        temporal_attn_mask=None,
        identity=None,
        text_key_padding_mask=None,
        tca_query_pos=None,
        img_query_pos=None,
        fold_joint_pe="query",
    ):
        """Route-mix attention over given per-branch K/V and the gate.

        The fused attention core every read path shares: q-routing,
        each sub-branch attending against the K/V handed in, and the
        router-weighted mix. K/V / gate are post-projection tensors as
        returned by :meth:`project_kv` and
        :meth:`TemporalJointGraphAttention.encode_joint_pe`.

        Args:
            query: ``(B, N, T_q, C)``.
            joint_pe: Gate for the state branch,
                ``(B, H, N, 1, M, head_dim)``.
            temp_joint_kv: The state branch's ``(k, v)``.
            img_kv: The image branch's ``(k, v)``.
            text_kv: The text branch's ``(k, v)``, or None when the
                batch carries no text -- the branch then contributes
                ``identity`` and the router blends state and image.
            temporal_pos_q: RoPE positions for the state Q.
            temporal_attn_mask: True blocks the state-branch key; see
                :meth:`_broadcast_attn_mask` for accepted ranks.
            identity: Residual base; defaults to ``query``.
            text_key_padding_mask: ``(B, L_text)``, True marks a padded
                text key.
            tca_query_pos: RoPE positions for the text Q.
            img_query_pos: RoPE positions for the image Q.
            fold_joint_pe: Which operand the state gate multiplies
                into; see :meth:`_fold_joint_pe`.

        Returns:
            ``(B, N, T_q, C)``.
        """
        bs, num_joint_q, num_step_q = query.shape[:3]
        if identity is None:
            identity = query

        query_flat = query.flatten(1, 2)
        identity_flat = identity.flatten(1, 2)

        def route():
            probs = self.router(query).unflatten(-1, (self.num_heads, 3))
            if self.training:
                # Drop the state branch (index 0) independently in every head.
                mask = torch.rand_like(probs) < self.state_drop_rate
                mask[..., 1:] = False
                probs = torch.where(mask, float("-inf"), probs)
            return (probs / self.scale).softmax(dim=-1).unsqueeze(-2)

        def state_attn():
            k, v = temp_joint_kv
            return self.temp_joint_attn.attend(
                query,
                k,
                v,
                joint_pe=joint_pe,
                temporal_pos_q=temporal_pos_q,
                temporal_attn_mask=temporal_attn_mask,
                identity=identity,
                fold_joint_pe=fold_joint_pe,
            )

        def text_attn():
            if text_kv is None:
                return identity
            k, v = text_kv
            return self.text_cross_attn.attend(
                query_flat,
                k,
                v,
                query_pos=tca_query_pos,
                key_padding_mask=text_key_padding_mask,
                identity=identity_flat,
            ).unflatten(1, (num_joint_q, num_step_q))

        def img_attn():
            k, v = img_kv
            return self.img_cross_attn.attend(
                query_flat,
                k,
                v,
                query_pos=img_query_pos,
                identity=identity_flat,
            ).unflatten(1, (num_joint_q, num_step_q))

        if self.parallel_attn and query.is_cuda:
            cur_stream = torch.cuda.current_stream(query.device)
            state_stream, text_stream, img_stream = self._get_branch_streams(
                query.device
            )
            for stream in (state_stream, text_stream, img_stream):
                stream.wait_stream(cur_stream)

            with torch.cuda.stream(state_stream):
                out_state_attn = state_attn()
            with torch.cuda.stream(text_stream):
                out_text_attn = text_attn()
            with torch.cuda.stream(img_stream):
                out_img_attn = img_attn()

            probs = route()
            for stream in (state_stream, text_stream, img_stream):
                cur_stream.wait_stream(stream)
        else:
            probs = route()
            out_state_attn = state_attn()
            out_text_attn = text_attn()
            out_img_attn = img_attn()

        return (
            (
                torch.stack(
                    [out_state_attn, out_text_attn, out_img_attn], dim=-1
                ).unflatten(-2, (self.num_heads, -1))
                * probs
            )
            .sum(dim=-1)
            .flatten(-2)
        )

    def forward(
        self,
        query,
        state_feature,
        joint_distance,
        temporal_pos_q,
        temporal_pos_k,
        temporal_attn_mask,
        text_feature,
        text_key_padding_mask,
        tca_query_pos,
        tca_key_pos,
        img_feature,
        ica_query_pos,
        ica_key_pos,
        identity=None,
    ):
        if identity is None:
            identity = query

        C = query.shape[3]  # noqa: N806
        joint_pe = self.temp_joint_attn.encode_joint_pe_cached(
            joint_distance, query, C
        )
        temp_joint_kv = self.temp_joint_attn.project_kv(
            state_feature, None, temporal_pos_k
        )
        img_kv = self.img_cross_attn.project_kv_cached(
            img_feature, img_feature, ica_key_pos
        )
        text_kv = (
            None
            if text_feature is None
            else self.text_cross_attn.project_kv_cached(
                text_feature, text_feature, tca_key_pos
            )
        )

        return self.attend_branches(
            query,
            joint_pe,
            temp_joint_kv,
            img_kv,
            text_kv=text_kv,
            temporal_pos_q=temporal_pos_q,
            temporal_attn_mask=temporal_attn_mask,
            identity=identity,
            text_key_padding_mask=text_key_padding_mask,
            tca_query_pos=tca_query_pos,
            img_query_pos=ica_query_pos,
        )


class AdaRMSNorm(nn.RMSNorm):
    """Modulate normalized action features using the diffusion timestep."""

    def __init__(
        self,
        normalized_shape,
        condition_dims,
        num_condition_mlp_layers=2,
        elementwise_affine=False,
        eps=1e-6,
        zero=False,
        ada_init="default",
        gate_bias=1.0,
        **kwargs,
    ):
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            **kwargs,
        )
        self.zero = zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                condition_dims, normalized_shape * (2 if not zero else 6)
            ),
        )
        self.ada_init = ada_init
        self.gate_bias = gate_bias
        self.init_weights()

    def init_weights(self):
        if self.ada_init in (None, "default"):
            return
        if self.ada_init != "identity":
            raise ValueError(f"Unsupported AdaRMSNorm init: {self.ada_init}")

        linear = self.adaLN_modulation[-1]
        with torch.no_grad():
            constant_(linear.weight, 0.0)
            constant_(linear.bias, 0.0)
            if self.zero:
                bias = linear.bias.view(-1, 6)
                bias[:, 2] = self.gate_bias
                bias[:, 5] = self.gate_bias

    def forward(self, x, c):
        x = super().forward(x)
        return self.apply_ada_func(x, c)

    def apply_ada_func(self, x, c):
        dims = x.shape[-1]
        ada_scale_shift = self.adaLN_modulation(c).unflatten(-1, (dims, -1))
        if ada_scale_shift.dim() != 4:
            ada_scale_shift = ada_scale_shift[:, None]
        x = x * (1 + ada_scale_shift[..., 0]) + ada_scale_shift[..., 1]
        if self.zero:
            gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_scale_shift[
                ..., 2:
            ].unbind(dim=-1)
        else:
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class UpsampleHead(nn.Module):
    """Upsample chunk features into per-step robot-state predictions."""

    def __init__(
        self,
        upsample_sizes,
        input_dim,
        dims,
        norm,
        act,
        norm_act_idx,
        num_output_layers=-1,
        out_dim=8,
    ):
        super().__init__()
        self.norm_act_idx = norm_act_idx
        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.act_and_norm = nn.ModuleList()
        dims = [input_dim] + dims
        for i, size in enumerate(upsample_sizes):
            if i in norm_act_idx:
                norm["normalized_shape"] = dims[i]
                self.act_and_norm.append(
                    nn.Sequential(build(act), build(norm))
                )
            else:
                self.act_and_norm.append(None)

            self.upsamples.append(
                nn.Upsample(
                    size=(size, 1), mode="bilinear", align_corners=True
                ),
            )
            self.convs.append(nn.Conv1d(dims[i], dims[i + 1], 3, padding=1))

        self.num_output_layers = num_output_layers
        if num_output_layers >= 1:
            self.output_layers = nn.Sequential()
            for _ in range(num_output_layers):
                self.output_layers.append(build(act))
                self.output_layers.append(
                    nn.Linear(dims[-1], dims[-1]),
                )
            self.output_layers.append(
                nn.Linear(dims[-1], out_dim),
            )

    def forward(self, x):
        bs, num_joint, num_chunk, state_dims = x.shape
        x = x.flatten(0, 1)
        # Keep channels-first (B*J, C, T) throughout the loop,
        # only permute to channels-last when norm requires it.
        x = x.permute(0, 2, 1)
        for i, layer in enumerate(self.upsamples):
            if i in self.norm_act_idx:
                x = x.permute(0, 2, 1)
                x = self.act_and_norm[i](x)
                x = x.permute(0, 2, 1)
            x = self.convs[i](layer(x.unsqueeze(-1)).squeeze(-1))
        x = x.permute(0, 2, 1)
        x = x.unflatten(0, (bs, num_joint))

        if self.num_output_layers >= 1:
            x = self.output_layers(x)
        return x


class ActionFFN(FFN):
    """Feed-forward block with a zero-initialized residual projection.

    Parameter names match the shared FFN and the original action checkpoint.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nn.init.zeros_(self.layers[-2].weight)
        nn.init.zeros_(self.layers[-2].bias)
