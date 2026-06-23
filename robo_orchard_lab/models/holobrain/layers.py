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

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from robo_orchard_lab.utils.build import build


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
    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        max_period=10000,
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
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def forward(self, t):
        t_freq = self.freqs * t[:, None]
        t_freq = torch.cat([torch.cos(t_freq), torch.sin(t_freq)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class RotaryEmbedding(nn.Module):
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

        B, N, C = query.shape  # noqa: N806
        M = key.shape[1]  # noqa: N806

        if self._use_cache and self._kv_cache is not None:
            # Reuse cached K/V (img/text features are constant across
            # all denoising timesteps).
            k, v = self._kv_cache
            q = self.q_proj(query)
            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            if query_pos is not None:
                q = self.position_encoder(q, query_pos)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            if value is None:
                value = key
            v = self.v_proj(value)
            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
            q, k = self.apply_position_encode(q, query_pos, k, key_pos)
            if self._use_cache:
                self._kv_cache = (k, v)

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

    def apply_position_encode(self, q, query_pos, k, key_pos):
        if query_pos is not None:
            q = self.position_encoder(q, query_pos)
        if key_pos is not None:
            k = self.position_encoder(k, key_pos)
        return q, k


class JointGraphAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
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
        self.position_encoder = ScalarEmbedder(embed_dims)
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
    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        max_position_embeddings=None,
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

        self.joint_pos_encoder = ScalarEmbedder(embed_dims)
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
        B, N, T_q, C = query.shape  # noqa: N806
        M, T_k = key.shape[1:3]  # noqa: N806
        q = self.q_proj(query)
        k = self.k_proj(key)
        if value is None:
            value = key
        v = self.v_proj(value)

        q = q.reshape(B, N, T_q, self.num_heads, -1).permute(
            0, 3, 1, 2, 4
        )  # b,h,n,tq,c
        k = k.reshape(B, M, T_k, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(B, M * T_k, self.num_heads, -1).permute(
            0, 2, 1, 3
        )  # b,h,m*tk,c

        q = self.temporal_position_encoder(q, temporal_pos_q)
        k = self.temporal_position_encoder(k, temporal_pos_k)

        if self._use_cache and self._joint_dist_cache is not None:
            joint_distance = self._joint_dist_cache
        else:
            joint_distance = self.joint_pos_encoder(
                joint_distance.flatten()
            ).reshape(B, N, M, C)  # bs, n, m, c
            joint_distance = joint_distance.unflatten(
                -1, (self.num_heads, -1)
            ).permute(0, 3, 1, 2, 4)  # bs, h, n, m, c
            joint_distance = joint_distance.unsqueeze(3)  # bs, h, n,1, m, c
            if self._use_cache:
                self._joint_dist_cache = joint_distance
        q = q.unsqueeze(4)  # bs, h, n, tq, 1, c
        q = q * joint_distance  # bs,h,n,tq,m,c

        attn = torch.einsum("bhnqmc,bhmkc->bhnqmk", q, k) * self.scale

        if temporal_attn_mask is not None:
            attn = torch.where(
                temporal_attn_mask[..., None, None, :, None, :],
                float("-inf"),
                attn,
            )
        attn = attn.reshape(B, self.num_heads, T_q * N, T_k * M)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, T_q, C)

        x = self.proj(x)
        x = x + identity
        return x


class MultiModalAttention(nn.Module):
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
        bs, num_joint_q, num_step_q = query.shape[:3]
        if identity is None:
            identity = query

        query_flat = query.flatten(1, 2)
        identity_flat = identity.flatten(1, 2)

        def route():
            probs = self.router(query)
            if self.training:
                mask = torch.rand_like(probs) < self.state_drop_rate
                mask[..., 1:] = False
                probs = torch.where(mask, float("-inf"), probs)
            return (
                (probs.unflatten(-1, (self.num_heads, 3)) / self.scale)
                .softmax(dim=-1)
                .unsqueeze(-2)
            )

        def state_attn():
            return self.temp_joint_attn(
                query=query,
                key=state_feature,
                joint_distance=joint_distance,
                temporal_pos_q=temporal_pos_q,
                temporal_pos_k=temporal_pos_k,
                temporal_attn_mask=temporal_attn_mask,
                identity=identity,
            )

        def text_attn():
            return self.text_cross_attn(
                query=query_flat,
                key=text_feature,
                key_padding_mask=text_key_padding_mask,
                query_pos=tca_query_pos,
                key_pos=tca_key_pos,
                identity=identity_flat,
            ).unflatten(1, (num_joint_q, num_step_q))

        def img_attn():
            return self.img_cross_attn(
                query=query_flat,
                key=img_feature,
                query_pos=ica_query_pos,
                key_pos=ica_key_pos,
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

        output = (
            (
                torch.stack(
                    [out_state_attn, out_text_attn, out_img_attn], dim=-1
                ).unflatten(-2, (self.num_heads, -1))
                * probs
            )
            .sum(dim=-1)
            .flatten(-2)
        )
        return output


class AdaRMSNorm(nn.RMSNorm):
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
