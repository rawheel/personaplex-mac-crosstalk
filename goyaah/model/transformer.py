# Adapted from Kyutai's moshi_mlx (MIT): https://github.com/kyutai-labs/moshi
# Copyright (c) Kyutai, all rights reserved.
#
# Changes for Goyaah: cross-attention and conv paths removed, the main transformer
# uses a bounded ring cache, and multi-token inputs get a causal mask so that system
# prompts can be prefilled in a single forward pass.

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .kv_cache import KVCache, RingKVCache


@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    context: int
    max_period: int
    positional_embedding: str

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


def causal_mask(t: int, prev: int, dtype) -> mx.array:
    """Additive mask for `t` new queries attending to `prev` cached + `t` new keys."""
    rows = mx.arange(prev, prev + t)
    cols = mx.arange(prev + t)
    return (rows[:, None] < cols[None]).astype(dtype) * -1e9


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.scale = cfg.head_dim ** (-0.5)
        self.rope = None
        if cfg.positional_embedding == "rope":
            self.rope = nn.RoPE(cfg.head_dim, traditional=True, base=cfg.max_period)

    def __call__(self, xs: mx.array, cache: KVCache | RingKVCache) -> mx.array:
        b, t, hd = xs.shape
        qkv = self.in_proj(xs).reshape(b, t, 3, self.cfg.num_heads, self.cfg.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        prev = cache.offset
        if self.rope is not None:
            q = self.rope(q, offset=prev)
            k = self.rope(k, offset=prev)

        k, v = cache.update_and_fetch(k, v)
        mask = causal_mask(t, prev, xs.dtype) if t > 1 else None
        xs = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        xs = xs.transpose(0, 2, 1, 3).reshape(b, t, hd)
        return self.out_proj(xs)


class MlpGating(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        hidden = 2 * cfg.dim_feedforward // 3
        if cfg.dim_feedforward == 4 * cfg.d_model:
            hidden = 11 * cfg.d_model // 4

        self.linear_in = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, cfg.d_model, bias=False)

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.linear_in(xs)
        b, t, _ = xs.shape
        xs = xs.reshape(b, t, 2, -1)
        return self.linear_out(nn.silu(xs[:, :, 0]) * xs[:, :, 1])


class TransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.gating = MlpGating(cfg)
        self.norm1 = nn.RMSNorm(cfg.d_model, 1e-8)
        self.norm2 = nn.RMSNorm(cfg.d_model, 1e-8)
        self.self_attn = Attention(cfg)

    def __call__(self, xs: mx.array, cache: KVCache | RingKVCache) -> mx.array:
        xs = xs + self.self_attn(self.norm1(xs), cache=cache)
        return xs + self.gating(self.norm2(xs))


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.layers = [TransformerLayer(cfg=cfg) for _ in range(cfg.num_layers)]

    def __call__(self, xs: mx.array, cache: list) -> mx.array:
        for layer, c in zip(self.layers, cache):
            xs = layer(xs, cache=c)
        return xs

    def make_cache(self) -> list[KVCache]:
        return [KVCache(self.cfg.head_dim, self.cfg.num_heads) for _ in self.layers]

    def make_ring_cache(self) -> list[RingKVCache]:
        return [
            RingKVCache(self.cfg.head_dim, self.cfg.num_heads, self.cfg.context)
            for _ in self.layers
        ]
