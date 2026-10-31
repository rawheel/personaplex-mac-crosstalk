# KVCache comes from Kyutai's moshi_mlx (MIT), itself adapted from:
# https://github.com/ml-explore/mlx-examples/blob/6c2369e4b97f49fb5906ec46033497b39931b25d/llms/mlx_lm/models/base.py#L1
# Copyright © 2023-2024 Apple Inc.
#
# RingKVCache is new in crosstalk.

import mlx.core as mx


class KVCache:
    """Growable cache, used by the depformer which is reset at every frame."""

    def __init__(self, head_dim: int, n_kv_heads: int):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 16

    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            shape = (B, self.n_kv_heads, n_steps * self.step, self.head_dim)
            new_k = mx.zeros(shape, keys.dtype)
            new_v = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        assert self.values is not None
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self):
        self.offset = 0
        self.keys = None
        self.values = None


class RingKVCache:
    """Fixed-size cache holding the last `capacity` positions.

    Keys are stored after RoPE has been applied, so their order in the buffer does not
    matter for single-token attention. Multi-token (prefill) updates are only allowed
    while the buffer has not wrapped yet, which is always the case for system prompts,
    and get a causal mask from the attention layer.

    The capacity matches the model's attention context (3000 frames, i.e. 4 minutes),
    so memory stays bounded however long the conversation lasts.
    """

    def __init__(self, head_dim: int, n_kv_heads: int, capacity: int):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.keys = None
        self.values = None
        self.offset = 0

    def _allocate(self, B: int, dtype):
        shape = (B, self.n_kv_heads, self.capacity, self.head_dim)
        self.keys = mx.zeros(shape, dtype)
        self.values = mx.zeros(shape, dtype)

    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        B, _, T, _ = keys.shape
        if self.keys is None:
            self._allocate(B, keys.dtype)
        assert self.values is not None
        if T == 1:
            idx = self.offset % self.capacity
            self.keys[..., idx : idx + 1, :] = keys
            self.values[..., idx : idx + 1, :] = values
            self.offset += 1
            n = min(self.offset, self.capacity)
            return self.keys[..., :n, :], self.values[..., :n, :]
        if self.offset + T > self.capacity:
            raise ValueError(
                f"prefill of {T} positions at offset {self.offset} exceeds the "
                f"cache capacity {self.capacity}"
            )
        prev = self.offset
        self.keys[..., prev : prev + T, :] = keys
        self.values[..., prev : prev + T, :] = values
        self.offset += T
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self):
        # Keep the buffers allocated: stale entries are never read since reads are
        # bounded by the offset.
        self.offset = 0
