"""Reads PersonaPlex voice prompt files (.pt) without PyTorch and without running pickle code.

A voice prompt is a PyTorch zip archive holding a dict with two tensors:
  - "embeddings": [frames, 1, 1, 4096] bfloat16, inputs to replay into the main transformer
  - "cache": [1, 17, 4] int64, the token cache to restore once the replay is done

torch.load would unpickle arbitrary objects. Here the pickle stream is parsed with an
allow-list that only knows how to rebuild plain tensors and dicts, so a tampered file
fails to load instead of executing code.
"""

from __future__ import annotations

import math
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

_STORAGE_DTYPES = {
    "BFloat16Storage": "bfloat16",
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32)


def _rebuild_tensor(storage, storage_offset, size, stride, *_unused):
    # The file chooses the view; it must lie inside the storage it came from. as_strided
    # doesn't check that, and a view past the end would read whatever memory follows.
    if not isinstance(storage, np.ndarray) or storage.ndim != 1:
        raise pickle.UnpicklingError("tensor data is not a storage")
    size, stride = tuple(size), tuple(stride)
    dims = (storage_offset, *size, *stride)
    if not all(type(n) is int and n >= 0 for n in dims) or len(size) != len(stride):
        raise pickle.UnpicklingError("bad tensor shape")
    if len(size) == 0:
        if storage_offset >= len(storage):
            raise pickle.UnpicklingError("tensor lies outside its storage")
        return storage[storage_offset : storage_offset + 1].reshape(())
    count = math.prod(size)
    last = storage_offset + sum((n - 1) * s for n, s in zip(size, stride))  # index of the last element
    if count and (last >= len(storage) or count > len(storage) - storage_offset):
        raise pickle.UnpicklingError("tensor lies outside its storage")
    itemsize = storage.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[storage_offset:],
        shape=size,
        strides=tuple(s * itemsize for s in stride),
    ).copy()


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage" or not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        raw = self._archive.read(f"{self._prefix}/data/{key}")
        dtype = _STORAGE_DTYPES[storage_type.name]
        if dtype == "bfloat16":
            return _bf16_to_f32(raw)
        return np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<")).copy()


def load_voice_pt(path: str | Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        pkl = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise ValueError(f"{path}: not a PyTorch archive")
        prefix = pkl[0][: -len("/data.pkl")]
        with archive.open(pkl[0]) as f:
            data = _SafeUnpickler(f, archive, prefix).load()
    if not isinstance(data, dict) or not {"embeddings", "cache"} <= data.keys():
        raise ValueError(f"{path}: expected a dict with 'embeddings' and 'cache'")
    return {"embeddings": data["embeddings"], "cache": data["cache"]}
