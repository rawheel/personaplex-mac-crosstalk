"""Voice files: a tampered .pt can't run code or read outside its own data, and the voice
archive can't write outside the folder it's extracted to.

    .venv/bin/python tests/test_voices.py
"""

import io
import os
import pickle
import sys
import tarfile
import tempfile
import types
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosstalk.convert import convert_voices  # noqa: E402
from crosstalk.voices import load_voice_pt  # noqa: E402

# Stand-ins with the names torch.save writes, so pickle produces a real PyTorch stream
# without PyTorch installed.
torch, torch_utils = types.ModuleType("torch"), types.ModuleType("torch._utils")
sys.modules.update({"torch": torch, "torch._utils": torch_utils})


def _rebuild_tensor_v2(*_):
    raise AssertionError("the stand-in is never called")


_rebuild_tensor_v2.__module__ = "torch._utils"
torch_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
for _name in ("BFloat16Storage", "LongStorage"):
    _cls = type(_name, (), {"__module__": "torch"})
    setattr(torch, _name, _cls)


class Storage:
    def __init__(self, storage_type, key, raw: bytes, numel: int):
        self.storage_type, self.key, self.raw, self.numel = storage_type, key, raw, numel


class Tensor:
    def __init__(self, storage, offset, size, stride):
        self.args = (storage, offset, size, stride)

    def __reduce__(self):
        return (_rebuild_tensor_v2, (*self.args, False, OrderedDict()))


class TorchPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Storage):
            return ("storage", obj.storage_type, obj.key, "cpu", obj.numel)
        return None


def write_pt(path: Path, data) -> None:
    buf = io.BytesIO()
    TorchPickler(buf, protocol=2).dump(data)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("voice/data.pkl", buf.getvalue())
        for storage in _storages(data):
            z.writestr(f"voice/data/{storage.key}", storage.raw)


def _storages(data):
    if isinstance(data, Storage):
        yield data
    elif isinstance(data, Tensor):
        yield from _storages(data.args[0])
    elif isinstance(data, dict):
        for v in data.values():
            yield from _storages(v)


def bf16_storage(values, key="0") -> Storage:
    raw = (np.asarray(values, np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
    return Storage(torch.BFloat16Storage, key, raw, len(values))


def expect(exc, fn, what):
    try:
        fn()
    except exc as e:
        print(f"{what}: refused ({type(e).__name__}: {e}): OK")
        return
    raise AssertionError(f"{what} was not refused")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        emb = [0.5, -1.25, 2.0, 3.0, -0.75, 8.0]
        cache = np.array([7, 11, 13], "<i8")
        good = {
            "embeddings": Tensor(bf16_storage(emb), 0, (3, 1, 1, 2), (2, 2, 2, 1)),
            "cache": Tensor(Storage(torch.LongStorage, "1", cache.tobytes(), 3), 0, (1, 3), (3, 1)),
        }
        write_pt(tmp / "good.pt", good)
        voice = load_voice_pt(tmp / "good.pt")
        assert voice["embeddings"].shape == (3, 1, 1, 2) and voice["embeddings"].dtype == np.float32
        assert np.array_equal(voice["embeddings"].reshape(-1), emb)
        assert np.array_equal(voice["cache"], [[7, 11, 13]])
        print("a genuine voice file loads: OK")

        class RunsCommand:
            def __reduce__(self):
                return (os.system, ("true",))

        write_pt(tmp / "evil.pt", {"embeddings": RunsCommand(), "cache": 0})
        expect(pickle.UnpicklingError, lambda: load_voice_pt(tmp / "evil.pt"), "a file that calls os.system")

        # A view bigger than its data: as_strided would read past the buffer.
        write_pt(tmp / "oob.pt", {"embeddings": Tensor(bf16_storage(emb), 0, (1000,), (1,)), "cache": 0})
        expect(pickle.UnpicklingError, lambda: load_voice_pt(tmp / "oob.pt"), "a tensor view past its storage")
        write_pt(tmp / "offset.pt", {"embeddings": Tensor(bf16_storage(emb), 4, (3,), (1,)), "cache": 0})
        expect(pickle.UnpicklingError, lambda: load_voice_pt(tmp / "offset.pt"), "an offset view past its storage")
        # Stride 0 repeats one element: a tiny storage could claim a terabyte-sized tensor.
        write_pt(tmp / "zero.pt", {"embeddings": Tensor(bf16_storage(emb), 0, (10**6, 10**6), (0, 0)), "cache": 0})
        expect(pickle.UnpicklingError, lambda: load_voice_pt(tmp / "zero.pt"), "a stride-0 memory bomb")

        # The voice archive: a member path that escapes the extraction folder.
        with tarfile.open(tmp / "voices.tgz", "w:gz") as tar:
            info = tarfile.TarInfo("voices/../../escape.pt")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"PK\x03\x04"))
        expect(tarfile.FilterError, lambda: convert_voices(tmp / "voices.tgz", tmp / "out"), "a tar member with ../")
        assert not (tmp.parent / "escape.pt").exists()


if __name__ == "__main__":
    main()
