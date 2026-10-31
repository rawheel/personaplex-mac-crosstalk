"""One-time conversion of NVIDIA's PersonaPlex checkpoint into a quantized MLX model.

The PyTorch checkpoint is 16.7 GB. Converting it once to 4-bit MLX weights (~4.5 GB) means
every later start loads a file a quarter of the size, and the original can be deleted.
Voice prompts are converted from PyTorch archives to safetensors at the same time, so the
runtime never needs PyTorch or pickle.

Usage:
    crosstalk-convert                 # 4-bit, writes to ~/.cache/crosstalk
    crosstalk-convert --bits 8
    crosstalk-convert --delete-original
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download, scan_cache_dir

from .model.lm import Lm, LmConfig, personaplex_7b_config
from .voices import load_voice_pt

HF_REPO = "nvidia/personaplex-7b-v1"
CHECKPOINT = "model.safetensors"
MIMI = "tokenizer-e351c8d8-checkpoint125.safetensors"
TEXT_TOKENIZER = "tokenizer_spm_32k_3.model"
DEFAULT_OUT = Path.home() / ".cache" / "crosstalk" / "personaplex-7b-v1"
CHECKPOINT_DEPFORMER_SLICES = 16


def map_weights(pth: dict[str, mx.array], cfg: LmConfig) -> dict[str, mx.array]:
    """Maps NVIDIA (PyTorch) parameter names to crosstalk (MLX) ones.

    Follows Kyutai's scripts/import_mlx.py: norm `alpha` tensors are squeezed, the
    depformer's per-step weights are split into one module per slice, and only the
    first `cfg.dep_q` of the checkpoint's 16 slices are kept.
    """
    out: dict[str, mx.array] = {
        "out_norm.weight": pth["out_norm.alpha"][0, 0],
        "text_emb.weight": pth["text_emb.weight"],
        "text_linear.weight": pth["text_linear.weight"],
    }
    for cb in range(cfg.n_q):
        out[f"audio_embs.{cb}.weight"] = pth[f"emb.{cb}.weight"]
    for k, v in pth.items():
        if k.startswith("transformer."):
            if k.endswith(".alpha"):
                v = v[0, 0]
            k = k.replace(".alpha", ".weight").replace(".in_proj_weight", ".in_proj.weight")
            out[k] = v

    for idx in range(cfg.dep_q):
        base = f"depformer.slices.{idx}"
        out[f"{base}.linear_in.weight"] = pth[f"depformer_in.{idx}.weight"]
        out[f"{base}.linear_out.weight"] = pth[f"linears.{idx}.weight"]
        emb = "depformer_text_emb" if idx == 0 else f"depformer_emb.{idx - 1}"
        out[f"{base}.emb.weight"] = pth[f"{emb}.weight"]
        for layer in range(cfg.depformer.num_layers):
            src = f"depformer.layers.{layer}"
            dst = f"{base}.transformer.layers.{layer}"
            out[f"{dst}.norm1.weight"] = pth[f"{src}.norm1.alpha"][0, 0]
            out[f"{dst}.norm2.weight"] = pth[f"{src}.norm2.alpha"][0, 0]
            out[f"{dst}.gating.linear_in.weight"] = pth[f"{src}.gating.{idx}.linear_in.weight"]
            out[f"{dst}.gating.linear_out.weight"] = pth[f"{src}.gating.{idx}.linear_out.weight"]
            for proj, name in (("in_proj_weight", "in_proj"), ("out_proj.weight", "out_proj")):
                chunks = mx.split(pth[f"{src}.self_attn.{proj}"], CHECKPOINT_DEPFORMER_SLICES)
                out[f"{dst}.self_attn.{name}.weight"] = chunks[idx]
    return out


def quantize_linears(model: nn.Module, bits: int) -> None:
    # Embeddings stay in bf16: they are lookups (no matmul to speed up) and keeping them
    # exact preserves the special "-1 means zero" behaviour of ScaledEmbedding.
    group_size = 32 if bits == 4 else 64
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )


def build_model(cfg: LmConfig, bits: int | None) -> Lm:
    model = Lm(cfg)
    model.set_dtype(mx.bfloat16)
    if bits is not None:
        quantize_linears(model, bits)
    return model


def convert_weights(checkpoint: Path, out_file: Path, bits: int | None) -> None:
    cfg = personaplex_7b_config()
    t0 = time.time()
    pth = mx.load(str(checkpoint))  # lazy: tensors are read when evaluated
    model = Lm(cfg)
    model.set_dtype(mx.bfloat16)
    model.load_weights(list(map_weights(pth, cfg).items()), strict=True)
    # Drop our references to the loaded tensors so each one can be freed as soon as
    # its quantized version has been computed.
    del pth
    if bits is not None:
        quantize_linears(model, bits)
    # Evaluate layer by layer so peak memory stays near the size of one bf16 layer
    # plus the quantized result, rather than the whole 16.7 GB checkpoint.
    for layer in model.transformer.layers:
        mx.eval(layer.parameters())
    mx.eval(model.parameters())
    params = dict(nn.utils.tree_flatten(model.parameters()))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp.safetensors")
    mx.save_safetensors(str(tmp), params, metadata={"bits": str(bits or 16)})
    tmp.replace(out_file)
    size = out_file.stat().st_size / 1e9
    print(f"wrote {out_file} ({size:.2f} GB) in {time.time() - t0:.0f}s")


def convert_voices(voices_tgz: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        for pt in sorted(Path(tmp).rglob("*.pt")):
            voice = load_voice_pt(pt)
            mx.save_safetensors(
                str(out_dir / f"{pt.stem}.safetensors"),
                {
                    "embeddings": mx.array(voice["embeddings"]).astype(mx.bfloat16),
                    "cache": mx.array(voice["cache"].astype("int32")),
                },
            )
    print(f"converted {len(list(out_dir.glob('*.safetensors')))} voices into {out_dir}")


def model_file(out_dir: Path, bits: int | None) -> Path:
    return out_dir / (f"model.q{bits}.safetensors" if bits else "model.bf16.safetensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, choices=[4, 8, 16], default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, help="local model.safetensors (default: download)")
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="remove the 16.7 GB PyTorch checkpoint from the Hugging Face cache afterwards",
    )
    args = parser.parse_args()
    bits = None if args.bits == 16 else args.bits

    checkpoint = args.checkpoint or Path(hf_hub_download(HF_REPO, CHECKPOINT))
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    target = model_file(out_dir, bits)
    if target.exists():
        print(f"{target} already exists, skipping weight conversion")
    else:
        convert_weights(checkpoint, target, bits)

    for name in (MIMI, TEXT_TOKENIZER):
        dst = out_dir / name
        if not dst.exists():
            shutil.copy(hf_hub_download(HF_REPO, name), dst)
    if not (out_dir / "voices").exists():
        convert_voices(Path(hf_hub_download(HF_REPO, "voices.tgz")), out_dir / "voices")
    (out_dir / "crosstalk.json").write_text(json.dumps({"source": HF_REPO, "bits": args.bits}))

    if args.delete_original:
        _delete_cached_checkpoint()


def _delete_cached_checkpoint() -> None:
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id != HF_REPO:
            continue
        for rev in repo.revisions:
            for f in rev.files:
                if f.file_name == CHECKPOINT:
                    size = f.size_on_disk / 1e9
                    f.blob_path.unlink(missing_ok=True)
                    f.file_path.unlink(missing_ok=True)
                    print(f"deleted cached {CHECKPOINT} ({size:.1f} GB)")
                    return


if __name__ == "__main__":
    main()
