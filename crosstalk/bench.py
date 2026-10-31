"""Checks and measurements for the MLX port.

    crosstalk-bench speed            # ms per 80 ms frame (must stay under 80 for real time)
    crosstalk-bench verify-prefill   # batched system prompt == frame-by-frame system prompt
    crosstalk-bench offline in.wav out.wav [--prompt ...] [--voice NATF2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import rustymimi
import sphn

from .agent import build_prompt, load_listings
from .convert import DEFAULT_OUT
from .engine import FRAME_SIZE, SAMPLE_RATE, SINE_TOKENS, Engine, set_seed

ASSISTANT_PROMPT = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."


def speed(engine: Engine, args):
    gen = engine.generator()
    prompt = build_prompt(load_listings(), set())
    tokens = engine.encode_prompt(prompt)
    for mode in ("prefill", "step"):
        t0 = time.perf_counter()
        gen.system_prompts(engine.voice(args.voice), tokens, mode=mode)
        print(f"system prompt ({len(tokens)} text tokens), {mode}: {time.perf_counter() - t0:.2f}s")

    times = []
    for i in range(args.frames):
        t0 = time.perf_counter()
        gen.step(SINE_TOKENS)
        times.append((time.perf_counter() - t0) * 1000)
    steady = np.array(times[10:])
    print(
        f"{len(steady)} frames: mean {steady.mean():.1f} ms, p50 {np.percentile(steady, 50):.1f}, "
        f"p90 {np.percentile(steady, 90):.1f}, max {steady.max():.1f} (budget 80 ms)"
    )
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB")


def verify_prefill(engine: Engine, args):
    tokens = engine.encode_prompt(ASSISTANT_PROMPT)
    probe = None
    results = {}
    for mode in ("step", "prefill"):
        gen = engine.generator()
        gen.system_prompts(engine.voice(args.voice), tokens, mode=mode)
        state = (gen.cache.copy(), gen.provided.copy(), gen.offset)
        p = gen._prepare(user=SINE_TOKENS)
        if probe is None:
            probe = p.input
        assert (p.input == probe).all(), "token state differs before the probe step"
        _, logits = engine.model.forward_codes(mx.array(p.input[None, :, None].astype(np.int32)))
        results[mode] = (state, np.array(logits.astype(mx.float32))[0, -1])
    (s1, l1), (s2, l2) = results["step"], results["prefill"]
    same_state = all(np.array_equal(a, b) for a, b in zip(s1[:2], s2[:2])) and s1[2] == s2[2]
    diff = np.abs(l1 - l2).max()
    agree = l1.argmax() == l2.argmax()
    print(f"token state identical: {same_state}")
    print(f"next text logits: max abs diff {diff:.4f}, same argmax: {agree}")
    top1, top2 = np.argsort(l1)[-5:][::-1], np.argsort(l2)[-5:][::-1]
    print(f"top-5 step:    {[engine.tokenizer.id_to_piece(int(t)) for t in top1]}")
    print(f"top-5 prefill: {[engine.tokenizer.id_to_piece(int(t)) for t in top2]}")


def offline(engine: Engine, args):
    set_seed(args.seed)
    pcm, _ = sphn.read(args.input, sample_rate=SAMPLE_RATE)
    pcm = pcm[0]
    gen = engine.generator()
    prompt = args.prompt if args.prompt is not None else ASSISTANT_PROMPT
    gen.system_prompts(engine.voice(args.voice), engine.encode_prompt(prompt))
    mimi = rustymimi.Tokenizer(engine.mimi_path, num_codebooks=8)
    frames = len(pcm) // FRAME_SIZE
    codes = mimi.encode_step(pcm[None, None, : frames * FRAME_SIZE])  # [1, 8, frames]
    out_codes, text, words = [], [], []  # words: (seconds into the input, piece)
    t0 = time.perf_counter()
    for i in range(codes.shape[-1]):
        out = gen.step(codes[0, :, i].astype(np.int64))
        if out is None:
            continue
        out_codes.append(out[1:])
        piece = engine.text_piece(int(out[0]))
        if piece:
            text.append(piece)
            words.append((round(i * FRAME_SIZE / SAMPLE_RATE, 2), piece))
    elapsed = time.perf_counter() - t0
    audio = mimi.decode_step(np.stack(out_codes, axis=-1)[None].astype(np.uint32))
    sphn.write_wav(args.output, audio[0, 0], SAMPLE_RATE)
    transcript = "".join(text).strip()
    Path(args.output).with_suffix(".json").write_text(json.dumps({"text": transcript, "words": words}))
    print(f"{frames} frames in {elapsed:.1f}s ({elapsed / frames * 1000:.1f} ms/frame)")
    print(f"agent said: {transcript}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["speed", "verify-prefill", "offline"])
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bits", type=int, choices=[4, 8, 16], default=4)
    parser.add_argument("--voice", default="NATF2")
    parser.add_argument("--prompt")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42424242)
    args = parser.parse_args()

    t0 = time.time()
    engine = Engine(args.model_dir, None if args.bits == 16 else args.bits)
    print(f"model loaded in {time.time() - t0:.1f}s")
    {"speed": speed, "verify-prefill": verify_prefill, "offline": offline}[args.command](engine, args)


if __name__ == "__main__":
    main()
