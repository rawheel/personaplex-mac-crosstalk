"""Voice previews: the agent greeting a silent caller in each voice, cached as WAV files.

NVIDIA ships the voices as embeddings only, with no sample audio, so a preview is rendered
by the model itself. That also makes it an honest preview: it's exactly how the agent will
sound on a call.
"""

from __future__ import annotations

import hashlib
import io
import wave
from pathlib import Path

import numpy as np
import rustymimi

from .engine import FRAME_SIZE, SAMPLE_RATE, SILENCE_TOKENS, Engine, set_seed

MAX_SECONDS = 8.0
SEED = 1234  # same greeting every time a cache is rebuilt
END_OF_GREETING = 15  # frames (1.2 s) without a new word after the agent has spoken


def preview_prompt(listings: dict) -> str:
    """A short form of the role prompt: enough for the agent to greet as itself."""
    return (
        f"You work for {listings['agency']} which is a real estate agency and your name is "
        f"{listings['agent_name']}. Information: You help callers find a home and book a viewing."
    )


def render(engine: Engine, voice: str, prompt: str) -> np.ndarray:
    """What the agent says to a caller who stays silent, cut to its opening greeting."""
    set_seed(SEED)
    gen = engine.generator()
    gen.system_prompts(engine.voice(voice), engine.encode_prompt(prompt))
    codes, spoke, quiet = [], False, 0
    for _ in range(int(MAX_SECONDS * SAMPLE_RATE / FRAME_SIZE)):
        out = gen.step(SILENCE_TOKENS)
        if out is None:
            continue
        codes.append(out[1:])
        if engine.text_piece(int(out[0])):
            spoke, quiet = True, 0
        elif spoke:
            quiet += 1
            if quiet >= END_OF_GREETING:
                break
    mimi = rustymimi.Tokenizer(engine.mimi_path, num_codebooks=8)
    pcm = mimi.decode_step(np.stack(codes, axis=-1)[None].astype(np.uint32))[0, 0]
    return trim(pcm)


def trim(pcm: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Cuts leading and trailing silence, keeping a little air around the speech."""
    win = SAMPLE_RATE // 50  # 20 ms
    n = len(pcm) // win
    if n == 0:
        return pcm
    rms = np.sqrt((pcm[: n * win].reshape(n, win) ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > threshold)
    if loud.size == 0:
        return pcm
    start = max(0, (loud[0] - 5) * win)  # 100 ms before the first sound
    end = min(len(pcm), (loud[-1] + 15) * win)  # 300 ms after the last
    out = pcm[start:end].astype(np.float32)
    fade = min(len(out) // 2, SAMPLE_RATE // 100)  # 10 ms fades so playback doesn't click
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    out[:fade] *= ramp
    out[len(out) - fade :] *= ramp[::-1]
    return out


def to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class PreviewCache:
    """One WAV per voice and prompt. The prompt is part of the file name, so changing the
    agency or agent name in listings.json renders fresh previews."""

    def __init__(self, directory: Path):
        self.directory = directory

    def path(self, voice: str, prompt: str) -> Path:
        key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return self.directory / f"{voice}-{key}.wav"

    def make(self, engine: Engine, voice: str, prompt: str) -> Path:
        path = self.path(voice, prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(to_wav(render(engine, voice, prompt)))
        tmp.replace(path)
        return path
