"""Streaming generation for PersonaPlex on MLX.

The token bookkeeping is a port of NVIDIA's LMGen (moshi/models/lm.py in
github.com/NVIDIA/personaplex): a small ring buffer of tokens per stream (text, 8 agent
audio codebooks, 8 user audio codebooks), each stream with its own delay, plus a mask of
which positions were provided (teacher forced) rather than sampled.

Two things differ from NVIDIA's implementation, both without changing the output:
  * Only the 8 agent codebooks are sampled by the depformer. NVIDIA also samples the 8
    user codebooks and then discards them, since the user's audio is always provided.
  * System prompts (voice, silence, text, silence) are fully teacher forced, so nothing
    sampled during them is ever used. Instead of one model step per frame, the whole
    prompt goes through the transformer in two batched passes ("prefill"). The bookkeeping
    still runs frame by frame so the state afterwards is identical. `prompt_mode="step"`
    keeps the frame-by-frame path for verification.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import sentencepiece

from .convert import DEFAULT_OUT, MIMI, TEXT_TOKENIZER, build_model, model_file
from .model.lm import Lm, personaplex_7b_config
from .model.sampling import Sampler

UNGENERATED = -2
# Text token fed during prompts and silences (NVIDIA's zero_text_code, the padding token).
TEXT_PAD = 3
# Mimi codes of a silent frame (agent stream) and of a 440 Hz tone (user stream), as used
# by NVIDIA when building system prompts.
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80 ms at 24 kHz, one model step
SILENCE_FRAMES = 6  # 0.5 s, as in NVIDIA's server


@dataclass
class Voice:
    name: str
    embeddings: mx.array  # [frames, 1, 1, dim]
    cache: np.ndarray  # [1 + n_q, CT]


@dataclass
class _Prepared:
    input: np.ndarray  # [K] tokens fed to the model at this step
    target: np.ndarray  # [K] tokens at the position being predicted
    provided: np.ndarray  # [K] whether each target token is forced
    input_pos: int
    target_pos: int


class Generator:
    def __init__(
        self,
        model: Lm,
        text_sampler: Sampler | None = None,
        audio_sampler: Sampler | None = None,
        repeat_penalty: float = 0.0,
        repeat_window: int = 40,
    ):
        self.model = model
        cfg = model.cfg
        self.cfg = cfg
        self.K = cfg.num_codebooks
        self.dep_q = cfg.dep_q
        self.delays = np.array(cfg.delays)
        self.max_delay = cfg.max_delay
        self.CT = self.max_delay + 3
        self.initial = np.array([cfg.text_card] + [cfg.audio_card] * cfg.n_q, dtype=np.int64)
        self.text_sampler = text_sampler or Sampler(temp=0.7, top_k=25)
        self.audio_sampler = audio_sampler or Sampler(temp=0.8, top_k=250)
        # Each use of a word token among the agent's last `repeat_window` lowers its logit
        # by `repeat_penalty`, which breaks loops like "it was so good, it was so good".
        # Special tokens (padding, i.e. staying quiet) are never penalized.
        self.repeat_penalty = repeat_penalty
        self.recent_words: collections.deque[int] = collections.deque(maxlen=repeat_window)
        self.reset()

    def reset(self):
        self.model.reset()
        self.cache = np.full((self.K, self.CT), UNGENERATED, dtype=np.int64)
        self.provided = np.zeros((self.K, self.CT), dtype=bool)
        self.offset = 0
        self.recent_words.clear()

    # --- bookkeeping (NVIDIA LMGen.prepare_step_input / process_transformer_output) ---

    def _write(self, first_k: int, tokens: np.ndarray):
        for i, tok in enumerate(tokens):
            k = first_k + i
            pos = (self.offset + self.delays[k]) % self.CT
            self.cache[k, pos] = tok
            self.provided[k, pos] = True

    def _prepare(self, user=None, agent=None, text=None) -> _Prepared | None:
        if user is not None:
            self._write(1 + self.dep_q, user)
        if agent is not None:
            self._write(1, agent)
        if text is not None:
            self._write(0, [text])
        for k, delay in enumerate(self.delays):
            if self.offset <= delay:
                self.cache[k, self.offset % self.CT] = self.initial[k]
                self.provided[k, self.offset % self.CT] = True
        if self.offset == 0:
            self.cache[:, 0] = self.initial
            self.offset += 1
            return None
        ip = (self.offset - 1) % self.CT
        tp = self.offset % self.CT
        return _Prepared(
            input=self.cache[:, ip].copy(),
            target=self.cache[:, tp].copy(),
            provided=self.provided[:, tp].copy(),
            input_pos=ip,
            target_pos=tp,
        )

    def _finish(self, p: _Prepared, text: int | None, audio: np.ndarray | None) -> np.ndarray | None:
        """Stores sampled tokens where nothing was forced. Returns the [1 + dep_q] tokens
        completed this step (text + agent audio, realigned across delays), if any."""
        tp = p.target_pos
        self.provided[:, p.input_pos] = False
        if text is not None and not self.provided[0, tp]:
            self.cache[0, tp] = text
        if audio is not None:
            gen = slice(1, self.dep_q + 1)
            self.cache[gen, tp] = np.where(self.provided[gen, tp], self.cache[gen, tp], audio)
        if self.offset <= self.max_delay:
            self.offset += 1
            return None
        idx = (self.offset - self.max_delay + self.delays[: self.dep_q + 1]) % self.CT
        out = self.cache[np.arange(self.dep_q + 1), idx]
        self.offset += 1
        return out

    # --- model steps ---

    def _sample_step(self, p: _Prepared) -> np.ndarray | None:
        codes = mx.array(p.input[None, :, None].astype(np.int32))
        transformer_out, text_logits = self.model.forward_codes(codes)
        text_logits = text_logits[:, -1:]
        if self.repeat_penalty and self.recent_words:
            ids, counts = np.unique(np.fromiter(self.recent_words, dtype=np.int32), return_counts=True)
            penalty = mx.array(-self.repeat_penalty * counts.astype(np.float32), dtype=text_logits.dtype)
            text_logits = text_logits.at[..., mx.array(ids)].add(penalty)
        text, _ = self.text_sampler(text_logits)
        target = mx.array(p.target[None].astype(np.int32))
        provided = mx.array(p.provided[None])
        next_text = mx.where(provided[:, :1], target[:, :1], text)
        gen = slice(1, self.dep_q + 1)
        audio = self.model.depformer.sample(
            transformer_out,
            self.audio_sampler,
            next_text,
            target[:, gen],
            provided[:, gen],
            self.model.depformer_cache,
        )
        mx.eval(text, audio)
        token = int(text.item())
        if token > TEXT_PAD and not p.provided[0]:
            self.recent_words.append(token)
        return self._finish(p, token, np.array(audio[0], dtype=np.int64))

    def step(self, user_tokens: np.ndarray) -> np.ndarray | None:
        """One 80 ms frame: user Mimi codes [8] in, [text, 8 agent codes] out (or None
        during the first frame of delay)."""
        p = self._prepare(user=user_tokens)
        if p is None:
            return None
        return self._sample_step(p)

    # --- system prompts (NVIDIA LMGen.step_system_prompts) ---

    def _forced(self, p: _Prepared) -> None:
        if not p.provided[: self.dep_q + 1].all():
            raise RuntimeError("prompt step is not fully teacher forced")
        self._finish(p, None, None)

    def system_prompts(self, voice: Voice | None, text_tokens: list[int], mode: str = "prefill"):
        """Runs the voice prompt, 0.5 s of silence, the text prompt and 0.5 s of silence."""
        self.reset()
        if voice is not None:
            dummy = self.initial[1:]
            n = voice.embeddings.shape[0]
            for i in range(n):
                p = None
                while p is None:
                    p = self._prepare(user=dummy[: self.dep_q], agent=dummy[self.dep_q :], text=TEXT_PAD)
                if mode == "step":
                    self.model.forward_embeddings(voice.embeddings[i])
                    mx.eval([c.keys for c in self.model.transformer_cache])
                self._forced(p)
            if mode == "prefill":
                self.model.forward_embeddings(voice.embeddings.reshape(1, n, -1))
            self.cache[:] = voice.cache[0]

        inputs = []
        texts = [TEXT_PAD] * SILENCE_FRAMES + list(text_tokens) + [TEXT_PAD] * SILENCE_FRAMES
        for tok in texts:
            p = self._prepare(user=SINE_TOKENS, agent=SILENCE_TOKENS, text=tok)
            if p is None:
                continue
            if mode == "step":
                self.model.forward_codes(mx.array(p.input[None, :, None].astype(np.int32)))
                mx.eval([c.keys for c in self.model.transformer_cache])
            else:
                inputs.append(p.input)
            self._forced(p)
        if inputs:
            codes = mx.array(np.stack(inputs, axis=1)[None].astype(np.int32))
            self.model.forward_codes(codes)
        mx.eval([(c.keys, c.values) for c in self.model.transformer_cache])


def wrap_system_prompt(text: str) -> str:
    text = text.strip()
    if text.startswith("<system>") and text.endswith("<system>"):
        return text
    return f"<system> {text} <system>"


class Engine:
    """Loads the converted model, tokenizer and voices once and hands out generators."""

    def __init__(self, model_dir: Path = DEFAULT_OUT, bits: int | None = 4):
        self.model_dir = Path(model_dir)
        weights = model_file(self.model_dir, bits)
        if not weights.exists():
            raise FileNotFoundError(f"{weights} not found, run `goyaah-convert` first")
        cfg = personaplex_7b_config()
        self.model = build_model(cfg, bits)
        self.model.load_weights(str(weights), strict=True)
        mx.eval(self.model.parameters())
        self.tokenizer = sentencepiece.SentencePieceProcessor(str(self.model_dir / TEXT_TOKENIZER))
        self.mimi_path = str(self.model_dir / MIMI)
        self.voices_dir = self.model_dir / "voices"
        self._voices: dict[str, Voice] = {}

    def voice_names(self) -> list[str]:
        return sorted(p.stem for p in self.voices_dir.glob("*.safetensors"))

    def voice(self, name: str) -> Voice:
        if name not in self._voices:
            if name not in self.voice_names():
                raise ValueError(f"unknown voice {name!r}, choose from {self.voice_names()}")
            data = mx.load(str(self.voices_dir / f"{name}.safetensors"))
            self._voices[name] = Voice(name, data["embeddings"], np.array(data["cache"], dtype=np.int64))
        return self._voices[name]

    def encode_prompt(self, text: str) -> list[int]:
        if not text.strip():
            return []
        return self.tokenizer.encode(wrap_system_prompt(text))

    def text_piece(self, token: int) -> str | None:
        if token in (0, TEXT_PAD):
            return None
        return self.tokenizer.id_to_piece(token).replace("▁", " ")

    def generator(self, **sampler_kwargs) -> Generator:
        return Generator(self.model, **sampler_kwargs)


def set_seed(seed: int | None):
    if seed is not None and seed >= 0:
        mx.random.seed(seed)
        np.random.seed(seed)


__all__ = ["Engine", "Generator", "Voice", "FRAME_SIZE", "SAMPLE_RATE", "set_seed"]
