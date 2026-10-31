"""Transcribes the caller's side of a call, so the UI can show the call as a conversation.

PersonaPlex only writes out its own words. The caller's audio is cut into utterances by a
level detector, and each utterance is transcribed by Whisper (faster-whisper, int8) on the
CPU, which leaves the GPU to the voice model.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
import warnings
from typing import Callable

import numpy as np
import sphn

# Apple's Accelerate BLAS raises floating point flags in this matmul under NumPy 2; the
# features come out right, the warnings are noise.
warnings.filterwarnings("ignore", r".* encountered in matmul", RuntimeWarning, "faster_whisper.feature_extractor")

SAMPLE_RATE = 24000  # the caller's audio as the server receives it
WINDOW = 480  # 20 ms level windows
MIN_LEVEL = 0.008  # RMS that counts as sound even in a silent room
START = (6, 10)  # 6 loud windows out of the last 10 (120 of 200 ms) start an utterance
END_WINDOWS = 35  # 700 ms of quiet ends it
PRE_ROLL = 15  # keep 300 ms from before the start, so the first syllable isn't clipped
MAX_SECONDS = 20.0


def load_model(name: str = "small.en"):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4)


class CallerTranscriber:
    """Feed it the caller's audio as it arrives. It calls emit() from its own threads with
        {"type": "caller_start", "id": n}           when the caller starts talking
        {"type": "caller_text", "id": n, "text": t} when that utterance is transcribed
    The text is empty when the sound turned out not to be speech."""

    def __init__(self, model, emit: Callable[[dict], None], log: Callable[[str], None] = print):
        self.model = model
        self.emit = emit
        self.log = log
        self.audio: queue.Queue[np.ndarray | None] = queue.Queue()
        self.utterances: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue()
        self.threads = [
            threading.Thread(target=self._detect, daemon=True),
            threading.Thread(target=self._transcribe, daemon=True),
        ]
        for t in self.threads:
            t.start()

    def feed(self, pcm: np.ndarray):
        self.audio.put(pcm)

    def close(self):
        """Stops listening. An utterance still being spoken is dropped; one already handed
        to Whisper finishes, so close() can take as long as one transcription."""
        self.audio.put(None)
        for t in self.threads:
            t.join(timeout=10)

    def _detect(self):
        pending = np.zeros(0, dtype=np.float32)
        before = collections.deque(maxlen=PRE_ROLL)
        recent = collections.deque(maxlen=START[1])
        noise = 0.002
        utterance: list[np.ndarray] | None = None
        quiet = 0
        next_id = 1
        while (chunk := self.audio.get()) is not None:
            pending = np.concatenate([pending, chunk.astype(np.float32, copy=False)])
            while len(pending) >= WINDOW:
                win, pending = pending[:WINDOW], pending[WINDOW:]
                level = float(np.sqrt(np.mean(win**2)))
                loud = level > max(MIN_LEVEL, 4 * noise)
                if not loud:
                    noise = 0.95 * noise + 0.05 * level  # follows the room's background level

                if utterance is None:
                    before.append(win)
                    recent.append(loud)
                    if sum(recent) >= START[0]:
                        utterance, quiet, uid = list(before), 0, next_id
                        next_id += 1
                        recent.clear()
                        self.emit({"type": "caller_start", "id": uid})
                    continue

                utterance.append(win)
                quiet = 0 if loud else quiet + 1
                if quiet >= END_WINDOWS or len(utterance) * WINDOW >= MAX_SECONDS * SAMPLE_RATE:
                    self.utterances.put((uid, np.concatenate(utterance)))
                    utterance = None
                    before.clear()
        self.utterances.put(None)

    def _transcribe(self):
        while (item := self.utterances.get()) is not None:
            uid, pcm = item
            text = ""
            t0 = time.perf_counter()
            try:
                audio = sphn.resample(pcm, src_sample_rate=SAMPLE_RATE, dst_sample_rate=16000)
                segments, _ = self.model.transcribe(
                    audio, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False
                )
                text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()
            except Exception as e:  # a failed transcription shouldn't end the call
                self.log(f"caller transcription failed: {type(e).__name__}: {e}")
            self.log(f"caller said ({len(pcm) / SAMPLE_RATE:.1f}s, {time.perf_counter() - t0:.2f}s to transcribe): {text!r}")
            self.emit({"type": "caller_text", "id": uid, "text": text})
