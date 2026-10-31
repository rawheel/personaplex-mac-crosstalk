"""Recordings: mixing both sides onto one clock, and grouping the call into turns.

    .venv/bin/python tests/test_recordings.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosstalk.engine import FRAME_SIZE  # noqa: E402
from crosstalk.recordings import conversation, mix  # noqa: E402


def test_mix():
    caller = [np.full(FRAME_SIZE, 0.1, np.float32) for _ in range(4)]
    agent = [(2, np.full(FRAME_SIZE, 0.2, np.float32)), (5, np.full(FRAME_SIZE, 0.3, np.float32))]
    out = mix(caller, agent)
    assert len(out) == 6 * FRAME_SIZE  # the agent's last frame runs past the caller's audio
    frame = lambda k: out[k * FRAME_SIZE : (k + 1) * FRAME_SIZE]  # noqa: E731
    assert np.allclose(frame(0), 0.1) and np.allclose(frame(2), 0.3)  # overlap adds up
    assert np.allclose(frame(4), 0.0) and np.allclose(frame(5), 0.3)
    print("mix: agent frames land where they were produced: OK")


def test_conversation():
    words = [(0.2, " Hello,"), (0.4, " this"), (0.6, " is"), (0.8, " Jess."),  # greeting
             (3.0, " Oh,"), (3.2, " nice."),  # after the caller starts at 2.0
             (8.0, " So"), (8.2, " anyway.")]  # after a long pause
    callers = [
        {"type": "caller_start", "id": 1, "at": 2.0},
        {"type": "caller_text", "id": 1, "text": "Hi, I loved Sinners."},
        {"type": "caller_start", "id": 2, "at": 5.0},
        {"type": "caller_text", "id": 2, "text": ""},  # a cough: dropped
    ]
    turns = conversation(words, callers)
    assert [(t["who"], t["text"]) for t in turns] == [
        ("agent", "Hello, this is Jess."),
        ("caller", "Hi, I loved Sinners."),
        ("agent", "Oh, nice."),
        ("agent", "So anyway."),
    ], turns
    assert [t["at"] for t in turns] == [0.2, 2.0, 3.0, 8.0]

    # From a real call: the caller starts in the frame where the agent's "?" lands.
    words = [(0.16, " Hello"), (0.24, "?"), (0.4, " Hi."), (6.7, " I"), (6.8, " guess.")]
    callers = [{"type": "caller_start", "id": 1, "at": 0.24}, {"type": "caller_text", "id": 1, "text": "Best show ever?"}]
    turns = conversation(words, callers)
    assert [(t["who"], t["text"]) for t in turns] == [
        ("agent", "Hello?"),
        ("caller", "Best show ever?"),
        ("agent", "Hi."),
        ("agent", "I guess."),
    ], turns
    print("conversation: turns split by the caller and by pauses, never mid-word: OK")


if __name__ == "__main__":
    test_mix()
    test_conversation()
