"""Saved calls: both sides mixed into one Ogg/Opus file, the two sides as a stereo file,
and the conversation as text.

Each call is a folder named by its start time, holding audio.ogg (the mix), tracks.ogg
(caller on the left channel, agent on the right) and call.json.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from pathlib import Path

import numpy as np
import sphn

from .engine import FRAME_SIZE, SAMPLE_RATE

AGENT_PAUSE = 2.0  # seconds without a word that start a new agent turn, as in the UI
_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def tracks(caller: list[np.ndarray], agent: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Both sides on one clock, [2, samples]: the caller's audio as the server received it,
    and each agent frame at the point in the call where the model produced it."""
    caller_pcm = np.concatenate(caller) if caller else np.zeros(0, np.float32)
    end = max([len(caller_pcm)] + [i * FRAME_SIZE + len(pcm) for i, pcm in agent])
    out = np.zeros((2, end), np.float32)
    out[0, : len(caller_pcm)] = caller_pcm
    for i, pcm in agent:
        out[1, i * FRAME_SIZE : i * FRAME_SIZE + len(pcm)] += pcm
    return out


def mix(caller: list[np.ndarray], agent: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """One mono track of both sides."""
    return np.clip(tracks(caller, agent).sum(axis=0), -1, 1)


def conversation(agent_words: list[tuple[float, str]], caller_events: list[dict]) -> list[dict]:
    """The call as turns, in order: {"who": "agent" or "caller", "at": seconds, "text": ...}.
    The agent's words are grouped the way the UI groups them: a new turn after a pause or
    once the caller has started talking, but only at the start of a word, so "?" or "'s"
    stays with the word before it."""
    starts = {e["id"]: e["at"] for e in caller_events if e["type"] == "caller_start"}
    callers = [
        {"who": "caller", "at": starts[e["id"]], "text": e["text"]}
        for e in caller_events
        if e["type"] == "caller_text" and e["text"] and e["id"] in starts
    ]
    caller_times = [c["at"] for c in callers]
    turns: list[dict] = []
    last = None
    interrupted = False  # the caller started talking since the agent's turn began
    for at, piece in agent_words:
        if last is not None and any(last < t <= at for t in caller_times):
            interrupted = True
        if last is None or (piece[:1].isspace() and (at - last > AGENT_PAUSE or interrupted)):
            turns.append({"who": "agent", "at": at, "text": ""})
            interrupted = False
        turns[-1]["text"] += piece
        last = at
    for turn in turns:
        turn["text"] = turn["text"].strip()
    # On a tie the caller goes first: they started talking in the frame the agent spoke in.
    return sorted(turns + callers, key=lambda m: (m["at"], m["who"] == "agent"))


class Recordings:
    def __init__(self, directory: Path):
        self.directory = directory

    def save(
        self,
        *,
        started: float,
        persona: str,
        agent: str,
        voice: str,
        caller_audio: list[np.ndarray],
        agent_audio: list[tuple[int, np.ndarray]],
        agent_words: list[tuple[float, str]],
        caller_events: list[dict],
    ) -> dict | None:
        both = tracks(caller_audio, agent_audio)
        audio = np.clip(both.sum(axis=0), -1, 1)
        if len(audio) < SAMPLE_RATE:  # under a second: nothing worth keeping
            return None
        rid = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}-{secrets.token_hex(2)}"
        folder = self.directory / rid
        folder.mkdir(parents=True)
        sphn.write_opus(str(folder / "audio.ogg"), audio[None], SAMPLE_RATE)
        # The two sides apart (caller left, agent right), for waveforms per speaker or
        # for training data, which wants exactly this layout.
        sphn.write_opus(str(folder / "tracks.ogg"), both, SAMPLE_RATE)
        meta = {
            "id": rid,
            "started": started,
            "duration": round(len(audio) / SAMPLE_RATE, 1),
            "persona": persona,
            "agent": agent,
            "voice": voice,
            "messages": conversation(agent_words, caller_events),
        }
        tmp = folder / "call.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(folder / "call.json")
        return meta

    def all(self) -> list[dict]:
        """Newest first."""
        found = []
        for meta_file in self.directory.glob("*/call.json"):
            if _ID.match(meta_file.parent.name):
                found.append(json.loads(meta_file.read_text()))
        return sorted(found, key=lambda m: m["started"], reverse=True)

    def audio(self, rid: str) -> Path | None:
        path = self.directory / rid / "audio.ogg"
        return path if _ID.match(rid) and path.exists() else None

    def delete(self, rid: str) -> bool:
        folder = self.directory / rid
        if not _ID.match(rid) or not folder.is_dir():
            return False
        shutil.rmtree(folder)
        return True
