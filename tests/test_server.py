"""Server tests with a tiny random model, the real tokenizer and the real Mimi codec.

Needs the tokenizer and Mimi files from nvidia/personaplex-7b-v1 (small downloads).
    .venv/bin/python tests/test_server.py
"""

import asyncio
import io
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sentencepiece
import sphn
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crosstalk.agent import Bookings  # noqa: E402
from crosstalk.convert import HF_REPO, MIMI, TEXT_TOKENIZER  # noqa: E402
from crosstalk.engine import Engine, Generator  # noqa: E402
from crosstalk.previews import PreviewCache  # noqa: E402
from crosstalk.server import make_app  # noqa: E402
from test_engine import make  # noqa: E402


class TinyEngine(Engine):
    def __init__(self):
        self.model, self._voice = make()
        self.tokenizer = sentencepiece.SentencePieceProcessor(hf_hub_download(HF_REPO, TEXT_TOKENIZER))
        self.mimi_path = hf_hub_download(HF_REPO, MIMI)

    def voice_names(self):
        return ["NATF2"]

    def voice(self, name):
        return self._voice

    def generator(self, **kw):
        return Generator(self.model, **kw)


class FakeWhisper:
    """Stands in for faster-whisper: says how much audio it was handed."""

    def transcribe(self, audio, **_):
        return [SimpleNamespace(text=f" {len(audio) / 16000:.1f} seconds", no_speech_prob=0.1)], None


async def caller_transcript(client: TestClient):
    """Speaks for 1 s, pauses for 1.5 s, and expects the utterance back as text."""
    ws = await client.ws_connect("/api/chat?codec=pcm&voice_prompt=NATF2&caller_transcript=1")
    assert await ws.receive_bytes(timeout=60) == b"\x00"
    t = np.arange(24000) / 24000
    speech = np.concatenate([0.2 * np.sin(2 * np.pi * 220 * t), np.zeros(36000)]).astype(np.float32)
    events = []
    for i in range(0, len(speech), 480):
        await ws.send_bytes(b"\x01" + (speech[i : i + 480] * 32767).astype("<i2").tobytes())
        await asyncio.sleep(0.02)
        while True:
            try:
                m = await ws.receive_bytes(timeout=0.001)
            except asyncio.TimeoutError:
                break
            if m[0] == 3:
                events.append(json.loads(m[1:]))
    await ws.close()
    return events


async def call(client: TestClient, codec: str, seconds: float = 3.0):
    ws = await client.ws_connect(f"/api/chat?codec={codec}&voice_prompt=NATF2&seed=1")
    msg = await ws.receive_bytes(timeout=60)
    assert msg == b"\x00", msg
    writer = sphn.OpusStreamWriter(24000)
    reader = sphn.OpusStreamReader(24000)
    t = np.arange(int(24000 * seconds)) / 24000
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    received, texts = 0, 0
    for i in range(0, len(audio), 1920):
        chunk = audio[i : i + 1920]
        if codec == "pcm":
            payload = (chunk * 32767).astype("<i2").tobytes()
        else:
            payload = writer.append_pcm(chunk)
        if payload:
            await ws.send_bytes(b"\x01" + payload)
        await asyncio.sleep(0.02)
        while True:
            try:
                m = await ws.receive_bytes(timeout=0.001)
            except asyncio.TimeoutError:
                break
            if m[0] == 1:
                received += (len(m) - 1) // 2 if codec == "pcm" else reader.append_bytes(m[1:]).size
            elif m[0] == 2:
                texts += 1
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        try:
            m = await ws.receive_bytes(timeout=0.2)
        except asyncio.TimeoutError:
            continue
        if m[0] == 1:
            received += (len(m) - 1) // 2 if codec == "pcm" else reader.append_bytes(m[1:]).size
    await ws.close()
    return received, texts


async def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = make_app(
            TinyEngine(), Bookings(Path(tmp) / "bookings.json"), PreviewCache(Path(tmp) / "previews"), FakeWhisper()
        )
        async with TestClient(TestServer(app)) as client:
            cfg = await (await client.get("/api/config")).json()
            assert cfg["voices"] == ["NATF2"] and "Harbor Lane Realty" in cfg["prompt"] and cfg["caller_transcript"]
            assert (await client.get("/")).status == 200
            assert (await client.get("/static/app.js")).status == 200

            r = await client.post("/api/bookings", json={"listing_id": "maple", "slot": "Saturday 10 AM", "name": "Ana"})
            booking = await r.json()
            assert r.status == 200, booking
            r = await client.post("/api/bookings", json={"listing_id": "maple", "slot": "Saturday 10 AM", "name": "Bo"})
            assert r.status == 400  # double booking refused
            cfg = await (await client.get("/api/config")).json()
            maple_times = cfg["prompt"].split("Maple Street viewings: ")[1].split(".")[0]
            assert maple_times == "Saturday at 2 PM, Sunday at 11 AM", maple_times
            listings = await (await client.get("/api/listings")).json()
            assert "Saturday 10 AM" not in listings["listings"][0]["free_slots"]
            assert (await client.delete(f"/api/bookings/{booking['id']}")).status == 200
            print("bookings API and prompt refresh: OK")

            t0 = time.perf_counter()
            r = await client.get("/api/voices/NATF2/preview")
            first = time.perf_counter() - t0
            assert r.status == 200 and r.headers["Content-Type"] == "audio/wav", r.status
            with wave.open(io.BytesIO(await r.read())) as w:
                assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (24000, 1, 2)
                seconds = w.getnframes() / 24000
            t0 = time.perf_counter()
            r = await client.get("/api/voices/NATF2/preview")
            assert r.status == 200
            print(f"voice preview: {seconds:.2f}s WAV, rendered in {first:.2f}s, cached in {time.perf_counter() - t0:.3f}s: OK")
            assert (await client.get("/api/voices/NOPE/preview")).status == 404

            r = await client.get("/api/chat", headers={"Origin": "https://evil.example", "Upgrade": "websocket"})
            assert r.status == 403
            print("cross-origin WebSocket refused: OK")

            for codec in ("pcm", "opus"):
                received, texts = await call(client, codec)
                print(f"{codec} call: received {received / 24000:.2f}s of agent audio, {texts} text messages")
                assert received > 24000, f"expected over 1 s of audio back, got {received} samples"

            events = await caller_transcript(client)
            assert [e["type"] for e in events] == ["caller_start", "caller_text"], events
            assert events[0]["id"] == events[1]["id"]
            print(f"caller transcript: {events[1]['text']!r} for 1 s of speech: OK")


if __name__ == "__main__":
    asyncio.run(main())
