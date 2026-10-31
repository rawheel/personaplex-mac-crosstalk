"""Web server: the real estate agent UI plus a WebSocket voice API.

WebSocket /api/chat speaks the same binary protocol as NVIDIA's PersonaPlex server, so
their web client and anything built against it work unchanged:
    server -> client  0x00                 ready (system prompts done)
    both directions   0x01 + audio bytes   Ogg/Opus at 24 kHz (or 16-bit PCM, see below)
    server -> client  0x02 + utf-8 text    the agent's words as it speaks them
Query parameters: text_prompt, voice_prompt (e.g. NATF2), seed, and codec=pcm to exchange
raw 16-bit little-endian mono PCM at 24 kHz instead of Opus (used by the bundled UI).
Without text_prompt, the real estate agent prompt is used.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import numpy as np
import rustymimi
import sphn
from aiohttp import web

from .agent import Bookings, build_prompt, load_listings
from .convert import DEFAULT_OUT
from .engine import FRAME_SIZE, SAMPLE_RATE, Engine, set_seed

WEB_DIR = Path(__file__).parent / "web"
DEFAULT_VOICE = "NATF2"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Session:
    """Runs one conversation. All Mimi and model work happens on a dedicated thread so
    the asyncio loop only shuffles bytes."""

    def __init__(self, engine: Engine, voice: str, prompt: str, seed: int | None):
        self.engine = engine
        self.voice = voice
        self.prompt = prompt
        self.seed = seed
        self.pcm_in: queue.Queue[np.ndarray | None] = queue.Queue()
        self.pcm_out: queue.Queue[np.ndarray] = queue.Queue()
        self.text_out: queue.Queue[str] = queue.Queue()
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error: str | None = None
        self.stats = {"frames": 0, "step_ms": 0.0, "slow_frames": 0}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self._loop()
        except Exception as e:  # surfaced to the client by the handler
            self.error = f"{type(e).__name__}: {e}"
            log(f"session error: {self.error}")
        finally:
            self.ready.set()
            self.stop.set()

    def _loop(self):
        engine = self.engine
        set_seed(self.seed)
        gen = engine.generator()
        t0 = time.time()
        gen.system_prompts(engine.voice(self.voice), engine.encode_prompt(self.prompt))
        log(f"system prompts done in {time.time() - t0:.2f}s")
        mimi = rustymimi.StreamTokenizer(engine.mimi_path, num_codebooks=8)
        self.ready.set()

        pending = np.zeros(0, dtype=np.float32)
        encoding = decoding = 0  # frames handed to Mimi's worker threads, not yet returned
        while not self.stop.is_set():
            idle = True
            try:
                chunk = self.pcm_in.get_nowait()
                if chunk is None:
                    break
                pending = np.concatenate([pending, chunk])
                while pending.shape[-1] >= FRAME_SIZE:
                    mimi.encode(pending[:FRAME_SIZE])
                    encoding += 1
                    pending = pending[FRAME_SIZE:]
                idle = False
            except queue.Empty:
                pass

            codes = mimi.get_encoded()
            if codes is not None:
                encoding -= 1
                idle = False
                user = np.asarray(codes).reshape(8).astype(np.int64)  # [8, 1] per frame
                t = time.perf_counter()
                out = gen.step(user)
                ms = (time.perf_counter() - t) * 1000
                self.stats["frames"] += 1
                self.stats["step_ms"] += ms
                if ms > 80:
                    self.stats["slow_frames"] += 1
                if out is not None:
                    piece = engine.text_piece(int(out[0]))
                    if piece:
                        self.text_out.put(piece)
                    mimi.decode(out[None, 1:].astype(np.uint32))  # [1, 8] -> 1920 samples
                    decoding += 1

            decoded = mimi.get_decoded()
            if decoded is not None:
                decoding -= 1
                idle = False
                self.pcm_out.put(np.asarray(decoded, dtype=np.float32).reshape(-1))
            if idle:
                time.sleep(0.002)

        # Collect the frames Mimi is still working on: dropping it mid-frame makes its
        # threads log "sending on a closed channel".
        deadline = time.time() + 1.0
        while (encoding > 0 or decoding > 0) and time.time() < deadline:
            if mimi.get_encoded() is not None:
                encoding -= 1
            elif mimi.get_decoded() is not None:
                decoding -= 1
            else:
                time.sleep(0.002)

    def summary(self) -> str:
        n = self.stats["frames"]
        if n == 0:
            return "no frames"
        avg = self.stats["step_ms"] / n
        return f"{n} frames, avg {avg:.1f} ms/frame (budget 80), {self.stats['slow_frames']} over budget"


class Server:
    def __init__(self, engine: Engine, bookings: Bookings):
        self.engine = engine
        self.bookings = bookings
        self.lock = asyncio.Lock()

    def current_prompt(self) -> str:
        return build_prompt(load_listings(), self.bookings.booked_slots())

    async def config(self, _request):
        return web.json_response(
            {
                "voices": self.engine.voice_names(),
                "default_voice": DEFAULT_VOICE,
                "prompt": self.current_prompt(),
                "busy": self.lock.locked(),
            }
        )

    async def listings(self, _request):
        data = load_listings()
        booked = self.bookings.booked_slots()
        for item in data["listings"]:
            item["free_slots"] = [s for s in item["slots"] if (item["id"], s) not in booked]
        return web.json_response(data)

    async def list_bookings(self, _request):
        return web.json_response(self.bookings.all())

    async def add_booking(self, request):
        body = await request.json()
        try:
            booking = self.bookings.add(body.get("listing_id", ""), body.get("slot", ""), body.get("name", ""))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        log(f"booked {booking['address']} {booking['slot']} for {booking['name']}")
        return web.json_response(booking)

    async def delete_booking(self, request):
        if not self.bookings.remove(request.match_info["id"]):
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"ok": True})

    async def chat(self, request):
        # Browsers let any page open a WebSocket to localhost; only accept our own page
        # (or non-browser clients, which send no Origin header).
        origin = request.headers.get("Origin")
        if origin is not None and urlsplit(origin).netloc != request.host:
            return web.Response(status=403, text="cross-origin connections are not allowed")
        ws = web.WebSocketResponse(max_msg_size=2**22)
        await ws.prepare(request)

        q = request.query
        voice = q.get("voice_prompt", "") or DEFAULT_VOICE
        voice = voice.removesuffix(".pt")
        prompt = q.get("text_prompt", "") or self.current_prompt()
        seed = int(q["seed"]) if q.get("seed") else None
        use_pcm = q.get("codec") == "pcm"
        if voice not in self.engine.voice_names():
            await ws.send_bytes(b"\x02" + f"[unknown voice {voice}]".encode())
            await ws.close()
            return ws

        # A call that just ended may still be shutting down; give it a moment.
        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=5)
        except asyncio.TimeoutError:
            await ws.send_bytes(b"\x02" + "[busy: another conversation is running]".encode())
            await ws.close()
            return ws
        try:
            await self._converse(ws, voice, prompt, seed, use_pcm)
        finally:
            self.lock.release()
        return ws

    async def _converse(self, ws, voice: str, prompt: str, seed: int | None, use_pcm: bool):
        log(f"conversation started (voice {voice}, codec {'pcm' if use_pcm else 'opus'})")
        session = Session(self.engine, voice, prompt, seed)
        session.thread.start()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, session.ready.wait)
        if session.error or ws.closed:
            # Make sure the model thread is done before the lock is released.
            session.stop.set()
            await loop.run_in_executor(None, session.thread.join)
            if session.error and not ws.closed:
                await ws.send_bytes(b"\x02" + f"[error: {session.error}]".encode())
            await ws.close()
            return
        await ws.send_bytes(b"\x00")
        opus_in = sphn.OpusStreamReader(SAMPLE_RATE)
        opus_out = sphn.OpusStreamWriter(SAMPLE_RATE)

        async def recv():
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.BINARY or not msg.data or msg.data[0] != 1:
                    continue
                payload = msg.data[1:]
                if use_pcm:
                    pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                else:
                    pcm = opus_in.append_bytes(payload)
                if pcm.size:
                    session.pcm_in.put(pcm)

        async def send():
            while not ws.closed and not session.stop.is_set():
                sent = False
                while not session.pcm_out.empty():
                    pcm = session.pcm_out.get_nowait()
                    if use_pcm:
                        data = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
                    else:
                        data = opus_out.append_pcm(pcm)
                    if data:
                        await ws.send_bytes(b"\x01" + data)
                    sent = True
                while not session.text_out.empty():
                    await ws.send_bytes(b"\x02" + session.text_out.get_nowait().encode())
                    sent = True
                if not sent:
                    await asyncio.sleep(0.005)

        tasks = [asyncio.create_task(recv()), asyncio.create_task(send())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            session.stop.set()
            session.pcm_in.put(None)
            await loop.run_in_executor(None, session.thread.join)
            log(f"conversation ended: {session.summary()}")
            if not ws.closed:
                await ws.close()

    async def index(self, _request):
        return web.FileResponse(WEB_DIR / "index.html")


def make_app(engine: Engine, bookings: Bookings) -> web.Application:
    server = Server(engine, bookings)
    app = web.Application()
    app.router.add_get("/", server.index)
    app.router.add_get("/api/config", server.config)
    app.router.add_get("/api/listings", server.listings)
    app.router.add_get("/api/bookings", server.list_bookings)
    app.router.add_post("/api/bookings", server.add_booking)
    app.router.add_delete("/api/bookings/{id}", server.delete_booking)
    app.router.add_get("/api/chat", server.chat)
    app.router.add_static("/static", WEB_DIR)
    return app


def main():
    parser = argparse.ArgumentParser(description="Run the crosstalk voice agent server.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bits", type=int, choices=[4, 8, 16], default=4)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--bookings", type=Path, default=DEFAULT_OUT.parent / "bookings.json")
    args = parser.parse_args()

    t0 = time.time()
    engine = Engine(args.model_dir, None if args.bits == 16 else args.bits)
    log(f"model loaded in {time.time() - t0:.1f}s")
    app = make_app(engine, Bookings(args.bookings))
    log(f"open http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
