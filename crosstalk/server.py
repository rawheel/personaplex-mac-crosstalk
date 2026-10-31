"""Web server: the real estate agent UI plus a WebSocket voice API.

WebSocket /api/chat speaks the same binary protocol as NVIDIA's PersonaPlex server, so
their web client and anything built against it work unchanged:
    server -> client  0x00                 ready (system prompts done)
    both directions   0x01 + audio bytes   Ogg/Opus at 24 kHz (or 16-bit PCM, see below)
    server -> client  0x02 + utf-8 text    the agent's words as it speaks them
Query parameters: text_prompt, voice_prompt (e.g. NATF2), seed, and codec=pcm to exchange
raw 16-bit little-endian mono PCM at 24 kHz instead of Opus (used by the bundled UI).
Without text_prompt, the prompt of `persona` is used (realestate, the default, or movies).

With caller_transcript=1 the server also transcribes the caller and sends
    server -> client  0x03 + utf-8 JSON    {"type": "caller_start", "id": n} as they start
                                           talking, then {"type": "caller_text", "id": n,
                                           "text": ...} once it's transcribed (empty if
                                           it wasn't speech)

With record=1 the call is saved when it ends: both sides mixed into one Ogg/Opus file plus
the conversation as text. GET /api/recordings lists them (newest first; wait=1 first waits
for a call that's ending to be saved), GET /api/recordings/{id}/audio plays one, DELETE
/api/recordings/{id} removes it.

GET /api/voices/{voice}/preview?persona=... returns a WAV of that persona greeting in that
voice. The first request renders it with the model (a few seconds); after that it's served
from disk.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
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
from .movies import build_prompt as movies_prompt
from .movies import load_movies
from .movies import preview_prompt as movies_preview_prompt
from .previews import PreviewCache, preview_prompt
from .recordings import Recordings
from .transcribe import CallerTranscriber, load_model

WEB_DIR = Path(__file__).parent / "web"
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_VOICE = "NATF2"
DEFAULT_PERSONA = "realestate"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def same_site_only(allowed_hosts: frozenset[str] | None):
    """Only this server's own page may use it. Websites you visit can send requests to
    localhost too, so every request is checked:
    - Host must be a loopback name. This stops DNS rebinding, where a site points its own
      domain at 127.0.0.1 so the browser treats it as the same origin as this server, and
      then reads your recordings. Pass None to accept any host (the server was bound to a
      network address on purpose).
    - Origin, when sent, must be this server. This stops other sites from booking viewings,
      deleting recordings or opening calls. Clients outside a browser send no Origin."""

    @web.middleware
    async def middleware(request, handler):
        host = urlsplit(f"//{request.host}").hostname or ""
        if allowed_hosts is not None and host not in allowed_hosts:
            return web.Response(status=403, text="unknown host")
        origin = request.headers.get("Origin")
        if origin is not None and urlsplit(origin).netloc != request.host:
            return web.Response(status=403, text="cross-origin requests are not allowed")
        return await handler(request)

    return middleware


class Session:
    """Runs one conversation. All Mimi and model work happens on a dedicated thread so
    the asyncio loop only shuffles bytes."""

    def __init__(
        self,
        engine: Engine,
        voice: str,
        prompt: str,
        seed: int | None,
        record: bool = False,
        repeat_penalty: float = 0.0,
    ):
        self.engine = engine
        self.voice = voice
        self.prompt = prompt
        self.seed = seed
        self.record = record
        self.repeat_penalty = repeat_penalty
        # For recordings, on one clock: frame k is caller audio from k * 80 ms.
        self.caller_audio: list[np.ndarray] = []
        self.agent_audio: list[tuple[int, np.ndarray]] = []  # (frame it was produced at, pcm)
        self.agent_words: list[tuple[float, str]] = []  # (seconds, piece)
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
        gen = engine.generator(repeat_penalty=self.repeat_penalty)
        t0 = time.time()
        gen.system_prompts(engine.voice(self.voice), engine.encode_prompt(self.prompt))
        log(f"system prompts done in {time.time() - t0:.2f}s")
        mimi = rustymimi.StreamTokenizer(engine.mimi_path, num_codebooks=8)
        self.ready.set()

        pending = np.zeros(0, dtype=np.float32)
        encoding = 0  # frames handed to Mimi's encoder thread, not yet returned
        decoding = collections.deque()  # same for its decoder: the frame each was produced at
        while not self.stop.is_set():
            idle = True
            try:
                chunk = self.pcm_in.get_nowait()
                if chunk is None:
                    break
                pending = np.concatenate([pending, chunk])
                while pending.shape[-1] >= FRAME_SIZE:
                    frame, pending = pending[:FRAME_SIZE], pending[FRAME_SIZE:]
                    mimi.encode(frame)
                    encoding += 1
                    if self.record:
                        self.caller_audio.append(frame)
                idle = False
            except queue.Empty:
                pass

            codes = mimi.get_encoded()
            if codes is not None:
                encoding -= 1
                idle = False
                user = np.asarray(codes).reshape(8).astype(np.int64)  # [8, 1] per frame
                k = self.stats["frames"]
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
                        self.agent_words.append((round(k * FRAME_SIZE / SAMPLE_RATE, 2), piece))
                    mimi.decode(out[None, 1:].astype(np.uint32))  # [1, 8] -> 1920 samples
                    decoding.append(k)

            decoded = mimi.get_decoded()
            if decoded is not None:
                k = decoding.popleft()
                idle = False
                pcm = np.asarray(decoded, dtype=np.float32).reshape(-1)
                self.pcm_out.put(pcm)
                if self.record:
                    self.agent_audio.append((k, pcm))
            if idle:
                time.sleep(0.002)

        # Collect the frames Mimi is still working on: dropping it mid-frame makes its
        # threads log "sending on a closed channel".
        deadline = time.time() + 1.0
        while (encoding > 0 or decoding) and time.time() < deadline:
            if mimi.get_encoded() is not None:
                encoding -= 1
            elif mimi.get_decoded() is not None:
                decoding.popleft()
            else:
                time.sleep(0.002)

    def summary(self) -> str:
        n = self.stats["frames"]
        if n == 0:
            return "no frames"
        avg = self.stats["step_ms"] / n
        return f"{n} frames, avg {avg:.1f} ms/frame (budget 80), {self.stats['slow_frames']} over budget"


class Server:
    def __init__(
        self, engine: Engine, bookings: Bookings, previews: PreviewCache, recordings: Recordings, whisper=None
    ):
        self.engine = engine
        self.bookings = bookings
        self.previews = previews
        self.recordings = recordings
        self.whisper = whisper  # transcribes the caller; None turns that off
        self.lock = asyncio.Lock()  # one user of the model at a time: a call or a preview
        self.in_call = False

    def personas(self) -> dict[str, dict]:
        """The agents the UI can call. Read fresh each time, so edits to the data files and
        new bookings show up without a restart."""
        listings, movies = load_listings(), load_movies()
        maya = listings["agent_name"].split()[0]
        return {
            "realestate": {
                "label": "Real estate agent",
                "name": maya,
                "title": listings["agency"],
                "tagline": f"Talk to {maya}, our sales agent, and book a viewing",
                "voice": DEFAULT_VOICE,
                "prompt": build_prompt(listings, self.bookings.booked_slots()),
                "preview_prompt": preview_prompt(listings),
                "repeat_penalty": 0.0,  # confirming a booking repeats the time on purpose
            },
            "movies": {
                "label": "Movie buff",
                "name": movies["name"],
                "title": movies["title"],
                "tagline": movies["tagline"],
                "voice": movies["voice"],
                "prompt": movies_prompt(movies),
                "preview_prompt": movies_preview_prompt(movies),
                "repeat_penalty": movies["repeat_penalty"],
                "topics": [{k: t[k] for k in ("title", "kind", "years", "note")} for t in movies["topics"]],
            },
        }

    async def config(self, _request):
        personas = self.personas()
        for p in personas.values():
            del p["preview_prompt"], p["repeat_penalty"]
        return web.json_response(
            {
                "voices": self.engine.voice_names(),
                "personas": personas,
                "default_persona": DEFAULT_PERSONA,
                "busy": self.in_call,
                "caller_transcript": self.whisper is not None,
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
        ws = web.WebSocketResponse(max_msg_size=2**22)
        await ws.prepare(request)

        q = request.query
        persona_id = q.get("persona", DEFAULT_PERSONA)
        persona = self.personas().get(persona_id)
        if persona is None:
            await ws.send_bytes(b"\x02" + f"[unknown persona {q['persona']}]".encode())
            await ws.close()
            return ws
        voice = q.get("voice_prompt", "") or persona["voice"]
        voice = voice.removesuffix(".pt")
        prompt = q.get("text_prompt", "") or persona["prompt"]
        seed = int(q["seed"]) if q.get("seed") else None
        use_pcm = q.get("codec") == "pcm"
        transcribe = q.get("caller_transcript") == "1" and self.whisper is not None
        record = {"persona": persona_id, "agent": persona["name"]} if q.get("record") == "1" else None
        repeat_penalty = persona["repeat_penalty"]
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
        self.in_call = True
        try:
            await self._converse(ws, voice, prompt, seed, use_pcm, transcribe, record, repeat_penalty)
        finally:
            self.in_call = False
            self.lock.release()
        return ws

    async def voice_preview(self, request):
        voice = request.match_info["voice"]
        if voice not in self.engine.voice_names():
            return web.json_response({"error": f"unknown voice {voice}"}, status=404)
        persona = self.personas().get(request.query.get("persona", DEFAULT_PERSONA))
        if persona is None:
            return web.json_response({"error": "unknown persona"}, status=404)
        prompt = persona["preview_prompt"]
        path = self.previews.path(voice, prompt)
        if not path.exists():
            if self.in_call:
                return web.json_response({"error": "new voices can be previewed after the call"}, status=409)
            # Previews clicked in quick succession queue up behind each other.
            try:
                await asyncio.wait_for(self.lock.acquire(), timeout=60)
            except asyncio.TimeoutError:
                return web.json_response({"error": "the model is busy, try again"}, status=409)
            job = None
            try:
                if not path.exists():
                    t0 = time.time()
                    job = asyncio.get_running_loop().run_in_executor(
                        None, self.previews.make, self.engine, voice, prompt
                    )
                    await asyncio.shield(job)
                    log(f"voice preview {voice} rendered in {time.time() - t0:.1f}s")
            except Exception as e:
                log(f"voice preview {voice} failed: {type(e).__name__}: {e}")
                return web.json_response({"error": f"could not render the preview: {e}"}, status=500)
            finally:
                # Hold the lock until the render thread is done, even if this request was
                # cancelled, so a call never shares the model with it.
                if job is not None and not job.done():
                    await asyncio.wait([job])
                self.lock.release()
        return web.FileResponse(path, headers={"Content-Type": "audio/wav"})

    async def _converse(
        self,
        ws,
        voice: str,
        prompt: str,
        seed: int | None,
        use_pcm: bool,
        transcribe: bool,
        record: dict | None,
        repeat_penalty: float,
    ):
        log(f"conversation started (voice {voice}, codec {'pcm' if use_pcm else 'opus'})")
        session = Session(self.engine, voice, prompt, seed, record=record is not None, repeat_penalty=repeat_penalty)
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
        started = time.time()
        opus_in = sphn.OpusStreamReader(SAMPLE_RATE)
        opus_out = sphn.OpusStreamWriter(SAMPLE_RATE)
        caller_events: queue.Queue[dict] = queue.Queue()  # for the client
        caller_log: list[dict] = []  # for the recording, including what arrives after hang-up

        def on_caller(event: dict):
            caller_events.put(event)
            caller_log.append(event)

        caller = CallerTranscriber(self.whisper, on_caller, log) if transcribe else None

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
                    if caller:
                        caller.feed(pcm)

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
                while not caller_events.empty():
                    await ws.send_bytes(b"\x03" + json.dumps(caller_events.get_nowait()).encode())
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
            if caller:
                await loop.run_in_executor(None, caller.close)
            log(f"conversation ended: {session.summary()}")
            if not ws.closed:
                await ws.close()
            if record:
                await self._save_recording(session, started, voice, record, caller_log)

    async def _save_recording(self, session: Session, started: float, voice: str, record: dict, caller_log: list):
        try:
            meta = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.recordings.save(
                    started=started,
                    voice=voice,
                    caller_audio=session.caller_audio,
                    agent_audio=session.agent_audio,
                    agent_words=session.agent_words,
                    caller_events=caller_log,
                    **record,
                ),
            )
        except Exception as e:  # never lose the server over a recording
            log(f"could not save the recording: {type(e).__name__}: {e}")
            return
        if meta:
            log(f"saved recording {meta['id']} ({meta['duration']:.0f}s, {len(meta['messages'])} turns)")

    async def list_recordings(self, request):
        # Right after hanging up, the call may still be saving; wait for it if asked.
        if request.query.get("wait") == "1":
            deadline = time.time() + 15
            while self.in_call and time.time() < deadline:
                await asyncio.sleep(0.1)
        return web.json_response(self.recordings.all())

    async def recording_audio(self, request):
        path = self.recordings.audio(request.match_info["id"])
        if path is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(path, headers={"Content-Type": "audio/ogg"})

    async def delete_recording(self, request):
        if not self.recordings.delete(request.match_info["id"]):
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"ok": True})

    async def index(self, _request):
        return web.FileResponse(WEB_DIR / "index.html")


def make_app(
    engine: Engine,
    bookings: Bookings,
    previews: PreviewCache,
    recordings: Recordings,
    whisper=None,
    allowed_hosts: frozenset[str] | None = LOOPBACK,
) -> web.Application:
    server = Server(engine, bookings, previews, recordings, whisper)
    app = web.Application(middlewares=[same_site_only(allowed_hosts)])
    app.router.add_get("/", server.index)
    app.router.add_get("/api/config", server.config)
    app.router.add_get("/api/listings", server.listings)
    app.router.add_get("/api/bookings", server.list_bookings)
    app.router.add_post("/api/bookings", server.add_booking)
    app.router.add_delete("/api/bookings/{id}", server.delete_booking)
    app.router.add_get("/api/chat", server.chat)
    app.router.add_get("/api/voices/{voice}/preview", server.voice_preview)
    app.router.add_get("/api/recordings", server.list_recordings)
    app.router.add_get("/api/recordings/{id}/audio", server.recording_audio)
    app.router.add_delete("/api/recordings/{id}", server.delete_recording)
    app.router.add_static("/static", WEB_DIR)
    return app


def main():
    parser = argparse.ArgumentParser(description="Run the crosstalk voice agent server.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bits", type=int, choices=[4, 8, 16], default=4)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--bookings", type=Path, default=DEFAULT_OUT.parent / "bookings.json")
    parser.add_argument("--recordings", type=Path, default=DEFAULT_OUT.parent / "recordings")
    parser.add_argument(
        "--whisper", default="small.en", help="Whisper model that transcribes the caller (e.g. base.en, small.en)"
    )
    parser.add_argument("--no-caller-transcript", action="store_true", help="show only the agent's words")
    args = parser.parse_args()

    t0 = time.time()
    engine = Engine(args.model_dir, None if args.bits == 16 else args.bits)
    log(f"model loaded in {time.time() - t0:.1f}s")
    whisper = None
    if not args.no_caller_transcript:
        t0 = time.time()
        whisper = load_model(args.whisper)  # downloaded on first use (small.en is about 480 MB)
        log(f"caller transcription: whisper {args.whisper}, loaded in {time.time() - t0:.1f}s")
    if args.host in LOOPBACK:
        allowed_hosts = LOOPBACK
    elif args.host in ("0.0.0.0", "::"):
        allowed_hosts = None
    else:
        allowed_hosts = LOOPBACK | {args.host}
    if args.host not in LOOPBACK:
        log(
            f"warning: listening on {args.host}. There is no login: anyone who can reach this "
            "address can talk to the agent, listen to your recordings and read your bookings."
        )
    app = make_app(
        engine,
        Bookings(args.bookings),
        PreviewCache(args.model_dir / "previews"),
        Recordings(args.recordings),
        whisper,
        allowed_hosts,
    )
    log(f"open http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
