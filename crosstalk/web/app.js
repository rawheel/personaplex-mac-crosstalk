// Real estate voice agent UI: listings, a full-duplex call, and viewing bookings.

const $ = (id) => document.getElementById(id);
const state = {
  listings: null,
  defaultPrompt: "",
  call: null, // { ws, ctx, stream, capture, playback }
  transcript: "",
  dismissed: new Set(),
};

const DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
const HOUR_WORDS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];

function setStatus(text, kind = "") {
  const el = $("status");
  el.textContent = text;
  el.className = `pill ${kind}`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

// --- listings and bookings ---

function money(n) {
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

function renderListings() {
  const { listings, agency, agent_name } = state.listings;
  $("agency").textContent = agency;
  $("agent-name").textContent = agent_name.split(" ")[0];
  $("agent-label").textContent = agent_name.split(" ")[0];
  $("listings").replaceChildren(
    ...listings.map((l) => {
      const card = document.createElement("article");
      card.className = "listing";
      card.innerHTML = `
        <div class="top"><span class="addr"></span><span class="price"></span></div>
        <p class="meta"></p><p class="hl"></p><div class="slots"></div>`;
      card.querySelector(".addr").textContent = l.address;
      card.querySelector(".price").textContent = money(l.price);
      card.querySelector(".meta").textContent =
        `${l.neighborhood} · ${l.beds} bd · ${l.baths} ba · ${l.sqft.toLocaleString()} sqft`;
      card.querySelector(".hl").textContent = l.highlights;
      card.querySelector(".slots").replaceChildren(
        ...l.slots.map((s) => {
          const chip = document.createElement("span");
          chip.className = l.free_slots.includes(s) ? "slot" : "slot taken";
          chip.textContent = s;
          return chip;
        }),
      );
      return card;
    }),
  );

  const listingSelect = $("book-listing");
  const current = listingSelect.value;
  listingSelect.replaceChildren(
    ...listings.map((l) => new Option(l.address, l.id)),
  );
  if (current) listingSelect.value = current;
  renderSlotOptions();
}

function renderSlotOptions() {
  const listing = state.listings.listings.find((l) => l.id === $("book-listing").value);
  const slots = listing ? listing.free_slots : [];
  const select = $("book-slot");
  const current = select.value;
  select.replaceChildren(
    ...(slots.length ? slots.map((s) => new Option(s, s)) : [new Option("No times left", "")]),
  );
  // Keep the user's choice when the listings refresh in the background.
  if (slots.includes(current)) select.value = current;
}

async function refreshListings() {
  state.listings = await api("/api/listings");
  renderListings();
}

async function refreshBookings() {
  const bookings = await api("/api/bookings");
  $("bookings").replaceChildren(
    ...bookings.map((b) => {
      const li = document.createElement("li");
      li.innerHTML = `<div><div class="addr"></div><div class="when"></div></div>`;
      li.querySelector(".addr").textContent = b.address;
      li.querySelector(".when").textContent = `${b.slot} · ${b.name}`;
      const cancel = document.createElement("button");
      cancel.className = "link";
      cancel.textContent = "Cancel";
      cancel.onclick = async () => {
        await api(`/api/bookings/${b.id}`, { method: "DELETE" });
        await refreshAll();
      };
      li.append(cancel);
      return li;
    }),
  );
}

async function refreshPrompt({ force = false } = {}) {
  const config = await api("/api/config");
  const promptBox = $("prompt");
  const untouched = promptBox.value === state.defaultPrompt;
  state.defaultPrompt = config.prompt;
  if (force || untouched) promptBox.value = config.prompt;
  return config;
}

async function refreshAll() {
  await Promise.all([refreshListings(), refreshBookings(), refreshPrompt()]);
}

async function book(listingId, slot, name) {
  $("book-error").textContent = "";
  try {
    await api("/api/bookings", {
      method: "POST",
      body: JSON.stringify({ listing_id: listingId, slot, name }),
    });
    await refreshAll();
    return true;
  } catch (e) {
    $("book-error").textContent = e.message;
    return false;
  }
}

// --- spotting a booking in what the agent says ---

// Looks for a listing's street name and one of its free viewing times in the agent's
// recent words, e.g. "...the Maple Street house on Saturday at 10...". Returns the
// most recently mentioned match.
function detectBooking(text) {
  if (!state.listings) return null;
  const recent = text.slice(-400).toLowerCase();
  let best = null;
  for (const l of state.listings.listings) {
    const at = recent.lastIndexOf(l.keyword.toLowerCase());
    if (at < 0) continue;
    for (const slot of l.free_slots) {
      const [day, hour, ampm] = slot.toLowerCase().split(" ");
      if (!DAYS.includes(day)) continue;
      // The day and the hour must be said together ("Sunday at eleven", "10 AM on
      // Saturday"); the latest such mention wins, so a confirmation beats a list of options.
      const time = `\\b(${hour}|${HOUR_WORDS[Number(hour)]})(:00)?\\b(\\s*(${ampm}|${ampm[0]}\\.?\\s*m\\.?|o'?clock))?`;
      const near = "[^.?!]{0,25}?";
      const re = new RegExp(`\\b${day}\\b${near}${time}|${time}${near}\\b${day}\\b`, "g");
      let score = -1;
      for (const m of recent.matchAll(re)) score = m.index + m[0].length;
      if (score < 0) continue;
      if (!best || score > best.score) best = { listing: l, slot, score };
    }
  }
  return best;
}

function updateSuggestion() {
  const match = detectBooking(state.transcript);
  const box = $("suggestion");
  if (!match) return;
  const key = `${match.listing.id}|${match.slot}`;
  if (state.dismissed.has(key)) return;
  box.dataset.key = key;
  box.dataset.listing = match.listing.id;
  box.dataset.slot = match.slot;
  $("suggestion-text").textContent = `${match.listing.address}, ${match.slot}`;
  box.hidden = false;
}

// --- the call ---

function appendTranscript(piece) {
  state.transcript += piece;
  const el = $("transcript");
  el.textContent = state.transcript.trimStart();
  el.scrollTop = el.scrollHeight;
  updateSuggestion();
}

function setCallUi(mode) {
  const btn = $("call-btn");
  btn.classList.toggle("active", mode === "live");
  btn.classList.toggle("connecting", mode === "connecting");
  $("call-label").textContent =
    mode === "live" ? "End call" : mode === "connecting" ? "Connecting…" : "Start call";
}

async function startCall() {
  setCallUi("connecting");
  setStatus("Preparing the agent…");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (e) {
    setStatus("Microphone blocked", "error");
    setCallUi("idle");
    return;
  }

  const ctx = new AudioContext({ latencyHint: "interactive" });
  await ctx.audioWorklet.addModule("/static/audio-worklets.js");
  const source = ctx.createMediaStreamSource(stream);
  const capture = new AudioWorkletNode(ctx, "capture-processor");
  const playback = new AudioWorkletNode(ctx, "playback-processor", { outputChannelCount: [2] });
  source.connect(capture);
  playback.connect(ctx.destination);

  const params = new URLSearchParams({
    codec: "pcm",
    voice_prompt: $("voice").value,
    text_prompt: $("prompt").value,
  });
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/chat?${params}`);
  ws.binaryType = "arraybuffer";
  const call = { ws, ctx, stream, capture, playback, ready: false };
  state.call = call;

  capture.port.onmessage = (e) => {
    if (e.data.level !== undefined) {
      $("mic-level").style.width = `${Math.min(100, e.data.level * 400)}%`;
      return;
    }
    if (call.ready && ws.readyState === WebSocket.OPEN) {
      const msg = new Uint8Array(1 + e.data.pcm.byteLength);
      msg[0] = 1;
      msg.set(new Uint8Array(e.data.pcm), 1);
      ws.send(msg);
    }
  };
  playback.port.onmessage = (e) => {
    $("agent-level").style.width = `${Math.min(100, e.data.level * 400)}%`;
  };

  ws.onmessage = (e) => {
    const data = new Uint8Array(e.data);
    const kind = data[0];
    if (kind === 0) {
      call.ready = true;
      setStatus("Live", "live");
      setCallUi("live");
    } else if (kind === 1) {
      const pcm = new Int16Array(data.slice(1).buffer);
      const f32 = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
      playback.port.postMessage(f32.buffer, [f32.buffer]);
    } else if (kind === 2) {
      const text = new TextDecoder().decode(data.slice(1));
      if (text.startsWith("[")) setStatus(text.slice(1, -1), "error");
      else appendTranscript(text);
    }
  };
  ws.onclose = () => endCall(false);
  ws.onerror = () => setStatus("Connection error", "error");
}

function endCall(closeSocket = true) {
  const call = state.call;
  if (!call) return;
  state.call = null;
  if (closeSocket && call.ws.readyState <= WebSocket.OPEN) call.ws.close();
  call.stream.getTracks().forEach((t) => t.stop());
  call.ctx.close();
  $("mic-level").style.width = "0";
  $("agent-level").style.width = "0";
  setCallUi("idle");
  if (!$("status").classList.contains("error")) setStatus("Ready");
  appendTranscript("\n\n");
  refreshAll();
}

// --- wiring ---

async function init() {
  try {
    const config = await refreshPrompt({ force: true });
    $("voice").replaceChildren(...config.voices.map((v) => new Option(v, v)));
    $("voice").value = config.default_voice;
    await Promise.all([refreshListings(), refreshBookings()]);
    $("call-btn").disabled = false;
    setStatus(config.busy ? "Busy in another tab" : "Ready");
  } catch (e) {
    setStatus("Server unavailable", "error");
  }

  $("call-btn").onclick = () => (state.call ? endCall() : startCall());
  $("book-listing").onchange = renderSlotOptions;
  $("book-form").onsubmit = async (e) => {
    e.preventDefault();
    if (!$("book-slot").value) return;
    await book($("book-listing").value, $("book-slot").value, $("book-name").value);
  };
  $("suggestion-book").onclick = async () => {
    const box = $("suggestion");
    if (await book(box.dataset.listing, box.dataset.slot, $("book-name").value)) box.hidden = true;
  };
  $("suggestion-dismiss").onclick = () => {
    const box = $("suggestion");
    state.dismissed.add(box.dataset.key);
    box.hidden = true;
  };
  $("clear-transcript").onclick = () => {
    state.transcript = "";
    $("transcript").textContent = "";
  };
  $("reset-prompt").onclick = () => refreshPrompt({ force: true });
}

init();
