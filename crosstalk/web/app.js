// Real estate voice agent UI: listings, a full-duplex call, and viewing bookings.

const $ = (id) => document.getElementById(id);
const state = {
  listings: null,
  defaultPrompt: "",
  call: null, // { ws, ctx, stream, capture, playback }
  transcript: "", // everything the agent said, for spotting bookings
  chat: { log: [], callers: new Map(), lastAgentAt: 0 }, // conversation bubbles
  callerTranscript: false, // whether the server transcribes the caller
  dismissed: new Set(),
  voice: null,
  preview: { audio: null, playing: null, loading: null, urls: new Map() },
};

const DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
const HOUR_WORDS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];

// NVIDIA's voice families: NAT voices sound more natural and conversational, VAR more varied.
const VOICE_GROUPS = [
  ["NATF", "Natural, female"],
  ["NATM", "Natural, male"],
  ["VARF", "Variety, female"],
  ["VARM", "Variety, male"],
];

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
      // "1 PM" or "one o'clock" can mean the 1 PM slot; "1 AM" can't.
      const other = ampm === "am" ? "p" : "a";
      const time = `\\b(${hour}|${HOUR_WORDS[Number(hour)]})(:00)?\\b(?!\\s*${other}\\.?\\s*m\\b)(\\s*(${ampm}|${ampm[0]}\\.?\\s*m\\.?|o[’']?clock))?`;
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

// --- voices ---

function renderVoices(names) {
  const groups = VOICE_GROUPS.map(([prefix, label]) => [label, names.filter((n) => n.startsWith(prefix))]);
  const grouped = new Set(groups.flatMap(([, list]) => list));
  groups.push(["Other", names.filter((n) => !grouped.has(n))]);
  $("voices").replaceChildren(
    ...groups
      .filter(([, list]) => list.length)
      .map(([label, list]) => {
        const row = document.createElement("div");
        row.className = "voice-group";
        const title = document.createElement("span");
        title.className = "voice-group-label";
        title.textContent = label;
        const chips = document.createElement("div");
        chips.className = "voice-chips";
        chips.append(
          ...list.map((name) => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "voice-chip";
            chip.dataset.voice = name;
            chip.innerHTML = `<span class="play" aria-hidden="true"></span><span class="name"></span>`;
            chip.querySelector(".name").textContent = name;
            chip.onclick = () => pickVoice(name);
            return chip;
          }),
        );
        row.append(title, chips);
        return row;
      }),
  );
  markVoices();
}

function markVoices() {
  const { playing, loading } = state.preview;
  for (const chip of document.querySelectorAll(".voice-chip")) {
    const name = chip.dataset.voice;
    chip.setAttribute("aria-pressed", String(name === state.voice));
    chip.classList.toggle("playing", name === playing);
    chip.classList.toggle("loading", name === loading);
    chip.title = name === playing ? "Stop" : name === loading ? "Preparing a preview…" : `Pick ${name} and hear it`;
  }
  $("voice-summary").textContent = state.voice ? `· ${state.voice}` : "";
}

function stopPreview() {
  const p = state.preview;
  if (p.audio) p.audio.pause();
  p.audio = null;
  p.playing = null;
}

// Clicking a voice picks it and plays its greeting; clicking it again while it plays stops it.
async function pickVoice(name) {
  const p = state.preview;
  const wasPlaying = p.playing === name;
  stopPreview();
  state.voice = name;
  localStorage.setItem("voice", name);
  $("voice-error").textContent = "";
  markVoices();
  if (wasPlaying || state.call) return; // no previews over a live call

  let url = p.urls.get(name);
  if (!url) {
    p.loading = name;
    markVoices();
    try {
      const res = await fetch(`/api/voices/${encodeURIComponent(name)}/preview`);
      if (!res.ok) throw new Error((await res.json()).error || res.statusText);
      url = URL.createObjectURL(await res.blob());
      p.urls.set(name, url);
    } catch (e) {
      $("voice-error").textContent = `No preview for ${name}: ${e.message}`;
      return;
    } finally {
      if (p.loading === name) p.loading = null;
      markVoices();
    }
    if (state.voice !== name || state.call) return; // moved on while it was being prepared
  }
  const audio = new Audio(url);
  p.audio = audio;
  p.playing = name;
  const done = () => {
    if (p.audio !== audio) return;
    p.audio = null;
    p.playing = null;
    markVoices();
  };
  audio.onended = done;
  audio.play().catch(done);
  markVoices();
}

// --- the conversation ---

// One bubble per turn. The agent's words arrive a token at a time and join its current
// bubble; a new one starts once the caller starts talking or after a pause. The caller's
// bubble appears as they start talking and fills in when the server has transcribed it.
const AGENT_PAUSE_MS = 2000;

function agentName() {
  return state.listings ? state.listings.agent_name.split(" ")[0] : "Agent";
}

// Keeps the newest message in view, unless the user has scrolled up to read.
function keepScrolled(change) {
  const el = $("transcript");
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  change();
  if (atBottom) el.scrollTop = el.scrollHeight;
}

function addBubble(who, text) {
  const el = document.createElement("div");
  el.className = `msg ${who}`;
  const name = document.createElement("span");
  name.className = "who";
  name.textContent = who === "agent" ? agentName() : "You";
  const body = document.createElement("p");
  body.textContent = text.trimStart();
  el.append(name, body);
  $("transcript").append(el);
  const entry = { who, el, body, text };
  state.chat.log.push(entry);
  return entry;
}

function setText(entry, text) {
  entry.text = text;
  entry.body.textContent = text.trimStart();
}

function addAgentWords(piece) {
  state.transcript += piece;
  const chat = state.chat;
  const now = performance.now();
  const last = chat.log.at(-1);
  keepScrolled(() => {
    if (last?.who === "agent" && now - chat.lastAgentAt < AGENT_PAUSE_MS) setText(last, last.text + piece);
    else addBubble("agent", piece);
  });
  chat.lastAgentAt = now;
  updateSuggestion();
}

function callerStarted(id) {
  keepScrolled(() => {
    const entry = addBubble("caller", "…");
    entry.el.classList.add("pending");
    state.chat.callers.set(id, entry);
  });
}

function callerSaid(id, text) {
  const chat = state.chat;
  const entry = chat.callers.get(id);
  if (!entry) return;
  chat.callers.delete(id);
  if (text) {
    keepScrolled(() => setText(entry, text));
    entry.el.classList.remove("pending");
    return;
  }
  // Not speech after all (a cough, a bump): drop the bubble and rejoin the agent's words
  // on either side of it.
  const i = chat.log.indexOf(entry);
  entry.el.remove();
  chat.log.splice(i, 1);
  const [prev, next] = [chat.log[i - 1], chat.log[i]];
  if (prev?.who === "agent" && next?.who === "agent") {
    setText(prev, prev.text + next.text);
    next.el.remove();
    chat.log.splice(i, 1);
  }
}

function addDivider(text) {
  for (const id of [...state.chat.callers.keys()]) callerSaid(id, ""); // never transcribed
  if (!state.chat.log.length) return;
  const el = document.createElement("div");
  el.className = "divider";
  el.textContent = text;
  keepScrolled(() => $("transcript").append(el));
  state.chat.log.push({ who: "divider", el });
}

function clearConversation() {
  state.transcript = "";
  state.chat = { log: [], callers: new Map(), lastAgentAt: 0 };
  $("transcript").replaceChildren();
}

// --- the call ---

function setCallUi(mode) {
  const btn = $("call-btn");
  btn.classList.toggle("active", mode === "live");
  btn.classList.toggle("connecting", mode === "connecting");
  $("call-label").textContent =
    mode === "live" ? "End call" : mode === "connecting" ? "Connecting…" : "Start call";
}

async function startCall() {
  stopPreview();
  markVoices();
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
    voice_prompt: state.voice,
    text_prompt: $("prompt").value,
  });
  if (state.callerTranscript) params.set("caller_transcript", "1");
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
      else addAgentWords(text);
    } else if (kind === 3) {
      const event = JSON.parse(new TextDecoder().decode(data.slice(1)));
      if (event.type === "caller_start") callerStarted(event.id);
      else if (event.type === "caller_text") callerSaid(event.id, event.text);
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
  state.transcript += "\n\n";
  addDivider("Call ended");
  refreshAll();
}

// --- wiring ---

async function init() {
  try {
    const config = await refreshPrompt({ force: true });
    const saved = localStorage.getItem("voice");
    state.voice = config.voices.includes(saved) ? saved : config.default_voice;
    state.callerTranscript = config.caller_transcript;
    renderVoices(config.voices);
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
  $("clear-transcript").onclick = clearConversation;
  $("reset-prompt").onclick = () => refreshPrompt({ force: true });
}

init();
