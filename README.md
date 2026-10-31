# personaplex-mac-crosstalk

Full-duplex voice agents that run entirely on your Mac. The agent listens and talks at
the same time: you can interrupt it, it backchannels ("mm-hm", "got it"), and it never
waits for a "turn". It runs NVIDIA's [PersonaPlex](https://github.com/NVIDIA/personaplex)
7B speech-to-speech model on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

The bundled demo is a real estate sales agent: browse the listings, call the agent, and
book viewings.

## Speed

Each step produces 80 ms of audio, so a step must take under 80 ms to keep up in real
time. On a MacBook Pro M4 Pro (24 GB), measured with `crosstalk-bench speed`:

| Weights | Per 80 ms frame | Peak memory | Call start (prompt of ~290 tokens) |
|---|---|---|---|
| 4-bit (default) | 26 ms (p90 26.4, max 30.4) | 7.2 GB | 1.1 s |
| 8-bit | 38 ms | 10.3 GB | 1.7 s |

Processing the system prompt one frame at a time, as NVIDIA's server does, takes 6.7 s
(4-bit) and 8.2 s (8-bit). The 8-bit row was measured with random weights of the same
shapes, which run at the same speed.

## Requirements

- Apple Silicon Mac with 16 GB+ of memory (tested on an M4 Pro with 24 GB)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- About 23 GB of free disk for the one-time conversion (6 GB afterwards)
- A Hugging Face account that has accepted the
  [PersonaPlex license](https://huggingface.co/nvidia/personaplex-7b-v1)

## Setup

```bash
git clone https://github.com/rawheel/personaplex-mac-crosstalk
cd personaplex-mac-crosstalk
uv venv --python 3.12 && uv pip install -e .

uvx --from huggingface_hub hf auth login      # paste a read token
.venv/bin/crosstalk-convert --delete-original # download, convert to 4-bit, free 16.7 GB
```

`crosstalk-convert` downloads NVIDIA's 16.7 GB checkpoint once, converts it to a 4-bit MLX
model (5.2 GB in `~/.cache/crosstalk`) and converts the 18 voices. With
`--delete-original` it then removes the PyTorch checkpoint. Use `--bits 8` for higher
quality if your Mac is fast enough (see `crosstalk-bench speed`).

## Run

```bash
.venv/bin/crosstalk-serve
```

Open http://localhost:8998, put on headphones and press **Start call**.

- **Listings** come from `crosstalk/data/listings.json`. Edit it to use your own homes,
  prices and viewing times. The agent's role prompt is generated from it; `name` is how
  the agent refers to a home and `keyword` is what the UI listens for.
- **Booking**: when the agent confirms an address and a time, a "Heard in the call" card
  offers to book it. You can also book with the form. Booked times are taken out of the
  agent's prompt for the next call, so it only offers free slots. Bookings are stored in
  `~/.cache/crosstalk/bookings.json`.
- **Voice and prompt**: pick one of 18 voices and edit the prompt under "Voice and prompt".

## API

`ws://localhost:8998/api/chat` speaks the same binary protocol as NVIDIA's PersonaPlex
server, so clients written for it work unchanged:

| Byte | Direction | Payload |
|---|---|---|
| `0x00` | server to client | ready, system prompt processed |
| `0x01` | both | audio: Ogg/Opus at 24 kHz, or 16-bit PCM with `codec=pcm` |
| `0x02` | server to client | the agent's words as UTF-8, as it speaks |

Query parameters: `text_prompt`, `voice_prompt` (e.g. `NATF2`), `seed`, `codec`. One
conversation runs at a time. Connections from other websites are refused.

## How it differs from NVIDIA's code

- **Runs on MLX** (Metal) instead of CUDA, with 4-bit or 8-bit weights.
- **Converted once.** Weights are converted and quantized once instead of at every start,
  and voices are converted to safetensors, so the app never needs PyTorch. Voice files are
  read with an allow-listed unpickler, so a tampered `.pt` file can't run code.
- **Half the depformer work.** NVIDIA samples 16 audio codebooks per frame and discards the
  8 that predict the user's audio. Only the agent's 8 are sampled here; the output is the
  same.
- **Prefilled system prompts.** Voice and role prompts are processed in two batched passes
  instead of one step per 80 ms frame, so calls start faster. `crosstalk-bench verify-prefill`
  checks the result matches the frame-by-frame path.
- **Bounded memory.** The attention cache is a fixed ring buffer of the model's 3000-frame
  context, so long calls don't grow memory.

## Tools

```bash
.venv/bin/crosstalk-bench speed            # ms per 80 ms frame
.venv/bin/crosstalk-bench verify-prefill   # batched prompt == frame-by-frame prompt
.venv/bin/crosstalk-bench offline in.wav out.wav --prompt "You enjoy having a good conversation."
```

`offline` runs a recorded caller through the agent and writes the agent's audio plus
`out.json` with its words and when it said them, which is handy for testing prompts.

## Limitations

- PersonaPlex can't call tools or look things up. It only knows what's in its prompt, and
  bookings are recorded by the UI, not by the model.
- It gets details wrong sometimes. In 15 scripted test calls it quoted prices and
  described the homes reliably, but it now and then offered another home's viewing time
  or confirmed a time without naming the home. The UI only offers to book times that
  really exist for the home it heard, so a mix-up can't create an impossible booking.
  Keep the listing count small: four homes is already a lot for a 7B model.
- The on-screen transcript is the model's own text stream, one token per 80 ms, so
  contractions sometimes come out garbled ("you' all set"). The audio is unaffected.
- English only. No echo cancellation beyond what the browser provides: use headphones.

## Credits and licenses

Code: MIT, see [LICENSE](LICENSE). Built on Kyutai's moshi_mlx and NVIDIA's PersonaPlex
code (both MIT). Model weights: NVIDIA Open Model License, not included here. See
[NOTICE.md](NOTICE.md).
