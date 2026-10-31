// Audio worklets for the voice call. Both resample between the AudioContext's native rate
// and the model's 24 kHz, so the page works whatever rate the browser picks.

const MODEL_RATE = 24000;
const FRAME = 1920; // 80 ms at 24 kHz, one model step

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / MODEL_RATE;
    this.pos = 0; // fractional read position in `this.input`
    this.input = new Float32Array(0);
    this.out = new Int16Array(FRAME);
    this.outLen = 0;
    this.levelAcc = 0;
    this.levelCount = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    const merged = new Float32Array(this.input.length + channel.length);
    merged.set(this.input);
    merged.set(channel, this.input.length);
    this.input = merged;

    while (this.pos + 1 < this.input.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const s = this.input[i] * (1 - frac) + this.input[i + 1] * frac;
      this.levelAcc += s * s;
      this.levelCount += 1;
      this.out[this.outLen++] = Math.max(-1, Math.min(1, s)) * 32767;
      if (this.outLen === FRAME) {
        this.port.postMessage({ pcm: this.out.buffer }, [this.out.buffer]);
        this.out = new Int16Array(FRAME);
        this.outLen = 0;
        this.port.postMessage({ level: Math.sqrt(this.levelAcc / this.levelCount) });
        this.levelAcc = 0;
        this.levelCount = 0;
      }
      this.pos += this.ratio;
    }
    const consumed = Math.floor(this.pos);
    this.input = this.input.slice(consumed);
    this.pos -= consumed;
    return true;
  }
}

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = MODEL_RATE / sampleRate;
    this.buffer = new Float32Array(MODEL_RATE * 4);
    this.read = 0;
    this.write = 0;
    this.started = false;
    this.port.onmessage = (e) => {
      if (e.data === "clear") {
        this.read = this.write = 0;
        this.started = false;
        return;
      }
      this.push(new Float32Array(e.data));
    };
  }

  available() {
    return this.write - this.read;
  }

  push(samples) {
    // Compact, and drop the oldest audio if we fell more than a second behind.
    if (this.available() + samples.length > MODEL_RATE) {
      this.read = this.write - Math.floor(MODEL_RATE / 4);
    }
    if (this.write + samples.length > this.buffer.length) {
      const left = this.buffer.slice(Math.floor(this.read), this.write);
      this.buffer.set(left, 0);
      this.write = left.length;
      this.read -= Math.floor(this.read);
    }
    this.buffer.set(samples, this.write);
    this.write += samples.length;
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    // Wait for ~100 ms of audio before starting, to absorb network and model jitter.
    if (!this.started && this.available() >= 2400) this.started = true;
    let level = 0;
    for (let n = 0; n < out.length; n++) {
      if (!this.started || this.read + 1 >= this.write) {
        out[n] = 0;
        this.started = false;
        continue;
      }
      const i = Math.floor(this.read);
      const frac = this.read - i;
      const s = this.buffer[i] * (1 - frac) + this.buffer[i + 1] * frac;
      out[n] = s;
      level += s * s;
      this.read += this.step;
    }
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out);
    if (currentFrame % 4096 < out.length) {
      this.port.postMessage({ level: Math.sqrt(level / out.length) });
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
registerProcessor("playback-processor", PlaybackProcessor);
