// Chat input bar with voice: one button that is a mic when the box is empty, send when there is
// text, and stop while recording. Voice mode plays a chime, shows a level meter and stops by
// itself once the speaker goes quiet; the recording goes to Python as a base64 16 kHz mono WAV
// ("audio" trigger), typed text as the "text" trigger. Pure helpers are exported for tests.

const TARGET_RATE = 16000;         // what Whisper expects
const FRAME_SIZE = 2048;           // samples per analysis frame (~43 ms at 48 kHz)
const PLACEHOLDER = "Ask about US airports...";
const MAX_TEXTAREA_PX = 200;

export const VAD_DEFAULTS = {
  calibrationMs: 300,              // measure background noise before listening for speech
  minThreshold: 0.015,             // RMS floor, so a silent room doesn't make the threshold ~0
  maxThreshold: 0.05,              // cap, so talking during calibration can't make speech "too quiet"
  noiseMultiplier: 3,              // speech must be this many times louder than the noise floor
  speechFrames: 3,                 // consecutive loud frames that count as the start of speech
  endSilenceMs: 1300,              // stop after this much quiet once speech has started
  noSpeechTimeoutMs: 7000,         // give up if nobody starts talking
  maxMs: 30000,                    // hard cap on recording length
};

// Decides when to stop. push() takes one frame's loudness and returns null (keep recording) or a
// reason: "silence" (speaker finished), "no-speech" or "max".
export class SilenceDetector {
  constructor(options = {}) {
    this.o = { ...VAD_DEFAULTS, ...options };
    this.noiseFloor = Infinity;
    this.threshold = this.o.minThreshold;
    this.loudRun = 0;
    this.speaking = false;
    this.lastLoudMs = 0;
  }

  push(rms, nowMs) {
    const o = this.o;
    if (nowMs < o.calibrationMs) {
      // The quietest calibration frame is the noise estimate: people often start talking right
      // after the chime, and an average would count their speech as background noise.
      this.noiseFloor = Math.min(this.noiseFloor, rms);
      this.threshold = Math.min(o.maxThreshold,
        Math.max(o.minThreshold, o.noiseMultiplier * this.noiseFloor));
    }
    if (rms > this.threshold) {
      this.loudRun += 1;
      this.lastLoudMs = nowMs;
      if (this.loudRun >= o.speechFrames) this.speaking = true;
    } else {
      this.loudRun = 0;
    }
    if (nowMs >= o.maxMs) return "max";
    if (this.speaking && nowMs - this.lastLoudMs >= o.endSilenceMs) return "silence";
    if (!this.speaking && nowMs >= o.noSpeechTimeoutMs) return "no-speech";
    return null;
  }
}

export function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

// Downsample by averaging each output sample's input window (a box filter limits aliasing).
export function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(samples.length, Math.max(start + 1, Math.floor((i + 1) * ratio)));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += samples[j];
    out[i] = sum / (end - start);
  }
  return out;
}

export function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);          // fmt chunk size
  view.setUint16(20, 1, true);           // PCM
  view.setUint16(22, 1, true);           // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);    // byte rate
  view.setUint16(32, 2, true);           // block align
  view.setUint16(34, 16, true);          // bits per sample
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

export function toBase64(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

// Which action the button performs: stop while recording, send when there's text, else mic.
export function buttonMode(text, recording) {
  if (recording) return "stop";
  return text.trim() ? "send" : "mic";
}

// Two short tones: rising when listening starts, falling when it stops.
function chime(ctx, frequencies) {
  const toneSeconds = 0.09;
  let t = ctx.currentTime;
  let last;
  for (const f of frequencies) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = f;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.2, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + toneSeconds);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + toneSeconds);
    t += toneSeconds + 0.02;
    last = osc;
  }
  return new Promise((resolve) => { last.onended = resolve; });
}

const svg = (path) => `<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">${path}</svg>`;
const ICONS = {
  mic: svg('<path fill="currentColor" d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z"/>'),
  send: svg('<path fill="currentColor" d="M12 4l-7 7 1.41 1.41L11 7.83V20h2V7.83l4.59 4.58L19 11z"/>'),
  stop: svg('<rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor"/>'),
};
const LABELS = { mic: "Ask by voice", send: "Send", stop: "Stop recording" };

export default function (component) {
  const { parentElement, setTriggerValue } = component;
  const bar = parentElement.querySelector(".bar");
  const textarea = parentElement.querySelector("textarea");
  const button = parentElement.querySelector("button");
  const meter = parentElement.querySelector(".meter > div");
  const status = parentElement.querySelector(".status");
  let session = null;      // active recording: { stop(reason) }
  let starting = false;    // between the click and the end of the start chime
  let unmounted = false;   // set by the cleanup function when Streamlit removes the component

  const render = () => {
    const mode = buttonMode(textarea.value, Boolean(session));
    if (button.dataset.mode !== mode) {
      button.dataset.mode = mode;
      button.innerHTML = ICONS[mode];
      button.setAttribute("aria-label", LABELS[mode]);
      button.title = LABELS[mode];
    }
    bar.classList.toggle("recording", Boolean(session));
    textarea.disabled = Boolean(session);
    if (!session) {
      textarea.placeholder = PLACEHOLDER;
      meter.style.width = "0%";
    }
  };

  const autosize = () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_TEXTAREA_PX)}px`;
  };

  const send = () => {
    const text = textarea.value.trim();
    if (!text) return;
    textarea.value = "";
    autosize();
    render();
    setTriggerValue("text", text);
  };

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    let ctx = null;
    try {
      ctx = new AudioContext();
      await chime(ctx, [660, 880]);  // finish the chime before recording so it isn't captured
      if (unmounted) throw new Error("chat bar was removed while starting");
    } catch (err) {
      // Release the microphone on any setup failure, or the browser's mic indicator stays on.
      stream.getTracks().forEach((t) => t.stop());
      if (ctx) ctx.close();
      throw err;
    }
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(FRAME_SIZE, 1, 1);
    const detector = new SilenceDetector();
    const chunks = [];
    const t0 = performance.now();
    let stopped = false;

    const stop = (reason) => {
      if (stopped) return;
      stopped = true;
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      session = null;
      if (reason === "unmount") { ctx.close(); return; }
      chime(ctx, [880, 520]).then(() => ctx.close());
      render();
      if (reason === "no-speech") {
        status.textContent = "Didn't hear anything. Press the mic to try again.";
        return;
      }
      const total = chunks.reduce((n, c) => n + c.length, 0);
      const samples = new Float32Array(total);
      let offset = 0;
      for (const c of chunks) { samples.set(c, offset); offset += c.length; }
      setTriggerValue("audio", toBase64(encodeWav(resample(samples, ctx.sampleRate, TARGET_RATE), TARGET_RATE)));
    };

    processor.onaudioprocess = (event) => {
      if (stopped) return;
      const frame = new Float32Array(event.inputBuffer.getChannelData(0));
      chunks.push(frame);
      const level = rms(frame);
      meter.style.width = `${Math.min(100, level * 600)}%`;
      const reason = detector.push(level, performance.now() - t0);
      if (detector.speaking) textarea.placeholder = "Listening... stops when you pause";
      if (reason) stop(reason);
    };
    source.connect(processor);
    processor.connect(ctx.destination);  // required for onaudioprocess to fire; outputs silence

    session = { stop };
    textarea.placeholder = "Listening... speak now";
    render();
  }

  textarea.oninput = () => { status.textContent = ""; autosize(); render(); };
  textarea.onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send();
    }
  };
  button.onclick = async () => {
    const mode = buttonMode(textarea.value, Boolean(session));
    if (mode === "send") { send(); return; }
    if (mode === "stop") { session.stop("manual"); return; }
    if (starting) return;
    starting = true;
    status.textContent = "";
    try {
      await startRecording();
    } catch (err) {
      if (unmounted) return;
      status.textContent = err && err.name === "NotAllowedError"
        ? "Microphone access was blocked. Allow it in the browser to ask by voice."
        : `Couldn't start the microphone: ${err && err.message ? err.message : err}`;
    } finally {
      starting = false;
    }
  };

  render();
  return () => {
    unmounted = true;
    if (session) session.stop("unmount");
  };
}
