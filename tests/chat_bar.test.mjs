// Tests for the chat bar's browser logic (button mode, silence detection, resampling, WAV encoding).
// Run with: node --test "tests/*.test.mjs"
import assert from "node:assert/strict";
import { test } from "node:test";

import { buttonMode, encodeWav, resample, SilenceDetector, toBase64 } from "../app/chat_bar.js";

const FRAME_MS = 43;
const frames = (ms, level) => Array(Math.round(ms / FRAME_MS)).fill(level);
// Speech with natural dips between syllables.
const talking = (ms) => Array(Math.round(ms / FRAME_MS)).fill(0).map((_, i) => (i % 5 === 4 ? 0.03 : 0.12));

function run(levels) {
  const detector = new SilenceDetector();
  for (let i = 0; i < levels.length; i += 1) {
    const reason = detector.push(levels[i], i * FRAME_MS);
    if (reason) return { reason, ms: i * FRAME_MS };
  }
  return { reason: null };
}

test("stops ~1.3 s after the speaker finishes", () => {
  const r = run([...frames(800, 0.004), ...frames(2000, 0.08), ...frames(3000, 0.005)]);
  assert.equal(r.reason, "silence");
  assert.ok(r.ms >= 2800 + 1300 && r.ms < 2800 + 1500, `stopped at ${r.ms} ms`);
});

test("a short pause mid-sentence does not stop recording", () => {
  const r = run([...frames(300, 0.004), ...frames(1000, 0.08), ...frames(800, 0.005),
    ...frames(1000, 0.08), ...frames(3000, 0.005)]);
  assert.equal(r.reason, "silence");
  assert.ok(r.ms > 3100, `stopped during the pause at ${r.ms} ms`);
});

test("gives up when nobody speaks, and a single cough is not speech", () => {
  assert.equal(run(frames(9000, 0.005)).reason, "no-speech");
  assert.equal(run([...frames(300, 0.004), 0.2, ...frames(8000, 0.005)]).reason, "no-speech");
});

test("speech that starts immediately after the chime is still detected", () => {
  assert.equal(run([...talking(3000), ...frames(3000, 0.005)]).reason, "silence");
});

test("a noisy room raises the speech threshold", () => {
  const r = run([...frames(300, 0.03), ...frames(1000, 0.06), ...frames(1000, 0.2), ...frames(2000, 0.03)]);
  assert.equal(r.reason, "silence");
  assert.ok(r.ms > 2300, "background-level sound (0.06) must not count as the end of speech");
});

test("recording is capped at 30 s", () => {
  assert.equal(run(talking(35000)).reason, "max");
});

test("resample averages 48 kHz down to 16 kHz", () => {
  const out = resample(Float32Array.from([0.3, 0.3, 0.3, -0.6, -0.6, -0.6]), 48000, 16000);
  assert.deepEqual(Array.from(out, (x) => Math.round(x * 10) / 10), [0.3, -0.6]);
});

test("encodeWav writes a valid 16-bit mono PCM header and clips samples", () => {
  const wav = encodeWav(Float32Array.from([0, 1, -1, 2]), 16000);
  const view = new DataView(wav.buffer);
  const text = (o, n) => String.fromCharCode(...wav.subarray(o, o + n));
  assert.equal(text(0, 4), "RIFF");
  assert.equal(text(8, 4), "WAVE");
  assert.equal(view.getUint16(22, true), 1);       // mono
  assert.equal(view.getUint32(24, true), 16000);   // sample rate
  assert.equal(view.getUint16(34, true), 16);      // bits per sample
  assert.equal(view.getUint32(40, true), 8);       // 4 samples x 2 bytes
  assert.deepEqual([1, 2, 3].map((i) => view.getInt16(44 + i * 2, true)), [32767, -32768, 32767]);
});

test("toBase64 round-trips large buffers", () => {
  const bytes = Uint8Array.from({ length: 100_000 }, (_, i) => i % 256);
  assert.deepEqual(Uint8Array.from(Buffer.from(toBase64(bytes), "base64")), bytes);
});

test("button is a mic when empty, send with text, stop while recording", () => {
  assert.equal(buttonMode("", false), "mic");
  assert.equal(buttonMode("   \n ", false), "mic");          // whitespace only still counts as empty
  assert.equal(buttonMode("Compare LAX and SNA", false), "send");
  assert.equal(buttonMode("", true), "stop");
  assert.equal(buttonMode("typed text", true), "stop");      // recording wins
});
