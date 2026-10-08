# Airport Investment Intelligence Agent

A chat agent that helps analysts at an airport-modernization investor find US airports where
renovation is likely to pay off, because demand is pressing against capacity and still growing.

It pulls public aviation data (BTS, FAA, OurAirports), ranks airports with a deterministic
**Investment Opportunity Score**, and uses an LLM to answer questions and explain the results. You
can type or speak your questions, and ask follow-ups. Examples:

- *Which airports in New England are strong candidates for terminal expansion?*
- *Compare LA and Santa Ana airport congestion levels.*
- *What is the percentage of long haul flights out of Anchorage airport?*
- *What is the unmet flight demand in SFO airport and why?*

Design, scoring method and tradeoffs: [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Quick start

About 5 minutes, mostly installing packages. The aviation data ships with the repo, so there's no
data download.

### What you need

| | |
|---|---|
| **Python 3.10 or newer** | Developed on 3.12 |
| **An API key for one LLM provider** | Anthropic, OpenAI or Google Gemini. Or a local model through Ollama, which needs no key |
| **Internet access** | To install packages and call the LLM. The data works offline |
| **About 1 GB of disk** | Mostly Python packages (~650 MB) |
| Node.js 22+ *(optional)* | Only to run the chat bar's JavaScript tests |

Commands below are for macOS and Linux. On Windows, use `.venv\Scripts\` instead of `.venv/bin/`.

### 1. Install

From the project folder, first check the Python version:

```bash
python3 --version      # must be 3.10 or newer; if not, use a newer one, e.g. python3.12
```

Then create an environment and install the packages:

```bash
python3 -m venv .venv          # or: python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If you use [uv](https://docs.astral.sh/uv/), this is faster:
`uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

### 2. Add an API key

Pick **one** of these:

- **In a file:** copy the template, then fill in the key for your provider:
  ```bash
  cp .env.example .env
  ```
  For a provider other than Anthropic, also set `LLM_PROVIDER` in `.env`: `openai`, `gemini` or
  `openai-compatible`. `.env` is git-ignored.
- **In the app:** skip this step and paste the key into the sidebar under **Model settings** (step 3).
  It's kept only in that browser session's memory and never written to disk.

| Provider | Default model | Key in `.env` |
|---|---|---|
| Anthropic Claude | `claude-sonnet-5-5` (also `claude-opus-5-5`, `claude-haiku-4-5`) | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-5.5` | `OPENAI_API_KEY` |
| Google Gemini | `gemini-3.8-flash` | `GEMINI_API_KEY` |
| Ollama, vLLM, LM Studio (any OpenAI-compatible server) | type a model name | optional `LLM_API_KEY`; set `LLM_BASE_URL` |

You can type any model name into the app's model box.

### 3. Start the app

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

It opens at **http://localhost:8501**. Click one of the example questions in the sidebar, or type
your own. Press `Ctrl+C` in the terminal to stop the app.

**Prefer the terminal?** These use the provider and key from `.env`:

```bash
.venv/bin/python -m scripts.ask --examples              # the four example questions
.venv/bin/python -m scripts.ask "Compare BOS and PVD"   # one question
.venv/bin/python -m scripts.ask                         # interactive chat with follow-ups
```

### 4. Check everything works (optional)

```bash
.venv/bin/python -m pytest tests          # Python tests; no network or API key needed
node --test "tests/*.test.mjs"            # chat bar tests (Node 22+)
```

---

### About the data

The agent gets its data from public APIs (BTS, FAA, OurAirports). To spare you a ~5-minute
download, the repo includes a compressed snapshot of those API responses (`data/snapshot/`,
3.6 MB, fetched 2026-10-08). The app reads the snapshot when it has nothing newer, and answers
always cite the date the data was fetched. Live FAA status is never in the snapshot; it's always
fetched live.

To fetch everything from the APIs yourself (~5 minutes) and, optionally, refresh the snapshot:

```bash
.venv/bin/python -m scripts.fetch_data --from-apis              # download; ignore the snapshot
.venv/bin/python -m scripts.fetch_data --from-apis --snapshot   # ...and rebuild data/snapshot/
```

## Using the app

- **Typing or speaking.** The chat bar has one button: a **🎤 mic** while the box is empty, which
  becomes **↑ send** once you type. Enter sends; Shift+Enter starts a new line.
- **Voice.** Press the mic. A rising chime means it's listening; just speak. It stops on its own
  about 1.3 seconds after you finish, with a falling chime, and sends your question. Press **■** to
  stop early. The browser asks for microphone permission the first time. Speech is transcribed on
  your machine by Whisper: no API key needed, and audio isn't sent anywhere. The first recording
  downloads a ~150 MB speech model.
- **Follow-ups.** The agent remembers the conversation, so you can ask "re-rank them with growth
  weighted 50%" or "why is Boston excluded?". **New conversation** in the sidebar starts over, and so
  does changing the provider or model.
- **Reading answers.** Each answer shows which tools it called (the 🔧 box) and ends with an
  **Assumptions and confidence** note: the data period, the assumptions that matter, and how sure it
  is.

## Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| "Set up a model to start chatting" | No API key found | Paste one under *Model settings*, or put it in `.env` and restart |
| "The API key was rejected" | Wrong key, or key for a different provider | Check the provider and key in *Model settings* |
| "Model ... was not found" | Model name not available for your key | Choose another model in *Model settings* |
| First page load spins for minutes | The bundled snapshot is missing, so data is downloading | Wait, or run `python -m scripts.fetch_data` to see progress |
| Long-haul answers say "approximation" | Route data unavailable (no snapshot and the download failed) | Run `python -m scripts.fetch_data` while online; everything else still works |
| Mic does nothing / "Microphone access was blocked" | Browser permission denied | Allow the microphone for localhost in the browser settings |
| "Voice transcription failed: ..." | Speech package missing, or the speech model can't download (e.g. offline) | Reinstall with `pip install -r requirements.txt` while online; typing still works |
| Answers mention a "stale copy" | A data refresh failed (e.g. offline), so cached data was used | Nothing to fix; reconnect and it refreshes on its own |
| "Port 8501 is already in use" | Another app is running | `streamlit run app/streamlit_app.py --server.port 8502` |

---

## Where to start reading

The reasoning lives in a small core; the rest is plumbing and interface.

1. [`docs/DESIGN.md`](docs/DESIGN.md): thesis, how each requirement in the brief is met, the
   scoring method, tradeoffs and limitations.
2. [`src/scoring/score.py`](src/scoring/score.py): the Investment Opportunity Score (gate, weighted
   percentile rank, unmet-demand flag). Pure functions, about 140 lines.
3. [`src/agent/tools.py`](src/agent/tools.py): what the LLM can call. Each tool wraps
   deterministic code and returns its evidence and data sources.
4. [`src/agent/agent.py`](src/agent/agent.py): the system prompt. It requires grounded numbers, an
   assumptions-and-confidence note, and clear scope.

The chat bar with voice input (`app/chat_bar.*`) is the brief's *bonus* feature, kept separate so it
doesn't touch the core.

## For code reviewers and AI agents

### Project map

| Path | Responsibility |
|---|---|
| `src/config.py` | All tunable values: data URLs, weights, gate and flag thresholds, regions, cache lifetimes. Each threshold has a comment with its reasoning |
| `src/data/sources/*.py` | One module per public source; each returns plain data through the cache |
| `src/data/cache.py` | Disk cache with fetch dates. Lookup order: runtime cache (`data/cache/`, git-ignored) → bundled snapshot (`data/snapshot/`, committed, read-only) → API. Serves a stale copy if a refresh fails |
| `src/data/repository.py` | Joins the sources into one `AirportMetrics` per airport, with provenance |
| `src/scoring/metrics.py` | Pure functions: raw monthly rows → per-airport metrics |
| `src/scoring/score.py` | Pure functions: metrics → gate, percentile components, score, unmet-demand flag |
| `src/agent/tools.py` | Tool definitions the LLM can call; validation and error handling |
| `src/agent/agent.py` | System prompt and conversation state |
| `src/llm/` | Provider interface (`base.py`), presets (`factory.py`), Claude and OpenAI-compatible tool loops |
| `src/voice/stt.py` | Local speech-to-text |
| `app/streamlit_app.py` | Streamlit UI: sidebar, chat history, running a turn |
| `app/chat_bar.{py,js,css}` | Chat bar component: mic/send button, chime, silence detection |
| `scripts/` | `fetch_data.py` (fetch data from the APIs; rebuild the snapshot), `ask.py` (terminal chat) |
| `data/snapshot/` | Committed, gzipped API responses so a fresh clone works without downloading |
| `tests/` | See the table below |

### Invariants to preserve

1. **The LLM never produces a number.** Every figure in an answer comes from a tool result, and
   scoring lives in `src/scoring/` as pure, deterministic functions with no network or LLM access.
2. **Tools return errors; they don't raise.** `run_tool` turns bad input and failures into
   `{"error": ...}` with `is_error`, so the model can recover.
3. **Conversation history is append-only and stays valid.** Every tool call in the history must have
   a matching result, even when a reply is cut off, or every later turn fails. Both providers close
   unanswered tool calls.
4. **Live data never affects a score.** FAA status is context only, so rankings are reproducible.
   It's also never written to the snapshot.
5. **The snapshot is read-only at runtime.** All writes go to `data/cache/`. Only
   `scripts/fetch_data.py --from-apis --snapshot` rebuilds `data/snapshot/`, and only from data just
   fetched from the APIs.
6. **Answers state assumptions, confidence and scope.** The system prompt requires it; tools supply
   the inputs (`assumed_components`, coverage notes, stale-copy flags).
7. **Secrets stay out of the repo.** `.env` and `data/cache/` are git-ignored, and keys pasted in
   the UI live only in session memory.

### How to verify a change

```bash
.venv/bin/python -m pytest tests     # offline: no network, no API key
node --test "tests/*.test.mjs"       # offline
uvx ruff check .                     # lint; Ruff >= 0.16, settings in pyproject.toml
```

End-to-end checks (`scripts.ask --examples`, the app) need network access and an API key.

| Tests | What they cover |
|---|---|
| `tests/test_scoring.py` | Gate, weights, percentiles, missing-data assumption, unmet-demand flag (incl. the absolute delay floor), cause-based congestion, metric calculations, route profile |
| `tests/test_providers.py` | Tool loops with fake model responses: streamed tool calls reassembled, results sent back, malformed arguments reported to the model, history kept valid when a reply stops mid tool call; provider presets |
| `tests/test_tools_and_cache.py` | Tool input validation; concurrent cache writes |
| `tests/test_voice.py` | WAV decoding for transcription |
| `tests/chat_bar.test.mjs` | Mic/send/stop button logic, silence detection scenarios, resampling, WAV encoding |

The chat bar was also tested end to end in headless Chromium with a recorded question fed in as the
microphone. That test isn't in the repo because it needs a browser download and a live API key.

### Configuration

Scoring weights and thresholds are in `src/config.py`; the agent can also re-rank with custom
weights per question. LLM settings are documented in `.env.example`. Claude requests on supported
models opt into Anthropic's server-side fallback (`fallbacks: "default"`), which retries a request
declined by a safety classifier on a fallback model. To turn it off, edit
`src/llm/anthropic_provider.py`.
