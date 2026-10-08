# Airport Investment Intelligence Agent — Design

**Author:** Amit.N.Shani@gmail.com · **Date:** 2026-10-08 · **Status:** built and tested on live data

Abbreviations are explained in §9.

## 1. Goal, thesis and scope

A chat agent that helps analysts screen US airports for renovation investment driven by increased
flight and passenger capacity.

**Thesis:** renovation pays off where demand is **pressing against capacity and still growing**.
Size alone is not the signal. A large airport whose airlines are cutting service is a weak bet; a
congested, full, growing one is a strong bet.

**Scope.** It is a first-pass screen: it surfaces candidates and explains why. It does **not**
measure terminal or gate capacity (no public source exists), estimate construction costs or
returns, forecast traffic, or cover non-US airports. The agent says so when asked.

## 2. How the brief's requirements are met

| Requirement | How |
|---|---|
| Use public APIs for aviation data | BTS (Socrata API and public download files), FAA status API, OurAirports (§3). A snapshot of the responses ships with the repo to skip the first download; `fetch_data --from-apis` runs the full API path |
| Rank or compare airports on a defined KPI | Investment Opportunity Score: a qualifying gate, then a weighted rank (§4) |
| Deterministic scoring, not only LLM output | All metrics and ranks are computed in Python and unit-tested. The LLM never computes or invents a number (§5) |
| Explain reasoning clearly | Every score returns its component breakdown, gate result and evidence list; the agent leads with the answer, then the evidence |
| Conversational follow-ups | Conversation history, including earlier tool results, is resent each turn ("re-rank with growth at 50%" works) |
| Chat interface (voice bonus) | Streamlit chat. Voice through a mic button, transcribed locally |
| Communicate assumptions, uncertainty and scope | Every data-backed answer ends with an **Assumptions and confidence** note: data window, the assumptions that matter for that question, and a high/medium/low confidence with the reason. Tools return what drives confidence: assumed components, partial coverage, stale or approximate sources. Out-of-scope questions are answered as far as the data allows, with the rest declined explicitly |

### Interpretations of the brief

Where the brief leaves a definition open, the build follows the interpretation below. Each would be
a small, local change if a different meaning was intended.

| The brief says | Interpretation | Why | If a different meaning was intended |
|---|---|---|---|
| "renovations will be **most profitable**" | A demand-side screen: airports where demand presses against capacity and is still growing | Profit needs costs and revenues, which no public source provides per airport. Unserved demand that capacity could capture is the driver of renovation returns | Add a financial layer on top of the score. An illustrative revenue figure (Passenger Facility Charges) is already returned, unscored |
| "what is the **unmet flight demand** in SFO" | A yes/no flag with its evidence (full planes, capacity delay, growth), not a count | Unserved demand isn't observed in public data; a count would rest on a capacity figure we can't source | Add an estimate, e.g. passengers above a stated capacity assumption |
| "compare **LA** and Santa Ana" | LA = Los Angeles International (LAX) | It's the airport usually meant by "LA" | The agent can already compare Burbank, Long Beach and Ontario on request |
| "percentage of **long haul** flights" | Departures over 3,000 miles, shown for all flights and split into passenger and cargo | No definition is given. 3,000 miles is longer than any coast-to-coast flight within the lower 48 states (New York–Los Angeles is about 2,500). Anchorage is mostly cargo, so the split changes the answer (30% vs 3%) | The threshold is a parameter; answers also show 1,500, 2,200 and 4,000 miles |
| "use **public APIs**" | BTS's Socrata API plus BTS's public download files, the FAA status API and OurAirports. A snapshot of the responses ships with the repo | BTS publishes delay causes and route distances only as files | `fetch_data --from-apis` runs on the APIs alone; the snapshot can be deleted |
| "**congestion** levels" | Capacity-caused (NAS) delay per arriving flight; all-cause delays as context | All-cause delays mix in airline problems and snowstorms (§4) | All-cause figures are already returned by the tools |
| "**New England**" | Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont | The US Census definition | Region lists are in `src/config.py` |
| "terminal expansion" | Judged on passenger demand, load factor and growth | Terminal capacity itself isn't in any public source | Add a curated gate-capacity table |
| Data recency (not specified) | The latest 12 months BTS has published (about a five-month lag), compared with the 12 before | Twelve months removes seasonality | The window length is a setting |

## 3. Data (public sources)

| Source | Access | Used for |
|---|---|---|
| OurAirports | HTTP CSV | IATA codes, regions (New England = CT/ME/MA/NH/RI/VT), names |
| BTS T-100 by origin (`data.bts.gov` `r495-tyji`) | Socrata JSON API | Monthly passengers, seats, departures, load factor, international split |
| BTS On-Time Performance | Monthly zip files | Delays and cancellations per airport, by cause |
| BTS T-100 segment (TranStats table FMG) | Download form | Route distances → long-haul share, passenger vs cargo |
| FAA NAS Status | XML API | Live delays and ground stops (context only) |

The data window is the trailing 12 months (TTM, May 2025 to Apr 2026), compared with the 12 months
before. BTS lags about five months. The two BTS download files aren't JSON endpoints, but they're the
only public source of delay causes and route distances.

**Caching and the bundled snapshot.** API responses are cached with their fetch date, which answers
cite. Data for a fixed past period never changes, so it never expires; new BTS months arrive through
a weekly "latest month" check. The repo ships those responses as a compressed, read-only snapshot
(`data/snapshot/`, 3.6 MB). A fresh clone therefore works immediately, and offline, without the
~5-minute first download. Lookup order: runtime cache → snapshot → API. Live FAA status is never
snapshotted. `scripts/fetch_data.py --from-apis` exercises the full API path.

## 4. Scoring: Investment Opportunity Score

**Stage 1, qualifying gate:** TTM passenger growth > 0 and load factor ≥ 70%. This makes
"constrained AND growing" explicit; a plain weighted sum would let size offset shrinking traffic.
The 70% floor is deliberately loose. Airports that fail are listed with the reason, never dropped
silently.

**Stage 2, weighted rank** of percentiles against a fixed set, the top 75 US airports, so a score
means the same thing in every question:

| Component | Weight | Metric (TTM) | Why |
|---|---|---|---|
| Congestion | 0.35 | NAS-caused delay minutes per arriving flight | Delay caused by airport and air-traffic capacity: the problem renovation fixes |
| Utilization | 0.25 | Load factor | Full planes mean demand close to supply |
| Growth | 0.25 | Passengers and departures vs prior TTM | Capacity built now must be used for decades |
| Scale | 0.15 | Passengers | Size of the prize; a tiebreaker, not a driver |

- **Missing data** (usually congestion at airports whose airlines don't report delays) is assumed to
  be at the median, and every answer says so.
- **Unmet-demand flag:** load factor ≥ 80%, congestion percentile ≥ 0.60, at least 4.0 NAS minutes
  per arrival, and growth ≥ 0. It is returned with the evidence for each test.
- **Not scored:** live FAA status, which changes by the minute and would make rankings
  irreproducible; and an illustrative Passenger Facility Charge figure (passengers × $4.50).
- Weights and thresholds are configurable (`src/config.py`), and the agent can re-rank with custom
  weights.

**What testing on live data changed.** Each change came from a result that looked wrong:
1. **All-cause → NAS-only congestion.** Snowstorm cancellations made Portland, Maine look twice as
   congested as SFO. NAS delay counts only airport and air-traffic causes. NAS cancellations (1–4 a
   month at small airports) and taxi-out time (dominated by de-icing) were tried and rejected.
2. **Dropping missing components → assuming the median.** Dropping them let HVN rank first in New
   England *because* its congestion was unknown.
3. **Percentile only → also a 4-minute floor for the flag.** Most large airports sit at 2–3 minutes,
   so the 60th percentile is just 2.7 minutes, and Provo (2.95) was flagged alongside Denver (5.6).
   The floor sits in the largest gap in the upper range (Houston 4.5, Miami 3.7).

**Sanity check and results.** The airports with the most NAS delay per arrival are EWR, LGA, BOS,
ORD, DCA, SFO and JFK: the industry's known constrained airports. On the brief's questions:
- **SFO** scores 0.91 and is flagged for unmet demand (82.6% load factor, +3.4% growth, 6.2 NAS min).
- **New England:** nothing is flagged. The growing small airports have low congestion, and Boston is
  congested but fails the growth gate with falling departures, which suggests airlines cutting
  service rather than a capacity ceiling.
- **LAX vs SNA:** both have low congestion (2.0 vs 1.7 min); SNA grows (+5.5%), LAX shrinks (−3.3%).
- **Anchorage:** 30% of departures fly over 3,000 miles, but only 3% of passenger flights do;
  56% of departures are all-cargo.

## 5. Where AI is used

| LLM | Deterministic code |
|---|---|
| Understands the question, picks and chains tools, explains results, handles follow-ups, flags uncertainty | Data retrieval, metrics, gate, ranking, unmet-demand flag |

**Tools:** `search_airports`, `get_airport_profile`, `compare_airports`, `score_airports`,
`get_route_profile`, `get_live_status`, `get_methodology`.

**System-prompt rules:** every number comes from a tool; background knowledge, such as SNA's
passenger cap, is labeled as such; the metric must match the question (congestion means NAS delay,
long-haul means route distances, split passenger and cargo); answers end with Assumptions and
confidence; scope is stated (§1).

**Model.** Provider-agnostic: native Claude (default `claude-sonnet-5-5`), plus one OpenAI-compatible
implementation covering OpenAI, Gemini and local models. The reviewer picks a provider and pastes a
key in the sidebar. Sonnet rather than Opus because code does the scoring and the model only
orchestrates and explains, so speed matters more than peak reasoning.

**Voice.** The chat bar's single button is a mic when the box is empty and send when there's text.
Recording plays a chime and stops on its own when the speaker pauses; detection runs in the
browser. Whisper transcribes locally, so voice needs no API key and audio stays on the machine.
Answers are text only, because tables and numbers read better than they sound.

| Brief question | Tools |
|---|---|
| New England terminal expansion | `score_airports(region)`; gate exclusions explained |
| LAX vs SNA congestion | `compare_airports` on NAS delay; all-cause delays as context |
| ANC long-haul % | `get_route_profile`; stated mileage threshold, passenger vs cargo split |
| SFO unmet demand | `get_airport_profile`; flag with evidence |

## 6. Key tradeoffs

1. **Gate + rank vs a weighted sum.** The gate makes the thesis explicit. The cost is thresholds that
   are arbitrary at the edges, so they're configurable and reported in every answer.
2. **Judgment weights.** There are no past renovation outcomes to fit weights against. Data-driven
   weights (entropy, PCA) were rejected because they have no causal meaning.
3. **Demand-side only.** No public source covers terminal capacity. Blind spot: an airport that just
   expanded can still score high.
4. **Bulk files vs JSON-only APIs.** Taking the files buys delay causes and route distances. The cost
   is a heavy first download (~400 MB, ~5 minutes) and a fragile form download.
5. **Shipping a data snapshot.** It spares reviewers that download and makes demos independent of
   slow or failing BTS servers, for 3.6 MB in the repo. The cost: a clone starts with data fetched on
   2026-10-08 rather than the latest. Answers cite that date, and the weekly check pulls newer BTS
   months when online.
6. **NAS cause as the congestion signal.** It separates capacity from airline and severe-weather
   problems. But airlines assign the cause, and NAS includes non-extreme weather such as SFO fog.
7. **Live data kept out of the score,** so rankings are reproducible.

## 7. Limitations

- Capacity is inferred, not measured. Validate any shortlist against terminal design capacity.
- The unmet-demand flag is less reliable for small airports. It also flags Punta Gorda,
  St. Pete–Clearwater and Orlando Sanford in Florida (thunderstorms count as NAS) and Lincoln, NE
  (which may inherit delays from the hubs its flights come from).
- On-time data covers only larger US carriers' domestic flights; some small airports have none.
- The segment form download is the most fragile source. If it fails, the long-haul answer falls back
  to per-airport averages, labeled as an approximation.
- Conversation memory lasts for the browser session only.

## 8. Architecture

```
src/config.py                 weights, thresholds, reference set, regions
src/data/sources/             one module per public source
src/data/{cache,repository}   dated cache + bundled snapshot; per-airport metrics with provenance
data/snapshot/                committed, gzipped API responses (read-only at runtime)
src/scoring/{metrics,score}   KPIs, gate, percentiles, ranking, unmet-demand flag
src/agent/{tools,agent}       tool definitions, system prompt
src/llm/                      provider interface; Claude and OpenAI-compatible tool loops
src/voice/stt.py              local speech-to-text (faster-whisper)
app/                          Streamlit UI; chat bar component (Python + JS + CSS)
tests/                        scoring, tool loops (fake clients), cache, audio, chat bar (Node)
```

## 9. Abbreviations

| Term | Meaning |
|---|---|
| BTS | Bureau of Transportation Statistics, the US agency that publishes the flight data |
| FAA | Federal Aviation Administration, which runs US air traffic control |
| NAS | National Airspace System. As a delay cause: airport operations, heavy traffic, air traffic control, non-extreme weather |
| T-100 | BTS data in which airlines report every route flown monthly |
| TTM | Trailing twelve months |
| Load factor | Share of seats filled |
| PFC | Passenger Facility Charge, up to $4.50 per boarding passenger, used to fund airport projects |
| IATA code | Three-letter airport code (SFO = San Francisco, ANC = Anchorage, PWM = Portland, Maine) |
