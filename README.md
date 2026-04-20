# Bullet — Script Intelligence Platform

An AI-powered screenplay analysis system that reads a PDF script and produces deep narrative intelligence: story summary, emotional arc, engagement scoring, improvement suggestions, and cliffhanger detection.

---

## Table of Contents

1. [What Problem This Solves](#1-what-problem-this-solves)
2. [Why RAG — The Core Architecture Decision](#2-why-rag--the-core-architecture-decision)
3. [Stage 1 — PDF to Scenes (Chunking)](#3-stage-1--pdf-to-scenes-chunking)
4. [Stage 2 — Indexing into ChromaDB (Vector Store)](#4-stage-2--indexing-into-chromadb-vector-store)
5. [Stage 3 — The Analysis Pipeline (LangGraph)](#5-stage-3--the-analysis-pipeline-langgraph)
   - [Why LangGraph](#why-langgraph)
   - [Node 1: Summary](#node-1-summary)
   - [Node 2: Emotional Arc](#node-2-emotional-arc)
   - [Node 3: Engagement Score](#node-3-engagement-score)
   - [Node 4: Improvement Suggestions](#node-4-improvement-suggestions)
   - [Node 5: Cliffhanger Detection](#node-5-cliffhanger-detection)
6. [Production Infrastructure](#6-production-infrastructure)
   - [Job Queue — Redis + Celery](#job-queue--redis--celery)
   - [ChromaDB — Per-Job Isolated Storage](#chromadb--per-job-isolated-storage)
   - [Rate Limiting](#rate-limiting)
   - [LLM Retry Logic](#llm-retry-logic)
   - [Cost Tracking](#cost-tracking)
   - [Centralised Config](#centralised-config)
7. [Model Strategy — GPT-4o vs GPT-4o-mini](#7-model-strategy--gpt-4o-vs-gpt-4o-mini)
8. [The Three Layers of Every Chunk](#8-the-three-layers-of-every-chunk)
9. [Full Data Flow Diagram](#9-full-data-flow-diagram)
10. [Tech Stack](#10-tech-stack)
11. [Setup and Running](#11-setup-and-running)

---

## 1. What Problem This Solves

A screenplay is a long, complex document — anywhere from 15,000 to 80,000 words. Analysing it with AI is not as simple as asking an LLM "what do you think of this script?" for three reasons:

**Token limits** — Even the largest LLM context windows struggle with a full feature screenplay plus detailed analysis instructions in a single prompt. And even if they fit, long prompts produce shallow, generic answers.

**Different questions need different context** — The question "does this script have a strong hook?" only needs the opening scenes. The question "is the dialogue realistic?" needs dialogue-heavy scenes. Sending the entire script to answer every question is wasteful and noisy.

**Different tasks need different thinking** — Detecting emotion scene-by-scene is a sequential task (each scene depends on the previous). Scoring engagement from four angles is a parallel task (each angle is independent). A single LLM call cannot express this structure.

This system solves all three using a **RAG pipeline orchestrated by LangGraph**.

---

## 2. Why RAG — The Core Architecture Decision

RAG stands for **Retrieval-Augmented Generation**. The core idea is:

> Instead of putting everything into one prompt, store your content in a searchable database first. Then for each specific question, retrieve only the relevant pieces and send those to the LLM.

### Without RAG (naive approach)

```
Full Script (50,000 words) + "Analyse everything" → LLM → Answer
```

Problems:
- Hits token limits on longer scripts
- LLM gets overwhelmed and produces generic output
- Expensive — paying for 50,000 tokens on every single analysis call
- Cannot give each specialist agent the context it specifically needs

### With RAG (our approach)

```
Full Script → Chunked → Stored in ChromaDB (vector + full text + summary)
                ↓
Each analysis node → "Find me scenes about X" → ChromaDB returns top matches
                ↓
LLM receives only the relevant scenes → Focused, accurate analysis
```

Benefits:
- Each agent gets exactly the context relevant to its job
- No token limit issues — summaries give full coverage in a small prompt
- Much cheaper — targeted retrieval instead of full-script repetition every time
- Better answers — less noise, more signal

---

## 3. Stage 1 — PDF to Scenes (Chunking)

### What happens

The uploaded PDF is read page by page and all text is extracted into one long string. Then the chunking begins.

### Why chunk at all

An LLM has a context window limit. Even if a script fits, sending everything produces worse results than sending focused pieces. Chunking breaks the script into **meaningful, searchable units**.

### How we chunk — Scene Detection

We scan the full text for standard screenplay scene headers using a pattern matcher:

```
INT. HOSPITAL - NIGHT
EXT. STREET - DAY
INT/EXT. CAR - MOVING
```

Every time one of these headers appears, it marks the start of a new scene. The text between two consecutive headers is one chunk.

**Why scene boundaries and not pages or fixed word counts?**

A scene is the natural narrative unit of a screenplay. It represents one location, one time period, one set of events. It has internal coherence — meaning it makes sense as a standalone unit. Pages are arbitrary (a scene can span 3 pages or 3 lines). Fixed word counts cut mid-scene, destroying meaning. ChromaDB's semantic search works best when each chunk carries a complete, coherent idea. Scene-based chunking gives us that.

### Fallback chunking

If a script has no standard scene headers (e.g. a prose treatment or non-standard format), the system falls back to splitting by character count (~3,600 characters per chunk, roughly one screenplay page). Each chunk is called "Section 1", "Section 2" etc.

---

## 4. Stage 2 — Indexing into ChromaDB (Vector Store)

This is where RAG actually happens. Every scene chunk gets stored in ChromaDB with **three layers of information**.

### Layer 1 — Vector Embedding

OpenAI's `text-embedding-3-small` model reads the full text of each scene and converts it into a vector — a list of 1,536 numbers that mathematically represents the *meaning* of that scene.

**Why vectors?**

Because keyword search is brittle. If you search for "conflict" you only find scenes that literally use the word conflict. But a scene where a character silently refuses to leave a room is full of conflict — no keyword matches it.

Vectors capture meaning, not words. Two scenes about different topics but similar emotional weight will have similar vectors. This is what makes semantic search work — "find me scenes about emotional tension" returns relevant scenes even if they use completely different words.

All vectors are stored with cosine similarity as the distance metric, meaning the search finds scenes whose meaning is closest to the query.

### Layer 2 — Full Text

The actual raw screenplay text of the scene (up to 2,000 characters) is stored as the document. This is retrieved when a node needs to read the real dialogue, action lines, or specific craft details.

### Layer 3 — Summary (The Key Innovation)

During indexing, GPT-4o-mini reads each scene and writes a 2-3 sentence summary covering: what happens, who is involved, and the emotional tone.

All scene summaries are generated **in parallel** using a thread pool — so a 30-scene script generates all 30 summaries simultaneously instead of one by one.

**Why store summaries?**

Different analysis tasks need different levels of detail:

| Task | What it needs | What we send |
|---|---|---|
| Overall story summary | Coverage of entire script | All summaries concatenated |
| Emotion per scene | Gist of what happened and the mood | That scene's summary |
| Dialogue analysis | Actual words spoken | Full text of relevant scenes |
| Hook analysis | Scenes that function as the opening | Full text of retrieved scenes |

Without summaries, you have two bad options: send everything (expensive, hits token limits) or truncate (loses information). Summaries give you a third option — **intelligent compression that preserves meaning**.

**The cost argument:**
A raw scene text is ~1,500 characters. Its summary is ~150 characters. When the emotional arc node processes 30 scenes, using summaries instead of raw text reduces token usage by roughly 10x — for the same quality of emotion detection.

### What ChromaDB stores per chunk

```
ID:         scene_3
Document:   [Full text of scene, up to 2000 chars]
Embedding:  [1536-dimensional vector]
Metadata: {
    scene_index:  3,
    scene_title:  "INT. HOSPITAL - NIGHT",
    char_count:   1840,
    summary:      "Dr. Ahmed delivers the fatal diagnosis to the family.
                   The mother breaks down while the father stands frozen.
                   The tone is one of devastating grief and disbelief."
}
```

---

## 5. Stage 3 — The Analysis Pipeline (LangGraph)

### Why LangGraph

LangGraph is a framework for building stateful, multi-step AI pipelines as a graph of nodes.

**The core value:** Each node has one specific job. It reads from a shared state, does its work, and writes its result back to the state. The next node sees everything — all previous results plus its own inputs.

**Why not just chain LLM calls manually?**

You could write simple sequential calls, but this breaks down when:
- Some nodes should run in parallel (engagement's 4 agents)
- State accumulation is needed (per-scene emotions building up)
- You need error handling per node without crashing the whole pipeline
- You want to add or reorder nodes without rewriting everything

LangGraph makes the pipeline's structure explicit, manageable, and extensible.

### The Shared State

Every node reads from and writes to a single `ScriptState` object that flows through the entire graph:

```
ScriptState {
    title            — script name
    full_text        — raw extracted text
    scenes           — list of all scene dicts
    collection       — ChromaDB collection (all nodes access this)
    scene_emotions   — accumulates as emotional arc node runs
    summary          — written by Node 1
    emotional_arc    — written by Node 2
    engagement_score — written by Node 3
    suggestions      — written by Node 4
    cliffhanger      — written by Node 5
    errors           — any node can append errors without crashing others
}
```

The `collection` being in state is critical — every node can query ChromaDB directly without needing it passed as a separate argument.

---

### Node 1: Summary

**Job:** Produce genre, overall story summary, central conflict, and main characters.

**What it does:**
1. Calls `get_all_summaries()` — fetches every chunk's summary from ChromaDB, ordered by scene index
2. Concatenates all summaries into one context string
3. Sends to **GPT-4o** with the summary prompt

**Why summaries here:**
A 40-scene script's summaries might total 3,000 words. The same script's full text is 40,000+ words. Summaries give GPT-4o complete narrative coverage — beginning, middle, and end — without hitting token limits or truncating anything. GPT-4o reads the entire story arc and produces an accurate, well-informed summary.

**Why GPT-4o and not GPT-4o-mini:**
The summary is the first thing users see. It needs to be accurate, well-written, and insightful. This is a synthesis task that benefits from stronger reasoning.

---

### Node 2: Emotional Arc

**Job:** Track how emotion evolves across every scene in narrative order, then synthesise the overall arc pattern.

**What it does:**

**Phase 1 — Per-scene analysis (sequential):**
1. Fetches all chunk summaries from ChromaDB in scene order
2. For each scene, sends its summary + the previous scene's emotion to GPT-4o-mini
3. GPT-4o-mini returns: dominant emotion, intensity (0–1), and emotional shift description
4. The result becomes the "previous emotion" context for the next scene

**Phase 2 — Arc synthesis:**
All per-scene emotions are passed to GPT-4o which identifies the overall arc pattern (e.g. "tragic descent", "redemptive rise"), names the emotional peak scene, and writes a description of the journey.

**Why sequential and not parallel:**
Emotion is contextual. Scene 15 feeling like "relief" only makes sense because Scene 14 was "terror". If you analyse all scenes simultaneously, each one loses that temporal context. The arc becomes a disconnected list of isolated emotions instead of a flowing narrative journey.

LangGraph's stateful accumulation is designed exactly for this — each iteration passes context forward.

**Why summaries and not full text:**
Emotion is detectable from what happened in a scene, not from its exact dialogue. "A doctor tells a family their loved one died, the mother collapses" is enough to identify grief. Sending 1,500 characters of raw dialogue for the same answer is ~10x more expensive for no quality gain.

---

### Node 3: Engagement Score

**Job:** Score the script's engagement across four dimensions and produce a final weighted score.

**What it does:**

Four specialist agents run **in parallel** using a thread pool. Each agent:
1. Queries ChromaDB semantically for its specific area
2. Receives the most relevant scene chunks (full text, not summaries)
3. Scores its factor 0–10 with reasoning and evidence

| Agent | ChromaDB Query | What it evaluates |
|---|---|---|
| Hook | "opening hook grabbing attention inciting incident introduction setup" | Does the script grab you immediately? |
| Conflict | "conflict struggle character motivation obstacle" | Are stakes and obstacles clear and compelling? |
| Tension | "tension suspense danger stakes rising action" | Does tension build effectively? |
| Cliffhanger | "ending resolution climax final conclusion payoff" | Does the ending leave the audience wanting more? |

After all four agents finish, a **synthesis agent** (GPT-4o) combines them into a final weighted engagement score with a verdict.

**Why parallel:**
These four agents have zero dependency on each other. Running them simultaneously instead of sequentially cuts the total time to roughly one quarter.

**Why semantic search and not fixed scene positions:**
The hook of a script is not always Scene 1 and Scene 2. It could be a cold open followed by a title card, then the real inciting incident in Scene 4. Semantic search finds the scenes that *function as* the hook, wherever they appear. Fixing to "first 2 scenes" blindly misses this.

**Why full text for engagement (not summaries):**
Engagement scoring requires judgment on specific craft elements — the quality of a line, the pacing of an action sequence, the specificity of a conflict beat. Summaries abstract away exactly these details. Full text is needed here.

---

### Node 4: Improvement Suggestions

**Job:** Identify the most impactful improvements the writer can make, from four specialist perspectives.

**What it does:**

Four critic agents run simultaneously, each querying ChromaDB for their domain:

| Critic | ChromaDB Query | What it critiques |
|---|---|---|
| Dialogue | "dialogue conversation character speech" | Is dialogue natural, distinctive, purposeful? |
| Pacing | "pacing rhythm scene transitions momentum" | Does the story move at the right speed? |
| Conflict | "conflict stakes obstacles character goals" | Are conflicts clearly defined and escalating? |
| Emotional | "emotion feeling character arc transformation" | Do characters change meaningfully? |

Each critic returns raw suggestions. A **synthesis agent** (GPT-4o) then:
- Deduplicates overlapping suggestions from multiple critics
- Ranks by impact: high / medium / low priority
- Identifies the single most critical fix
- Returns a clean, ordered list of improvements

**Why four critics and not one:**
One LLM asked to critique everything tends to produce generic feedback. A critic told "you are a dialogue specialist — only evaluate dialogue" focuses its attention and produces specific, actionable notes. The synthesis step integrates all specialist views into a coherent whole. This is the same reason film productions have separate editors, sound designers, and directors of photography.

---

### Node 5: Cliffhanger Detection

**Job:** Identify the single most suspenseful moment in the script and explain its narrative mechanism.

**What it does:**
1. Semantically queries ChromaDB for "cliffhanger suspense revelation twist danger unresolved"
2. Always includes the final 2 scenes regardless of their semantic score
3. Deduplicates and sorts by scene order
4. Sends to GPT-4o which identifies the key moment, names the tension type, quotes it, and scores effectiveness 0–10

**Why always include the final scenes:**
Semantic search finds scenes that *sound like* a cliffhanger. But sometimes a screenplay's actual ending is understated — a quiet, ambiguous final image. Its vector score for "suspense" might be low, but it is the cliffhanger by definition. Guaranteeing the final scenes are always included prevents this miss.

---

## 6. Production Infrastructure

The analysis pipeline is powerful but expensive and slow — a single script can take 60–120 seconds and make 40+ OpenAI calls. Without production infrastructure, multiple users hitting the system simultaneously would cause rate limit crashes, users accidentally triggering duplicate analyses, and no visibility into costs. This section covers every layer added to make the system production-ready.

---

### Job Queue — Redis + Celery

**The problem without a queue:**

```
User 1 uploads → 40 OpenAI calls start immediately
User 2 uploads → 40 more OpenAI calls start immediately
User 3 uploads → OpenAI rate limit hit → everything crashes
```

**How the queue works:**

```
User uploads script
    │
    ▼
app.py extracts raw text from PDF
    │
    ▼
analyse_script.delay(title, raw_text)   ← submits job to Redis, returns job_id instantly
    │
    ▼
Celery worker (separate process) picks job off queue
    │
    ├── Runs ingestion + ChromaDB indexing
    ├── Runs LangGraph pipeline
    └── Stores result in Redis under job_id
    │
    ▼
app.py polls AsyncResult(job_id) every 3 seconds
    │
    ▼
When job.state == SUCCESS → fetch result → render UI
```

**Redis** acts as two things simultaneously:
- **Message broker** — holds the queue of pending jobs ("here are jobs waiting to be processed")
- **Result backend** — stores completed job results for 1 hour so the UI can retrieve them

**Celery** is the worker layer. It pulls jobs off the Redis queue and processes them. Key configuration decisions:

| Setting | Value | Why |
|---|---|---|
| `task_acks_late=True` | Enabled | Only marks job as done after it completes. If the worker crashes mid-analysis, the job goes back to the queue instead of being lost |
| `task_reject_on_worker_lost=True` | Enabled | If the worker process dies, the job is requeued automatically |
| `result_expires=3600` | 1 hour | Results stay in Redis for 1 hour — enough time to view them, doesn't clog memory forever |
| `worker_pool=gevent` | gevent | Windows-compatible pool. Default prefork pool fails on Windows due to multiprocessing restrictions |
| `worker_concurrency=4` | 4 | 4 analyses can run in parallel — configurable via `CELERY_MAX_WORKERS` in `.env` |
| `task_soft_time_limit=600` | 10 min | Raises an exception if a job runs longer than 10 minutes, preventing stuck jobs |
| `task_time_limit=660` | 11 min | Hard kill 60 seconds after soft limit — guaranteed termination |

**Files:**
- [`celery_app.py`](celery_app.py) — Celery instance and all configuration
- [`tasks.py`](tasks.py) — The `analyse_script` task definition

---

### ChromaDB — Per-Job Isolated Storage

**The problem without isolation:**

Without job-scoped storage, multiple users writing to ChromaDB simultaneously would overwrite each other's collections, mix up scene data between users, and produce wrong analysis results.

**How it works:**

Every Celery job gets its own isolated ChromaDB directory:

```
./chroma_db/
    ├── a3f92b1c-.../ ← Job 1's ChromaDB (User 1)
    ├── 7d84e209-.../ ← Job 2's ChromaDB (User 2)
    └── 1bc30f44-.../ ← Job 3's ChromaDB (User 3)
```

Each job passes `persist_path=f"./chroma_db/{job_id}"` to `store_chunks_in_chromadb()`. ChromaDB creates a `PersistentClient` at that path — completely isolated from all other jobs.

**Cleanup:** The `tasks.py` `finally` block always deletes the job's ChromaDB directory after the pipeline finishes, even if the job failed. This prevents disk accumulation.

```
finally:
    shutil.rmtree(chroma_path, ignore_errors=True)
```

In development (no Celery), `persist_path=None` uses `EphemeralClient()` in-memory — no disk required.

---

### Rate Limiting

**The problem without rate limiting:**

One user (or a script) can submit unlimited analyses, burning through your OpenAI budget in minutes.

**How it works:**

Redis stores a counter per session:

```
Key:   rate_limit:{session_id}
Value: 3                          ← number of analyses this hour
TTL:   3600 seconds               ← auto-expires after 1 hour
```

On every submit:
1. Increment the counter
2. If counter > limit → block the request and show an error
3. Counter auto-resets after the TTL window expires

**Fail open:** If Redis is unavailable, the rate limiter returns `allowed=True`. A Redis outage should not take down the whole app — it degrades gracefully to no rate limiting rather than blocking all users.

**Configuration (`.env`):**
```
RATE_LIMIT_MAX=5       # max analyses per session per window
RATE_LIMIT_WINDOW=3600 # window duration in seconds (1 hour)
```

**File:** [`rate_limiter.py`](rate_limiter.py)

---

### LLM Retry Logic

**The problem without retries:**

OpenAI's API occasionally returns rate limit errors (`429`) or connection drops, especially under parallel load. Without retries, one transient error fails the entire analysis.

**How it works:**

Every `call_llm()` is wrapped with `tenacity` — a retry library:

```
call_llm() fails with RateLimitError
    │
    ▼
Wait 2 seconds (exponential backoff starts)
    │
    ▼
Retry #1 → fails again
    │
    ▼
Wait 4 seconds
    │
    ▼
Retry #2 → succeeds → continues normally
```

**Retry configuration:**

| Setting | Value | Why |
|---|---|---|
| Retries on | `RateLimitError`, `APIConnectionError`, `APITimeoutError` | Only retry transient errors, not auth failures or bad requests |
| Max attempts | 3 | Beyond 3 retries the API is likely seriously degraded |
| Min wait | 2 seconds | Gives the API time to recover from a brief spike |
| Max wait | 30 seconds | Exponential backoff caps here — prevents 5-minute waits |
| `reraise=True` | Enabled | After all retries exhausted, raises the original exception so the Celery task can handle it |

**File:** [`llm.py`](llm.py)

---

### Cost Tracking

**The problem without tracking:**

You have no idea what each analysis costs. A bug that causes infinite retries or sends enormous prompts could cost hundreds of dollars before you notice.

**How it works:**

Every `call_llm()` call reads the `usage` object from OpenAI's response (input tokens + output tokens) and adds the cost to a thread-safe session accumulator.

Cost rates stored in `config.py`:
```
GPT-4o:       $2.50 / 1M input tokens,  $10.00 / 1M output tokens
GPT-4o-mini:  $0.15 / 1M input tokens,  $0.60 / 1M output tokens
text-embedding-3-small: $0.02 / 1M tokens
```

After each analysis completes, the UI shows:
```
✓ 28 scenes analysed · Cost: $0.0842 · 47 LLM calls · 38,291 total tokens
```

The accumulator is thread-safe (uses a `threading.Lock`) so parallel LLM calls from the engagement and suggestions nodes don't corrupt the counter.

**File:** [`llm.py`](llm.py) — `get_session_cost()`, `reset_session_cost()`, `_track_cost()`

---

### Centralised Config

All configurable values live in one file — [`config.py`](config.py). No more hunting through multiple files to change a model name or rate limit.

```python
# config.py controls:
FAST_MODEL           = "gpt-4o-mini"
SMART_MODEL          = "gpt-4o"
REDIS_URL            = "redis://localhost:6379/0"
CELERY_MAX_WORKERS   = 4
RATE_LIMIT_MAX       = 5
RATE_LIMIT_WINDOW    = 3600
LLM_MAX_RETRIES      = 3
CHROMA_BASE_DIR      = "./chroma_db"
COST_PER_1M          = { ... }  # pricing table
```

Everything reads from environment variables first, with sensible defaults. To change any behaviour in production, update `.env` — no code changes needed.

---

## 7. Model Strategy — GPT-4o vs GPT-4o-mini

| Task | Model | Why |
|---|---|---|
| Per-chunk summarisation (ingestion) | GPT-4o-mini | Runs 30+ times in parallel, summaries are simple |
| Per-scene emotion detection | GPT-4o-mini | One per scene, emotion from summary is a narrow task |
| Specialist agents (hook, conflict, tension, critics) | GPT-4o-mini | Each is a focused, well-scoped task — mini handles these well |
| Summary synthesis | GPT-4o | Needs strong narrative reasoning across the whole story |
| Emotional arc synthesis | GPT-4o | Pattern recognition across 30+ data points needs strong model |
| Engagement synthesis | GPT-4o | Weighing four expert opinions requires judgment |
| Suggestions synthesis | GPT-4o | Deduplication and ranking needs nuanced reasoning |
| Cliffhanger detection | GPT-4o | Narrative interpretation, not just classification |

The rule: **cheap model for narrow tasks, smart model for synthesis and judgment**.

---

## 8. The Three Layers of Every Chunk

```
┌─────────────────────────────────────────────────────┐
│                   CHROMADB CHUNK                    │
├─────────────────────────────────────────────────────┤
│  VECTOR EMBEDDING                                   │
│  text-embedding-3-small → 1536 numbers              │
│  Used for: semantic search ("find scenes like X")   │
├─────────────────────────────────────────────────────┤
│  FULL TEXT                                          │
│  Raw screenplay text, up to 2000 chars              │
│  Used for: engagement agents, suggestion critics    │
│  (tasks needing actual craft-level detail)          │
├─────────────────────────────────────────────────────┤
│  SUMMARY (in metadata)                              │
│  2-3 sentences from GPT-4o-mini                     │
│  Used for: overall summary node, emotional arc      │
│  (tasks needing full coverage cheaply)              │
└─────────────────────────────────────────────────────┘
```

---

## 9. Full Data Flow Diagram

```
PDF Upload
    │
    ▼
Extract Full Text (PyPDF2)
    │
    ▼
Detect Scene Boundaries (regex on INT./EXT. headers)
    │
    ├── Scenes found    → Split at each header
    └── No headers      → Split by ~3600 char chunks
    │
    ▼
For each scene chunk [ALL IN PARALLEL]:
    ├── Generate vector embedding  (text-embedding-3-small)
    └── Generate summary           (GPT-4o-mini)
    │
    ▼
Store in ChromaDB:
    { id, full_text, vector, metadata: { summary, scene_title, scene_index } }
    │
    ▼
LangGraph Pipeline — ScriptState flows through all 5 nodes
    │
    ├─── Node 1: SUMMARY
    │       ├── get_all_summaries() from ChromaDB (scene order)
    │       ├── Concatenate all summaries into one context
    │       └── GPT-4o → genre, summary, conflict, characters
    │
    ├─── Node 2: EMOTIONAL ARC
    │       ├── get_all_summaries() from ChromaDB (scene order)
    │       ├── [SEQUENTIAL] For each scene:
    │       │       summary + prev_emotion → GPT-4o-mini
    │       │       → dominant_emotion, intensity, shift
    │       └── All scene emotions → GPT-4o → arc pattern, peak scene
    │
    ├─── Node 3: ENGAGEMENT SCORE [4 agents PARALLEL]
    │       ├── Hook agent:        query "opening hook inciting incident"
    │       ├── Conflict agent:    query "conflict struggle motivation"
    │       ├── Tension agent:     query "tension suspense danger stakes"
    │       └── Cliffhanger agent: query "ending climax resolution"
    │               └── 4 factor scores → GPT-4o → overall_score, verdict
    │
    ├─── Node 4: SUGGESTIONS [4 critics PARALLEL]
    │       ├── Dialogue critic:  query "dialogue conversation speech"
    │       ├── Pacing critic:    query "pacing rhythm transitions"
    │       ├── Conflict critic:  query "conflict stakes obstacles"
    │       └── Emotional critic: query "emotion character arc"
    │               └── Raw suggestions → GPT-4o → ranked, deduplicated list
    │
    └─── Node 5: CLIFFHANGER
            ├── query "cliffhanger suspense revelation twist"
            ├── Always include final 2 scenes
            └── GPT-4o → moment, tension_type, mechanism, effectiveness_score
    │
    ▼
Streamlit UI renders all results
    ├── Summary card
    ├── Emotional arc with per-scene emotion chips
    ├── Engagement score + factor breakdown bars
    ├── Improvement suggestions with priority levels
    └── Cliffhanger quote + analysis
```

---

## 10. Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM (smart) | GPT-4o | Best reasoning for synthesis tasks |
| LLM (fast) | GPT-4o-mini | Cheap and fast for narrow per-chunk tasks |
| Embeddings | text-embedding-3-small | Best cost/quality ratio for semantic search |
| Vector Store | ChromaDB (PersistentClient) | Per-job isolated storage on disk, cosine similarity |
| Pipeline Orchestration | LangGraph | Stateful multi-node graph with parallel + sequential control |
| Job Queue Broker | Redis | Holds pending jobs and stores completed results |
| Async Task Worker | Celery + gevent | Processes analyses in background, Windows-compatible pool |
| Rate Limiting | Redis counter | Per-session request limiting with auto-expiry TTL |
| LLM Resilience | tenacity | Exponential backoff retry on rate limits and connection drops |
| Cost Tracking | Custom accumulator | Token-level cost tracking per session, thread-safe |
| PDF Parsing | PyPDF2 | Extract raw text from uploaded screenplay PDFs |
| Output Validation | Pydantic | Type-safe structured outputs from LLM responses |
| Frontend | Streamlit | Web UI with job submission, polling, and result rendering |
| Config Management | python-dotenv + config.py | All settings in one place, env-var driven |

---

## 11. Setup and Running

### Prerequisites

- Python 3.10+
- OpenAI API key
- Redis (via Docker or native install)

### Install dependencies

```bash
pip install -r requirements.txt
pip install gevent   # Windows-compatible Celery pool
```

### Configure

Edit `.env` in the `script_analysis/` directory:

```env
OPENAI_API_KEY=sk-your-key-here

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_MAX_WORKERS=4
CELERY_TASK_TIMEOUT=600

# Rate limiting
RATE_LIMIT_MAX=5
RATE_LIMIT_WINDOW=3600

# LLM retries
LLM_MAX_RETRIES=3
LLM_RETRY_MIN_WAIT=2
LLM_RETRY_MAX_WAIT=30

# ChromaDB storage path
CHROMA_BASE_DIR=./chroma_db
```

### Start Redis

```bash
# Using Docker (recommended)
docker run -d -p 6379:6379 --name redis redis

# Verify it's running
redis-cli ping   # should return: PONG
```

### Start Celery worker (Terminal 1)

```bash
cd script_analysis
celery -A celery_app.celery_app worker --loglevel=info -Q analysis
```

You should see this on startup — confirming the task is registered:
```
[tasks]
  . tasks.analyse_script
```

### Start Streamlit (Terminal 2)

```bash
cd script_analysis
streamlit run app.py
```

### Use

1. Upload a PDF screenplay
2. Enter the script title
3. Click **Analyse Script** — job is submitted to the queue instantly
4. UI polls every 3 seconds showing queue/progress status
5. When complete, results render with scene count, cost, and token usage
6. Optionally export full analysis as JSON

---

## Key Design Principles

**Right tool for the right job** — mini for narrow tasks, GPT-4o for synthesis. Summaries for coverage, full text for depth, semantic search for relevance.

**Pay once, use many times** — summaries are generated once at ingestion and reused by multiple pipeline nodes. The vector index is built once and queried five times.

**Parallel where independent, sequential where dependent** — emotional arc must be sequential (each scene needs the previous). Every other multi-agent step runs in parallel.

**Specialist over generalist** — four focused critic agents produce better suggestions than one agent asked to critique everything. Each specialist goes deep on its domain.

**Never send everything to everyone** — each node gets exactly the context it needs, retrieved semantically. No node receives the full raw script directly.
