# LangGraph GitHub Triage Agent

A GitHub issue classifier built on [LangGraph](https://langchain-ai.github.io/langgraph/),
running on Groq. Given a repo and an issue number, it fetches the issue, classifies it as
`BUG` / `FEATURE` / `DOCS` / `QUESTION`, verifies that classification with a second LLM
pass, and escalates to a human when the verifier stays unconvinced.

This is the LangGraph counterpart to the Google ADK implementation in
[`adk-github-agent`](https://github.com/aranya-chatterjee/adk-github-agent). The numbered
files map ADK concepts onto their LangGraph equivalents one topic at a time; `agent_graph.py`
is the real, unified agent.

## The graph

```
fetch ──(error)──> END
  │
  └─(ok)─> classify ──> verify ──(CONFIRMED)──> finalize ──> END
                          │
                     (UNCERTAIN)
                          │
                          v
                     retry_bump ──(retry_count < 2)──> classify
                          │
                     (>= 2 retries)
                          v
                      escalate ──(interrupt: waits for a human)──> finalize ──> END
```

Design notes worth knowing:

- **Fetch failures become state, not exceptions.** `fetch_node` catches `RequestException`
  and writes `{"error": ...}`, which a conditional edge routes straight to `END` — the same
  pattern the ADK codebase uses in `tools.py`.
- **The retry counter lives in a node, not a router.** Routing functions must be pure, so
  `retry_bump_node` does the incrementing and `check_retry` only reads.
- **Escalation uses `interrupt()`**, which requires a checkpointer at compile time. The
  graph compiled at import in `agent_graph.py` has none; `main.py` wires one in.
- **`thread_id` is derived per issue** (`{repo}-issue-{number}`). Hardcoding it would let
  one paused escalation silently overwrite another's state.

## Files

| File | What it is |
|---|---|
| `agent_graph.py` | The real agent — the unified graph, single source of truth |
| `main.py` | FastAPI deployment: `/classify`, `/resume`, `/health` |
| `01_graph_fundamentals.py` | State, nodes, edges; a minimal fetch → classify graph |
| `02_state_and_reducers.py` | Reducers — appending to state instead of overwriting it |
| `03_conditional_routing.py` | Deterministic branching, unlike ADK's LLM-driven `sub_agents` |
| `04_loops_and_termination.py` | Cycles plus `recursion_limit` as a safety cap |
| `05_structured_verification.py` | `.with_structured_output()`, LangChain's `output_schema` |
| `06_human_in_the_loop.py` | `interrupt()` / `Command(resume=...)` across one `thread_id` |
| `07_persistence_and_deployment.py` | Checkpointers + a FastAPI wrapper |
| `Dockerfile` | Serves `main.py` on port 8081 |

## Setup

```bash
python -m venv venv
venv/Scripts/activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Create a `.env` in this directory:

```
GROQ_API_KEY="your_groq_key"
GITHUB_TOKEN="your_github_token"
```

`GITHUB_TOKEN` is optional — it only raises the GitHub API rate limit for `fetch_node`.
`GROQ_API_KEY` is required. `.env` is gitignored; never commit it.

### Model selection

Both `agent_graph.py` and `05_structured_verification.py` read `GROQ_MODEL`, defaulting to
`openai/gpt-oss-120b`:

```bash
GROQ_MODEL=openai/gpt-oss-120b
```

Groq retires model ids periodically. `llama-3.3-70b-versatile` — which this codebase
originally hardcoded — is already gone and returns `model_not_found`. Check
`client.models.list()` for what your key can actually reach before changing this.

Whatever you pick must **support tool calling** if it is used for verification:
`.with_structured_output()` is implemented as a tool call, and models such as
`groq/compound-mini` reject it with `` `tool calling` is not supported with this model ``.
`compound-mini` is fine for the free-text classification in `01`.

## Running

Directly:

```bash
python agent_graph.py
```

As an API:

```bash
uvicorn main:app --host 127.0.0.1 --port 8081
```

Then classify an issue:

```bash
curl -X POST http://127.0.0.1:8081/classify \
  -H "Content-Type: application/json" \
  -d '{"repo": "vercel/next.js", "issue_number": 97472}'
```

```json
{"status": "done", "label": "BUG", "error": "", "question": ""}
```

If the verifier stays uncertain through both retries, the response comes back as
`{"status": "paused_for_human", "question": "..."}` instead. Resume it with the same repo
and issue number:

```bash
curl -X POST http://127.0.0.1:8081/resume \
  -H "Content-Type: application/json" \
  -d '{"repo": "vercel/next.js", "issue_number": 97472, "human_label": "FEATURE"}'
```

Interactive docs are at `http://127.0.0.1:8081/docs`.

## Verified behaviour

Run against `vercel/next.js#97472` ("Soft navigation leaves the previous route's metadata
mounted in head"), the agent returns:

| Field | Value |
|---|---|
| `label` | `BUG` |
| `verdict` | `CONFIRMED` |
| `retry_count` | `0` |
| `final_label` | `BUG` |

Consistent across four invocation paths: direct `graph.invoke`, `curl` against `/classify`,
`01_graph_fundamentals.py`, and the Swagger UI.

## Known limitations

- **State is lost on restart.** Both `main.py` and `07` run on `InMemorySaver`, so a paused
  escalation does not survive a process restart. The `PostgresSaver` swap is present but
  commented out and has not been exercised.
- **The topic scripts (`01`–`07`) do not call `load_dotenv()`.** Run bare, they fail on a
  missing `GROQ_API_KEY`. Export the variables into your shell, or add the call yourself.
- **Escalation is unproven on real data.** Every real run so far confirmed on the first
  pass, so the retry loop and the `interrupt()` path have only been exercised through `06`,
  which uses synthetic state. `/resume` has never been called against a genuinely paused run.
- **One LLM call per node, no caching.** A single `/classify` is two Groq calls, more with
  retries. Groq's free tier caps around 12,000 tokens/minute; concurrent runs will 429.

## Docker

```bash
docker build -t langgraph-triage .
docker run -p 8081:8081 -e GROQ_API_KEY=... -e GITHUB_TOKEN=... langgraph-triage
```

Untested — the image has not been built.
