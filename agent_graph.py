"""
Unified LangGraph GitHub Triage Agent

Combines Topics 1-6 into ONE coherent graph (Topic 7 provides the
persistence/deployment layer around this). This is the LangGraph
equivalent of the ADK codebase's issue_classifier/agent.py -- the single
source of truth, not a teaching demo.

Flow:
    fetch -> (error? -> END) -> classify -> verify
        -> CONFIRMED -> finalize -> END
        -> UNCERTAIN -> retry_bump -> (retry_count < 2? -> classify | -> escalate)
    escalate -> finalize -> END   (interrupt()-based HITL, needs a checkpointer)
"""

import os
from typing import Literal, Optional, TypedDict

import requests
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt


# ---------------------------------------------------------------------------
# State schema -- every field this graph reads or writes lives here
# ---------------------------------------------------------------------------
class TriageState(TypedDict):
    repo: str
    issue_number: int
    title: str
    body: str
    label: str
    verdict: str
    reason: str
    retry_count: int
    human_label: str
    final_label: str
    error: Optional[str]


class ClassificationVerdict(BaseModel):
    verdict: Literal["CONFIRMED", "UNCERTAIN"]
    reason: str


# Single source of truth for the model, mirroring the ADK codebase's
# issue_classifier/config.py. Overridable so a retired model id (Groq drops
# these periodically -- llama-3.3-70b-versatile is already gone) doesn't
# require a code change.
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# Node: fetch -- tool-level failures are CAUGHT and turned into a state
# signal, same pattern as ADK's {"error": ...} dict (tools.py, Topic 2).
# ---------------------------------------------------------------------------
def fetch_node(state: TriageState) -> dict:
    try:
        url = f"https://api.github.com/repos/{state['repo']}/issues/{state['issue_number']}"
        token = os.environ.get("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title", ""),
            "body": (data.get("body") or "")[:3000],
            "error": None,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"fetch failed: {e}"}


def check_fetch(state: TriageState) -> str:
    return "error" if state.get("error") else "ok"


# ---------------------------------------------------------------------------
# Node: classify -- real LLM call, same model used throughout (Groq)
# ---------------------------------------------------------------------------
def classify_node(state: TriageState) -> dict:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=GROQ_MODEL_NAME)
    prompt = f"""Classify this GitHub issue into exactly one label:
BUG, FEATURE, DOCS, or QUESTION. Return ONLY the label.

Title: {state['title']}
Body: {state['body']}"""
    response = llm.invoke(prompt)
    return {"label": response.content.strip().upper()}


# ---------------------------------------------------------------------------
# Node: verify -- structured output, the real "verify" step (Topic 5/6)
# ---------------------------------------------------------------------------
def verify_node(state: TriageState) -> dict:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=GROQ_MODEL_NAME)
    structured_llm = llm.with_structured_output(ClassificationVerdict)
    prompt = f"""Review this classification.
Title: {state['title']}
Body: {state['body']}
Assigned label: {state['label']}

Return CONFIRMED with a brief reason if well-supported, or UNCERTAIN
with a reason if ambiguous, mixed-concern, or debatable."""
    verdict: ClassificationVerdict = structured_llm.invoke(prompt)
    return {"verdict": verdict.verdict, "reason": verdict.reason}


def check_verdict(state: TriageState) -> str:
    return "confirmed" if state["verdict"] == "CONFIRMED" else "uncertain"


# ---------------------------------------------------------------------------
# Node: retry_bump -- bounded retry counter (mirrors ADK's
# increment_retry_count, Topic 7). Routing decisions must be pure, so the
# counter increment happens in a NODE, not inside the router function.
# ---------------------------------------------------------------------------
def retry_bump_node(state: TriageState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def check_retry(state: TriageState) -> str:
    return "retry" if state["retry_count"] < 2 else "escalate"


# ---------------------------------------------------------------------------
# Node: escalate -- interrupt()-based HITL (Topic 6). Requires a checkpointer
# at compile time (wired in main.py, not here).
# ---------------------------------------------------------------------------
def escalate_node(state: TriageState) -> dict:
    human_answer = interrupt(
        f"Classification uncertain after {state['retry_count']} retries.\n"
        f"Issue: {state['title']}\n"
        f"Proposed label: {state['label']} ({state['reason']})\n"
        f"What should the correct label be?"
    )
    return {"human_label": human_answer}


# ---------------------------------------------------------------------------
# Node: finalize -- human_label (if escalated) wins over the model's label
# ---------------------------------------------------------------------------
def finalize_node(state: TriageState) -> dict:
    return {"final_label": state.get("human_label") or state["label"]}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    builder = StateGraph(TriageState)

    builder.add_node("fetch", fetch_node)
    builder.add_node("classify", classify_node)
    builder.add_node("verify", verify_node)
    builder.add_node("retry_bump", retry_bump_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "fetch")
    builder.add_conditional_edges("fetch", check_fetch, {"error": END, "ok": "classify"})
    builder.add_edge("classify", "verify")
    builder.add_conditional_edges(
        "verify", check_verdict, {"confirmed": "finalize", "uncertain": "retry_bump"}
    )
    builder.add_conditional_edges(
        "retry_bump", check_retry, {"retry": "classify", "escalate": "escalate"}
    )
    builder.add_edge("escalate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# Default: no checkpointer (interrupt() won't work without one -- main.py
# wires a real checkpointer for the deployed API).
graph = build_graph()


if __name__ == "__main__":
    result = graph.invoke({
        "repo": "vercel/next.js",
        "issue_number": 12345,  # replace with a real open issue number
        "title": "", "body": "", "label": "", "verdict": "", "reason": "",
        "retry_count": 0, "human_label": "", "final_label": "", "error": None,
    })
    if result.get("error"):
        print(f"Fetch failed: {result['error']}")
    else:
        print(f"Final label: {result['final_label']}")
