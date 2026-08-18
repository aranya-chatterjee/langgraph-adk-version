"""
LangGraph Topic 1: StateGraph, Nodes, Edges

Direct port of ADK Topic 1-2 (Agent + Tools) into LangGraph's explicit
node/edge model. Reuses the SAME fetch_github_issue logic from the ADK
codebase -- only the orchestration layer changes.

Run:
    pip install langgraph langchain-groq
    python langgraph_version/topic1_basic_graph.py
"""

import os
from typing import TypedDict

import requests
from langgraph.graph import StateGraph, START, END


# --- State schema: explicit, typed. Compare to ADK's implicit tool_context.state ---
class TriageState(TypedDict):
    repo: str
    issue_number: int
    title: str
    body: str
    label: str


# --- Reused directly from issue_classifier/tools.py logic ---
def _fetch_issue(repo: str, issue_number: int) -> dict:
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {"title": data.get("title", ""), "body": (data.get("body") or "")[:3000]}


# --- Node 1: fetch (equivalent to ADK's fetch_github_issue tool call) ---
def fetch_node(state: TriageState) -> dict:
    """Fetches the issue. Returns a PARTIAL state update -- LangGraph merges
    it into the running state automatically."""
    issue = _fetch_issue(state["repo"], state["issue_number"])
    return {"title": issue["title"], "body": issue["body"]}


# --- Node 2: classify (equivalent to ADK's classifier_agent reasoning) ---
def classify_node(state: TriageState) -> dict:
    """Classifies using an LLM call. Explicit -- no framework magic deciding
    when/whether to call the model."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="groq/compound-mini")
    prompt = f"""Classify this GitHub issue into exactly one label:
BUG, FEATURE, DOCS, or QUESTION. Return ONLY the label.

Title: {state['title']}
Body: {state['body']}"""
    response = llm.invoke(prompt)
    return {"label": response.content.strip().upper()}


# --- Build the graph ---
builder = StateGraph(TriageState)
builder.add_node("fetch", fetch_node)
builder.add_node("classify", classify_node)

builder.add_edge(START, "fetch")
builder.add_edge("fetch", "classify")
builder.add_edge("classify", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "repo": "vercel/next.js",
        "issue_number": 97472,   # replace with a real open issue number
        "title": "",
        "body": "",
        "label": "",
    })
    print(f"Label: {result['label']}")
