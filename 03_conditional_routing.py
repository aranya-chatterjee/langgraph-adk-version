"""
LangGraph Topic 3: Conditional Edges (Explicit Routing)

ADK equivalent: sub_agents (Topic 4) -- the orchestrator LLM implicitly
decided which specialist to call, based on instruction text. Here, YOU
write a router function that reads state and returns the next node's
name as a plain string. No LLM judgment call involved in the routing
decision itself.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TriageState(TypedDict):
    repo: str
    issue_number: int
    label: str
    request_type: str  # "classify" or "summarize" -- set by the caller


def classify_node(state: TriageState) -> dict:
    return {"label": "BUG"}


def summarize_node(state: TriageState) -> dict:
    return {"label": "N/A (summarized, not classified)"}


def route_request(state: TriageState) -> str:
    """The router function: reads state, returns the NEXT NODE'S NAME.
    This is a plain Python function -- fully deterministic, testable in
    isolation without ever calling an LLM."""
    if state["request_type"] == "summarize":
        return "summarize"
    return "classify"


builder = StateGraph(TriageState)
builder.add_node("classify", classify_node)
builder.add_node("summarize", summarize_node)

# add_conditional_edges(source, router_fn, {router_output: destination_node})
builder.add_conditional_edges(
    START,
    route_request,
    {
        "classify": "classify",
        "summarize": "summarize",
    },
)
builder.add_edge("classify", END)
builder.add_edge("summarize", END)

graph = builder.compile()


if __name__ == "__main__":
    r1 = graph.invoke({"repo": "x/y", "issue_number": 1, "label": "", "request_type": "classify"})
    print("classify path ->", r1["label"])

    r2 = graph.invoke({"repo": "x/y", "issue_number": 1, "label": "", "request_type": "summarize"})
    print("summarize path ->", r2["label"])

    # Unlike ADK's sub_agents, this routing is 100% reproducible: same
    # request_type ALWAYS produces the same path. No LLM "mood" involved.
