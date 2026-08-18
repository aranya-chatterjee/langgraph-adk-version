"""
LangGraph Topic 6: Human-in-the-Loop via interrupt()

ADK equivalent: escalate_to_human + conversational pause (Topic 7).
LangGraph's interrupt() PAUSES the graph mid-execution, saves state to
the checkpointer, and waits for the caller to resume with a value --
this requires a checkpointer (see topic7) because the graph's execution
is literally suspended, not just "asking a question in the chat".
"""

from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class TriageState(TypedDict):
    title: str
    body: str
    label: str
    verdict: str
    human_label: str


def classify_node(state: TriageState) -> dict:
    return {"label": "BUG", "verdict": "UNCERTAIN"}  # simulate an uncertain case


def check_verdict(state: TriageState) -> str:
    if state["verdict"] == "UNCERTAIN":
        return "escalate"
    return "done"


def escalate_node(state: TriageState) -> dict:
    """This is the equivalent of ADK's escalate_to_human tool -- but
    instead of just recording a pending-review entry and letting the
    conversation continue, interrupt() PAUSES graph execution entirely
    until someone calls the graph again with Command(resume=...)."""
    human_answer = interrupt(
        f"Classification uncertain for: {state['title']}\n"
        f"Proposed label: {state['label']}\n"
        f"What should the correct label be?"
    )
    return {"human_label": human_answer}


builder = StateGraph(TriageState)
builder.add_node("classify", classify_node)
builder.add_node("escalate", escalate_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify", check_verdict, {"escalate": "escalate", "done": END}
)
builder.add_edge("escalate", END)

# interrupt() REQUIRES a checkpointer -- without it, LangGraph has nowhere
# to save the paused state, and resuming would be impossible.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "issue-42"}}

    # First call: hits interrupt(), pauses, returns control to caller
    result = graph.invoke(
        {"title": "Ambiguous issue", "body": "...", "label": "", "verdict": "", "human_label": ""},
        config=config,
    )
    print("Paused. Interrupt payload:", result.get("__interrupt__"))

    # Second call: resume with a human-provided value, using the SAME thread_id
    final = graph.invoke(Command(resume="FEATURE"), config=config)
    print("Resumed. Final state:", final)
