"""
LangGraph Topic 4: Loops (edges pointing backward) + recursion_limit

ADK equivalent: LoopAgent + check_batch_complete + escalate=True (Topic 5).
In LangGraph, a "loop" is just an edge that points back to an earlier
node. The exit condition is a conditional edge that routes to END instead
of back to the loop body. recursion_limit is LangGraph's equivalent of
ADK's max_iterations -- a hard safety cap.
"""

from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END


class BatchState(TypedDict):
    issue_numbers: list[int]
    classified: Annotated[list[dict], operator.add]
    current_index: int


def classify_one(state: BatchState) -> dict:
    idx = state["current_index"]
    issue_number = state["issue_numbers"][idx]
    return {
        "classified": [{"issue_number": issue_number, "label": "BUG"}],
        "current_index": idx + 1,
    }


def check_batch_complete(state: BatchState) -> str:
    """The equivalent of ADK's check_batch_complete tool -- but here it's
    a router function that returns where to go next, instead of setting
    tool_context.actions.escalate = True."""
    if state["current_index"] >= len(state["issue_numbers"]):
        return "done"
    return "continue"


builder = StateGraph(BatchState)
builder.add_node("classify_one", classify_one)

builder.add_edge(START, "classify_one")

# This is the loop: "continue" routes BACK to classify_one, forming a cycle.
builder.add_conditional_edges(
    "classify_one",
    check_batch_complete,
    {
        "continue": "classify_one",  # <- loop back
        "done": END,
    },
)

# recursion_limit is LangGraph's max_iterations equivalent -- a hard cap
# independent of your exit-condition logic, same safety-net principle
# we discussed for ADK's LoopAgent(max_iterations=20).
graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"issue_numbers": [101, 102, 103, 104], "classified": [], "current_index": 0},
        config={"recursion_limit": 10},  # safety cap, like ADK's max_iterations
    )
    print(f"Classified {len(result['classified'])} issues:")
    for c in result["classified"]:
        print(f"  #{c['issue_number']}: {c['label']}")
