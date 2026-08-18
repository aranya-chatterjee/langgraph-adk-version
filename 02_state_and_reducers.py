"""
LangGraph Topic 2: State Schema + Reducers

ADK equivalent: Session/State (Topic 3) -- tool_context.state was a mutable
dict you manually read/append/write. Here, State is a TYPED schema, and
"how updates combine" is defined via reducers instead of manual append logic.

Key concept: by default, a node's partial-update REPLACES a field's old
value. If you want a field to ACCUMULATE (like ADK's classified_issues list
that grew across turns), you annotate it with a reducer function.
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


# --- Without a reducer: each node's return REPLACES this field ---
# --- With Annotated[list, operator.add]: LangGraph APPENDS instead ---
class BatchState(TypedDict):
    repo: str
    issue_numbers: list[int]
    # Equivalent to ADK's tool_context.state["classified_issues"].append(...)
    # -- but declarative, no manual read-modify-write needed.
    classified: Annotated[list[dict], operator.add]
    current_index: int


def classify_one(state: BatchState) -> dict:
    """Classifies the issue at current_index. Returns a partial update:
    - classified: a list containing ONE new entry -- the reducer (operator.add)
      concatenates it onto the existing list instead of overwriting it.
    - current_index: a plain int -- no reducer means this REPLACES the old value.
    """
    idx = state["current_index"]
    issue_number = state["issue_numbers"][idx]

    # In a real graph this would call fetch + an LLM, as in topic1.
    label = "BUG"  # placeholder for demonstration

    return {
        "classified": [{"issue_number": issue_number, "label": label}],
        "current_index": idx + 1,
    }


builder = StateGraph(BatchState)
builder.add_node("classify_one", classify_one)
builder.add_edge(START, "classify_one")
builder.add_edge("classify_one", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "repo": "vercel/next.js",
        "issue_numbers": [101, 102, 103],
        "classified": [],
        "current_index": 0,
    })
    print(result["classified"])       # [{'issue_number': 101, 'label': 'BUG'}]
    print(result["current_index"])    # 1

    # Compare: if `classified` had no reducer, calling this node again
    # would OVERWRITE the list with just the new entry, losing prior results
    # -- exactly the bug we'd get if we forgot ADK's read-then-append pattern.
