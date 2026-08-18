"""
LangGraph Topic 7: Checkpointer (persistence) + FastAPI deployment

ADK equivalent: InMemorySessionService's state-loss problem (Topic 3) +
FastAPI/Docker deployment (Topic 10). InMemorySaver loses everything on
restart, same as ADK's InMemorySessionService. PostgresSaver is
LangGraph's answer to that -- durable, survives container restarts,
directly solving the idempotency/state-loss risk we discussed.
"""

from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

# --- For local dev (loses state on restart, like ADK's InMemorySessionService) ---
from langgraph.checkpoint.memory import InMemorySaver

# --- For production (durable -- pip install langgraph-checkpoint-postgres) ---
# from langgraph.checkpoint.postgres import PostgresSaver
# checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
# checkpointer.setup()  # creates tables on first run


class TriageState(TypedDict):
    repo: str
    issue_number: int
    label: str


def classify_node(state: TriageState) -> dict:
    return {"label": "BUG"}


builder = StateGraph(TriageState)
builder.add_node("classify", classify_node)
builder.add_edge(START, "classify")
builder.add_edge("classify", END)

checkpointer = InMemorySaver()   # swap for PostgresSaver in production
graph = builder.compile(checkpointer=checkpointer)


# --- FastAPI wrapper, same shape as the ADK version's main.py ---
app = FastAPI(title="LangGraph GitHub Triage Agent")


class ClassifyRequest(BaseModel):
    repo: str
    issue_number: int
    thread_id: str  # groups related calls into one persisted conversation


class ClassifyResponse(BaseModel):
    label: str


@app.post("/classify", response_model=ClassifyResponse)
async def classify_issue(req: ClassifyRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke(
        {"repo": req.repo, "issue_number": req.issue_number, "label": ""},
        config=config,
    )
    return ClassifyResponse(label=result["label"])


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
