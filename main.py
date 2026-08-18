"""
FastAPI deployment for the unified LangGraph agent (agent_graph.py).

Uses a per-issue thread_id (never hardcoded -- see the thread_id
collision discussion) so concurrent escalations don't overwrite each
other's paused state.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8081

Test:
    curl -X POST http://localhost:8081/classify \
        -H "Content-Type: application/json" \
        -d '{"repo": "vercel/next.js", "issue_number": 12345}'

    # If the response has "status": "paused_for_human", resume with:
    curl -X POST http://localhost:8081/resume \
        -H "Content-Type: application/json" \
        -d '{"repo": "vercel/next.js", "issue_number": 12345, "human_label": "FEATURE"}'
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory

from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command
from agent_graph import build_graph

# --- Dev default: state lost on restart, same limitation as InMemorySessionService ---
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# --- Production: durable across restarts (pip install langgraph-checkpoint-postgres) ---
# from langgraph.checkpoint.postgres import PostgresSaver
# checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
# checkpointer.setup()

graph = build_graph(checkpointer=checkpointer)

app = FastAPI(title="LangGraph GitHub Triage Agent")


def _thread_id(repo: str, issue_number: int) -> str:
    """Never hardcode this -- must be unique per conversation or paused
    HITL state from one issue silently overwrites another's."""
    return f"{repo}-issue-{issue_number}"


class ClassifyRequest(BaseModel):
    repo: str
    issue_number: int


class ResumeRequest(BaseModel):
    repo: str
    issue_number: int
    human_label: str


class ClassifyResponse(BaseModel):
    status: str          # "done" | "paused_for_human" | "error"
    label: str = ""
    error: str = ""
    question: str = ""   # populated when status == "paused_for_human"


@app.post("/classify", response_model=ClassifyResponse)
async def classify_issue(req: ClassifyRequest):
    config = {
        "configurable": {"thread_id": _thread_id(req.repo, req.issue_number)},
        "recursion_limit": 15,  # safety cap -- see Topic 4
    }
    result = graph.invoke(
        {
            "repo": req.repo, "issue_number": req.issue_number,
            "title": "", "body": "", "label": "", "verdict": "", "reason": "",
            "retry_count": 0, "human_label": "", "final_label": "", "error": None,
        },
        config=config,
    )

    if result.get("__interrupt__"):
        question = result["__interrupt__"][0].value
        return ClassifyResponse(status="paused_for_human", question=question)
    if result.get("error"):
        return ClassifyResponse(status="error", error=result["error"])
    return ClassifyResponse(status="done", label=result["final_label"])


@app.post("/resume", response_model=ClassifyResponse)
async def resume_issue(req: ResumeRequest):
    config = {"configurable": {"thread_id": _thread_id(req.repo, req.issue_number)}}
    result = graph.invoke(Command(resume=req.human_label), config=config)
    return ClassifyResponse(status="done", label=result["final_label"])


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
