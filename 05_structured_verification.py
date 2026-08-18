"""
LangGraph Topic 5: Structured Output Verification

ADK equivalent: output_schema on the verifier_agent (Topic 6). In
LangGraph there's no special "output_schema" agent parameter -- you
simply have an LLM call return a Pydantic-validated object directly,
inside a plain node function. Same semantic-failure-catching purpose,
implemented with plain Python + Pydantic instead of a framework feature.
"""

import os
from typing import Literal, TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END


class ClassificationVerdict(BaseModel):
    verdict: Literal["CONFIRMED", "UNCERTAIN"]
    reason: str


class TriageState(TypedDict):
    title: str
    body: str
    label: str
    verdict: str
    reason: str


def classify_node(state: TriageState) -> dict:
    return {"label": "BUG"}


def verify_node(state: TriageState) -> dict:
    """Calls an LLM with structured output (Pydantic model) instead of
    free text. `.with_structured_output()` is LangChain's equivalent of
    ADK's output_schema -- it forces the model's response to validate
    against ClassificationVerdict."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
    structured_llm = llm.with_structured_output(ClassificationVerdict)

    prompt = f"""Review this classification. Title: {state['title']}
Body: {state['body']}
Assigned label: {state['label']}

Return CONFIRMED with a brief reason if well-supported, or UNCERTAIN
with a reason if ambiguous or debatable."""

    verdict: ClassificationVerdict = structured_llm.invoke(prompt)
    return {"verdict": verdict.verdict, "reason": verdict.reason}


builder = StateGraph(TriageState)
builder.add_node("classify", classify_node)
builder.add_node("verify", verify_node)

builder.add_edge(START, "classify")
builder.add_edge("classify", "verify")
builder.add_edge("verify", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "title": "App crashes on login",
        "body": "Getting a 500 error on POST /auth/login.",
        "label": "",
        "verdict": "",
        "reason": "",
    })
    print(f"Label: {result['label']}")
    print(f"Verdict: {result['verdict']} -- {result['reason']}")
