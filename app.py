import operator
import os
from contextlib import nullcontext
from typing import Literal

import uvicorn
from databricks_langchain import ChatDatabricks
from fastapi import FastAPI, Request
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import mlflow
from mlflow.tracing import set_tracing_context_from_http_request_headers

# Capture graph execution, LLM calls, and tools in the UC experiment configured
# through MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID.
mlflow.langchain.autolog()


@tool
def calculate(
    operation: Literal["add", "subtract", "multiply", "divide"], a: float, b: float
) -> float:
    """Calculate a single arithmetic operation on two numbers."""
    return {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }[operation](a, b)


agent = create_agent(
    ChatDatabricks(endpoint=os.environ["LLM_ENDPOINT"]),
    tools=[calculate],
    system_prompt=(
        "You are a calculator agent. Always call the calculate tool to solve arithmetic; "
        "never answer without using it. Give a concise answer with the appropriate units "
        "and a brief explanation."
    ),
)
app = FastAPI()


class InvokeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/invoke")
def invoke(body: InvokeRequest, request: Request):
    headers = dict(request.headers)

    # Apps proxies rewrite traceparent. Restore the client's original parent;
    # other hosting environments can use the standard header directly.
    if original_parent := headers.get("x-mlflow-traceparent"):
        headers["traceparent"] = original_parent

    # Import context before creating spans so autologged agent spans join the
    # client's trace. Requests without a parent start their own trace.
    parent_context = (
        set_tracing_context_from_http_request_headers(headers)
        if "traceparent" in headers
        else nullcontext()
    )
    with parent_context, mlflow.start_span("remote-handler", span_type="AGENT") as span:
        span.set_inputs({"question": body.question})
        result = agent.invoke(
            {"messages": [{"role": "user", "content": body.question}]},
            config={"recursion_limit": 10},
        )
        answer = result["messages"][-1].text
        span.set_outputs({"answer": answer})
        return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("DATABRICKS_APP_PORT", "8000")))
