import json
import os

import requests
from databricks.sdk import WorkspaceClient

import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, ToolCallCorrectness
from mlflow.tracing import get_tracing_context_headers_for_http_request

workspace = WorkspaceClient()
APP_URL = os.environ["APP_URL"].rstrip("/")

# This is passed directly to evaluate: input keys match predict_fn's arguments.
EXAMPLES = [
    {
        "inputs": {
            "question": (
                "A shop has 17 notebooks and receives 25 more. How many notebooks does it have now?"
            )
        },
        "expectations": {"expected_response": "The shop now has 42 notebooks."},
    },
    {
        "inputs": {
            "question": "There are 91 concert tickets and 38 are sold. How many tickets remain?"
        },
        "expectations": {"expected_response": "There are 53 tickets remaining."},
    },
    {
        "inputs": {
            "question": (
                "A delivery contains 12 boxes with 7 mugs in each box. How many mugs are delivered?"
            )
        },
        "expectations": {"expected_response": "The delivery contains 84 mugs."},
    },
    {
        "inputs": {
            "question": (
                "A meal costs $48 and the total tip is $12. "
                "Three people split the combined total equally. How much does each person pay?"
            )
        },
        "expectations": {
            "expected_response": "Each person pays $20, including their share of the tip."
        },
    },
    {
        "inputs": {"question": "My account balance is -$9. I deposit $14. What is my new balance?"},
        "expectations": {"expected_response": "The new account balance is $5."},
    },
]


@mlflow.trace(name="predict")
def predict_fn(question):
    # Carry this prediction's context to the remote agent. MLflow's preliminary
    # prediction call can run without an active trace, so these headers may be empty.
    headers = get_tracing_context_headers_for_http_request()
    if traceparent := headers.get("traceparent"):
        # Preserve the caller's parent across the Databricks Apps proxy.
        headers["X-MLflow-Traceparent"] = traceparent
    response = requests.post(
        f"{APP_URL}/api/invoke",
        json={"question": question},
        headers={
            "Authorization": f"Bearer {workspace.config.oauth_token().access_token}",
            **headers,
        },
        timeout=(10, 120),
        allow_redirects=False,
    )
    response.raise_for_status()
    return response.json()["answer"]


if __name__ == "__main__":
    # MLflow reads its tracking, experiment, warehouse, and judge settings from
    # the environment. The tool-use judge reads the combined client/server trace.
    result = mlflow.genai.evaluate(
        data=EXAMPLES,
        predict_fn=predict_fn,
        scorers=[Correctness(), RelevanceToQuery(), ToolCallCorrectness()],
    )
    print(json.dumps(result.metrics, indent=2))
