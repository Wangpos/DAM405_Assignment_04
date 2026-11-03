"""AWS Lambda handler: breast-cancer diagnosis inference.

Works for a direct invocation and for an API Gateway proxy event (HTTP API v2 or REST v1).
Request body:
    {"features": {"mean_radius": 17.99, ...}}            one record, or
    {"instances": [{"mean_radius": 17.99, ...}, ...]}    a batch
"""
import base64
import json
import logging
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "model"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "100"))

# Loaded once per execution environment during the init phase, then reused by
# every warm invocation. This is the main cold-start cost of the function.
_load_started = time.perf_counter()
MODEL = joblib.load(MODEL_DIR / "model.joblib")
METADATA = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES = METADATA["feature_names"]
CLASS_NAMES = METADATA["class_names"]
MODEL_LOAD_MS = round((time.perf_counter() - _load_started) * 1000, 1)
_cold_start = True


def lambda_handler(event, context):
    global _cold_start
    cold_start, _cold_start = _cold_start, False
    started = time.perf_counter()

    try:
        X = _to_matrix(_extract_instances(event))
    except ValueError as exc:
        logger.warning("Bad request: %s", exc)
        return _response(400, {"error": str(exc)})

    probabilities = MODEL.predict_proba(X)
    predictions = [
        {
            "label": CLASS_NAMES[int(np.argmax(row))],
            "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, row)},
        }
        for row in probabilities
    ]
    inference_ms = round((time.perf_counter() - started) * 1000, 2)

    # One structured log line per request -> searchable in CloudWatch Logs Insights.
    logger.info(json.dumps({
        "event": "prediction",
        "model_version": METADATA["model_version"],
        "batch_size": len(predictions),
        "labels": [p["label"] for p in predictions],
        "cold_start": cold_start,
        "model_load_ms": MODEL_LOAD_MS,
        "inference_ms": inference_ms,
        "request_id": getattr(context, "aws_request_id", None),
    }))
    return _response(200, {
        "model_version": METADATA["model_version"],
        "predictions": predictions,
        "cold_start": cold_start,
        "inference_ms": inference_ms,
    })


def _extract_instances(event) -> list:
    if not isinstance(event, dict):
        raise ValueError("Event must be a JSON object")

    payload = event
    if "body" in event:  # API Gateway proxy integration
        body = event["body"]
        if not body:
            raise ValueError("Request body is empty")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        try:
            payload = json.loads(body) if isinstance(body, str) else body
        except json.JSONDecodeError as exc:
            raise ValueError(f"Body is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    if "instances" in payload:
        instances = payload["instances"]
    elif "features" in payload:
        instances = [payload["features"]]
    else:
        raise ValueError('Expected "features" (object) or "instances" (list of objects)')

    if not isinstance(instances, list) or not instances:
        raise ValueError('"instances" must be a non-empty list')
    if len(instances) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch too large: {len(instances)} > {MAX_BATCH_SIZE}")
    return instances


def _to_matrix(instances: list) -> np.ndarray:
    rows = []
    for i, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise ValueError(f"Instance {i} must be an object of feature: value")
        missing = [f for f in FEATURES if f not in instance]
        if missing:
            raise ValueError(f"Instance {i} is missing {len(missing)} feature(s), e.g. {missing[:3]}")
        try:
            row = [float(instance[f]) for f in FEATURES]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Instance {i} has a non-numeric feature value") from exc
        if not all(math.isfinite(v) for v in row):
            raise ValueError(f"Instance {i} has a NaN or infinite feature value")
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
