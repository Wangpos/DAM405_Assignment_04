"""Runs the pipeline end-to-end into a temp dir, then invokes the Lambda handler on its model."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from pipeline import QualityGateError, steps

REPO = Path(__file__).resolve().parents[1]
EVENTS = REPO / "events"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    raw = steps.extract("test", data_root=root / "data")
    split = steps.transform(raw, "test", data_root=root / "data")
    model_dir = steps.train(split["train"], "test", artifact_root=root / "artifacts")
    metrics = steps.evaluate(model_dir, split["test"])
    serving_dir = steps.promote(model_dir, serving_dir=root / "serving")
    return {"split": split, "model_dir": model_dir, "metrics": metrics, "serving": serving_dir}


@pytest.fixture(scope="module")
def app(trained):
    os.environ["MODEL_DIR"] = trained["serving"]
    try:
        spec = importlib.util.spec_from_file_location("lambda_app", REPO / "lambda" / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        del os.environ["MODEL_DIR"]
    return module


def _load(name):
    return json.loads((EVENTS / name).read_text())


def test_model_meets_thresholds(trained):
    assert trained["metrics"]["accuracy"] >= 0.93


def test_transform_is_idempotent(trained):
    train_path = Path(trained["split"]["train"])
    first = train_path.read_text()
    steps.transform(train_path.with_name("raw.csv"), "test", data_root=train_path.parents[1])
    assert train_path.read_text() == first


def test_evaluate_gate_blocks_weak_model(trained):
    with pytest.raises(QualityGateError):
        steps.evaluate(trained["model_dir"], trained["split"]["test"], {"accuracy": 1.01})


def test_direct_invocation(app):
    response = app.lambda_handler(_load("direct_event.json"), None)
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["predictions"][0]["label"] == "malignant"


def test_api_gateway_event(app):
    response = app.lambda_handler(_load("api_event.json"), None)
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["predictions"][0]["label"] == "benign"


def test_batch_request(app):
    response = app.lambda_handler(_load("request.json"), None)
    labels = [p["label"] for p in json.loads(response["body"])["predictions"]]
    assert labels == ["malignant", "benign"]


@pytest.mark.parametrize(
    "event",
    [
        {"body": "not json"},
        {"body": json.dumps({"something": 1})},
        {"instances": []},
    ],
)
def test_bad_requests_return_400(app, event):
    assert app.lambda_handler(event, None)["statusCode"] == 400


def test_missing_feature_returns_400(app):
    event = copy.deepcopy(_load("direct_event.json"))
    del event["features"]["worst_area"]
    response = app.lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert "missing" in json.loads(response["body"])["error"]
