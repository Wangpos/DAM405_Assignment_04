"""Pipeline stages shared by the Airflow DAG and the local runner.

Each stage reads its inputs from files, writes its outputs atomically into a
directory keyed by run id, and returns the output path(s). Re-running a stage
for the same run id overwrites the same files, so every stage is idempotent.
Only small strings (paths, metrics) are passed between stages, never data.
"""
import json
import logging
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import QualityGateError, config, schema
from pipeline.storage import (
    atomic_copy,
    atomic_dump_joblib,
    atomic_write_csv,
    atomic_write_json,
    run_dir,
)

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract(run_id: str, data_root=None, inject_bad_data: bool = False) -> str:
    """Pull the source dataset and land it unchanged as raw.csv."""
    df = load_breast_cancer(as_frame=True).frame
    if inject_bad_data:
        df = _inject_bad_data(df)

    out = run_dir(data_root or config.DATA_ROOT, run_id) / "raw.csv"
    atomic_write_csv(df, out)
    log.info("Extracted %d rows x %d columns -> %s", *df.shape, out)
    return str(out)


def _inject_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate a broken upstream feed so the quality gate can be demonstrated."""
    bad = df.copy()
    bad.loc[bad.index[:5], "mean radius"] = np.nan
    bad.loc[bad.index[5:10], "mean area"] = -1.0
    bad.loc[bad.index[10], "target"] = 2
    log.warning("Injected bad records into the extract (demo mode)")
    return bad


def transform(raw_path, run_id: str, data_root=None) -> dict:
    """Rename to API feature names, de-duplicate and make a stratified train/test split."""
    df = pd.read_csv(raw_path).rename(columns=schema.to_feature_name)
    before = len(df)
    df = df.drop_duplicates()

    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[schema.TARGET],
        random_state=config.RANDOM_STATE,
    )
    out_dir = run_dir(data_root or config.DATA_ROOT, run_id)
    paths = {"train": out_dir / "train.csv", "test": out_dir / "test.csv"}
    atomic_write_csv(train_df, paths["train"])
    atomic_write_csv(test_df, paths["test"])
    log.info(
        "Dropped %d duplicates; train=%d test=%d", before - len(df), len(train_df), len(test_df)
    )
    return {name: str(path) for name, path in paths.items()}


def _xy(df: pd.DataFrame):
    return df[schema.FEATURES].to_numpy(dtype=float), df[schema.TARGET].to_numpy(dtype=int)


def train(train_path, run_id: str, artifact_root=None) -> str:
    """Fit scaler + logistic regression as ONE sklearn Pipeline.

    Keeping the scaler inside the model artifact means the Lambda receives raw
    measurements and applies exactly the training-time scaling (no train/serve skew).
    """
    X, y = _xy(pd.read_csv(train_path))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
        ]
    )
    model.fit(X, y)

    model_dir = run_dir(artifact_root or config.ARTIFACT_ROOT, run_id)
    atomic_dump_joblib(model, model_dir / "model.joblib")
    atomic_write_json(
        {
            "model_version": model_dir.name,
            "algorithm": "StandardScaler + LogisticRegression",
            "trained_at": _now(),
            "train_rows": int(len(y)),
            "feature_names": schema.FEATURES,
            "class_names": schema.CLASS_NAMES,
            "sklearn_version": sklearn.__version__,
        },
        model_dir / "model_card.json",
    )
    log.info("Trained model on %d rows -> %s", len(y), model_dir)
    return str(model_dir)


def evaluate(model_dir, test_path, thresholds: dict | None = None) -> dict:
    """Score the hold-out set and enforce the model-quality gate."""
    thresholds = thresholds or config.DEFAULT_THRESHOLDS
    model = joblib.load(Path(model_dir) / "model.joblib")
    X, y = _xy(pd.read_csv(test_path))
    proba_benign = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, proba_benign)),
        # Missing a malignant tumour is the costly error, so track its recall explicitly.
        "malignant_recall": float(recall_score(y, pred, pos_label=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "test_rows": int(len(y)),
        "thresholds": thresholds,
        "evaluated_at": _now(),
    }
    atomic_write_json(metrics, Path(model_dir) / "metrics.json")
    log.info("Metrics: %s", {k: metrics[k] for k in thresholds})

    failures = [
        f"{name}={metrics[name]:.4f} < {minimum}"
        for name, minimum in thresholds.items()
        if metrics[name] < minimum
    ]
    if failures:
        raise QualityGateError("Model below deployment threshold: " + "; ".join(failures))
    return metrics


def promote(model_dir, serving_dir=None) -> str:
    """Copy an approved model + metadata into the Lambda build context."""
    src = Path(model_dir)
    dst = Path(serving_dir or config.SERVING_MODEL_DIR)
    card = json.loads((src / "model_card.json").read_text())
    metrics = json.loads((src / "metrics.json").read_text())

    atomic_copy(src / "model.joblib", dst / "model.joblib")
    atomic_write_json({**card, "metrics": metrics, "promoted_at": _now()}, dst / "metadata.json")
    log.info("Promoted model %s -> %s", card["model_version"], dst)
    return str(dst)


def sam_deploy(repo_root=None) -> None:
    """Rebuild the Lambda image and deploy the SAM stack.

    AWS credentials come from the worker's environment / AWS profile, never from code.
    """
    cwd = repo_root or config.REPO_ROOT
    for cmd in (
        ["sam", "build"],
        ["sam", "deploy", "--no-confirm-changeset", "--no-fail-on-empty-changeset"],
    ):
        log.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=cwd, check=True)


def smoke_test(api_url: str, request_path=None) -> dict:
    """POST a known request to the live endpoint and check that predictions come back."""
    body = Path(request_path or config.SAMPLE_REQUEST).read_bytes()
    request = urllib.request.Request(
        api_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    if not payload.get("predictions"):
        raise RuntimeError(f"Endpoint returned no predictions: {payload}")
    log.info("Smoke test OK: %s", payload)
    return payload
