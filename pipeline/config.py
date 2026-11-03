"""Paths and defaults. Anything environment-specific is read from env vars, never hard-coded."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("ML_DATA_ROOT", REPO_ROOT / "data"))
ARTIFACT_ROOT = Path(os.environ.get("ML_ARTIFACT_ROOT", REPO_ROOT / "artifacts"))
# The Lambda image is built from ./lambda, so a promoted model lands here.
SERVING_MODEL_DIR = REPO_ROOT / "lambda" / "model"
SAMPLE_REQUEST = REPO_ROOT / "events" / "request.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Minimum metric values a model must reach before it may be deployed.
DEFAULT_THRESHOLDS = {"accuracy": 0.93, "roc_auc": 0.97, "malignant_recall": 0.90}
