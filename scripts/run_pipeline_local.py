"""Run the same pipeline stages as the Airflow DAG, without Airflow.

Useful to produce a first model for `sam build` before Airflow is set up.

    python scripts/run_pipeline_local.py                    # clean run
    python scripts/run_pipeline_local.py --inject-bad-data  # watch the quality gate fail
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import QualityGateError, quality, steps  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--inject-bad-data", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        raw = steps.extract(args.run_id, inject_bad_data=args.inject_bad_data)
        quality.validate_raw(raw)
        split = steps.transform(raw, args.run_id)
        model_dir = steps.train(split["train"], args.run_id)
        metrics = steps.evaluate(model_dir, split["test"])
        serving_dir = steps.promote(model_dir)
    except QualityGateError as exc:
        logging.error("Pipeline stopped by quality gate: %s", exc)
        return 1

    print(json.dumps({k: metrics[k] for k in ("accuracy", "roc_auc", "malignant_recall")}, indent=2))
    print(f"Model promoted to {serving_dir} - now run `sam build`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
