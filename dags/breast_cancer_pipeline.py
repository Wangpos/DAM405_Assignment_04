"""
### Breast-cancer classifier: data → model → serverless endpoint

`extract → validate_raw_data (GX gate) → transform → train → evaluate_model (model gate) → deploy_model → smoke_test`

* **Idempotent:** every task writes to a folder keyed by the DAG `run_id`; a retry overwrites the same files.
* **Atomic:** each task does one job and writes files via temp-file + rename.
* **No secrets in code:** thresholds and endpoints come from Airflow Variables; AWS credentials
  come from the worker's AWS profile / environment.
* Trigger with `{"inject_bad_data": true}` to watch the data-quality gate stop the run.
"""
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.models.param import Param

# Pipeline modules (sklearn, Great Expectations) are imported inside tasks so the
# scheduler can parse this file quickly. Variables are read via Jinja at run time,
# not at parse time.


def _var(name: str, default: str) -> str:
    return "{{ var.value.get('%s', '%s') }}" % (name, default)


@dag(
    dag_id="breast_cancer_serverless_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=1)},
    params={
        "inject_bad_data": Param(
            False, type="boolean", description="Corrupt the extract to demo the quality gate"
        )
    },
    tags=["dam405", "mlops", "serverless"],
    doc_md=__doc__,
)
def breast_cancer_serverless_pipeline():
    @task
    def extract(run_id=None, params=None) -> str:
        from pipeline import steps

        return steps.extract(run_id, inject_bad_data=bool(params["inject_bad_data"]))

    @task
    def validate_raw_data(raw_path: str) -> str:
        from pipeline import QualityGateError, quality

        try:
            return quality.validate_raw(raw_path)
        except QualityGateError as exc:
            # Bad data will not fix itself on retry, so fail fast.
            raise AirflowFailException(str(exc)) from exc

    @task(multiple_outputs=True)
    def transform(raw_path: str, report_path: str, run_id=None) -> dict:
        # report_path is only taken so the gate is a hard data dependency.
        from pipeline import steps

        return steps.transform(raw_path, run_id)

    @task
    def train(train_path: str, run_id=None) -> str:
        from pipeline import steps

        return steps.train(train_path, run_id)

    @task
    def evaluate_model(
        model_dir: str, test_path: str, min_accuracy: str, min_roc_auc: str, min_recall: str
    ) -> dict:
        from pipeline import QualityGateError, steps

        thresholds = {
            "accuracy": float(min_accuracy),
            "roc_auc": float(min_roc_auc),
            "malignant_recall": float(min_recall),
        }
        try:
            return steps.evaluate(model_dir, test_path, thresholds)
        except QualityGateError as exc:
            raise AirflowFailException(str(exc)) from exc

    @task
    def deploy_model(model_dir: str, metrics: dict, deploy_with_sam: str) -> str:
        from pipeline import steps

        serving_dir = steps.promote(model_dir)
        if deploy_with_sam.lower() == "true":
            steps.sam_deploy()
        return serving_dir

    @task
    def smoke_test(serving_dir: str, api_url: str) -> dict:
        if not api_url:
            raise AirflowSkipException("Variable ml_inference_api_url not set; skipping")
        from pipeline import steps

        return steps.smoke_test(api_url)

    raw_path = extract()
    report_path = validate_raw_data(raw_path)
    split = transform(raw_path, report_path)
    model_dir = train(split["train"])
    metrics = evaluate_model(
        model_dir,
        split["test"],
        min_accuracy=_var("ml_min_accuracy", "0.93"),
        min_roc_auc=_var("ml_min_roc_auc", "0.97"),
        min_recall=_var("ml_min_malignant_recall", "0.90"),
    )
    serving_dir = deploy_model(model_dir, metrics, deploy_with_sam=_var("ml_deploy_with_sam", "false"))
    smoke_test(serving_dir, api_url=_var("ml_inference_api_url", ""))


breast_cancer_serverless_pipeline()
