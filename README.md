# DAM405 Assignment 4: Serverless Model Deployment & Pipeline Orchestration

This project trains a breast-cancer diagnosis classifier (scikit-learn, Breast Cancer Wisconsin dataset). It then **serves the model from AWS Lambda** behind an API Gateway HTTP API, defined as code with **AWS SAM**. An **Apache Airflow DAG** orchestrates the full pipeline: extract → validate (Great Expectations gate) → transform → train → evaluate (model gate) → deploy → smoke test.

```
.
├── lambda/                  # Serverless inference
│   ├── app.py               #   Lambda handler
│   ├── Dockerfile           #   container image (Python 3.12, arm64)
│   ├── requirements.txt
│   └── model/               #   promoted model.joblib + metadata.json (generated)
├── template.yaml            # IaC: Lambda + HTTP API + log group (AWS SAM)
├── samconfig.toml           # SAM build/deploy settings
├── dags/breast_cancer_pipeline.py   # Airflow DAG
├── pipeline/                # Stage logic used by the DAG (testable without Airflow)
│   ├── steps.py             #   extract / transform / train / evaluate / promote / deploy
│   ├── quality.py           #   Great Expectations data-quality gate
│   ├── schema.py            #   data contract (feature names)
│   ├── storage.py           #   atomic file writes
│   └── config.py
├── events/                  # sample Lambda / API payloads
├── scripts/                 # local runner + Airflow env setup
├── tests/                   # pytest
└── REPORT.md                # written report
```

## Prerequisites

| Tool | Version used | Needed for |
|---|---|---|
| Python | 3.12 | pipeline, Airflow, tests |
| Docker Desktop | running | `sam build` / `sam local` |
| AWS SAM CLI | ≥ 1.120 | build, local invoke, deploy |
| AWS CLI + account | v2 | *only* for the real cloud deploy |

> No AWS account? Every step except 6 works with **AWS SAM local** (Lambda + API Gateway emulated in Docker).

## 1. Create the environment

```bash
cd DAM405_Assignment4
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Install Airflow (with constraints) and the project requirements

```bash
AIRFLOW_VERSION=2.10.5
PYTHON_VERSION=3.12
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install -r requirements.txt
```

## 3. Run the tests

```bash
pytest -v
```
📸 **Screenshot:** the pytest summary showing all tests passed.

## 4. Train a first model (without Airflow)

```bash
python scripts/run_pipeline_local.py
```
This writes `lambda/model/model.joblib` and `metadata.json`, which the Lambda image needs.
📸 **Screenshot:** the printed metrics (accuracy / roc_auc / malignant_recall).

Optional: show the quality gate rejecting bad data:
```bash
python scripts/run_pipeline_local.py --run-id bad --inject-bad-data
```
📸 **Screenshot:** the `Pipeline stopped by quality gate` error.

## 5. Serverless inference with AWS SAM (local emulation)

```bash
sam validate --lint
sam build
```
📸 **Screenshot:** `Build Succeeded`.

**a) Invoke the function directly with an event**
```bash
sam local invoke InferenceFunction --event events/direct_event.json
sam local invoke InferenceFunction --event events/api_event.json
```
📸 **Screenshot:** the response `{"statusCode": 200, ... "label": "malignant" ...}` plus the `REPORT ... Init Duration` line, which shows the cold start.

**b) Through the emulated API Gateway**
```bash
sam local start-api            # terminal 1
curl -s -X POST http://127.0.0.1:3000/predict \
     -H "Content-Type: application/json" \
     -d @events/request.json | python -m json.tool     # terminal 2
```
📸 **Screenshot:** the curl JSON response (2 predictions) and the `start-api` log.

## 6. Deploy to real AWS (if you have an account)

```bash
aws configure            # or: export AWS_PROFILE=your-profile   (credentials are never stored in the repo)
sam deploy --guided      # first time; accepts defaults from samconfig.toml
```
Copy the `PredictUrl` output, then:
```bash
curl -s -X POST "<PredictUrl>" -H "Content-Type: application/json" -d @events/request.json
sam logs -n InferenceFunction --stack-name dam405-bc-inference --tail
```
📸 **Screenshots:** the CloudFormation outputs, the curl response, the Lambda console *Monitor* tab, and the CloudWatch log line `"event": "prediction"`.

Tear down afterwards to avoid charges: `sam delete --stack-name dam405-bc-inference`

## 7. Run the Airflow pipeline

```bash
source scripts/airflow_env.sh       # sets AIRFLOW_HOME, DAGS_FOLDER, PYTHONPATH
airflow standalone                  # UI at http://localhost:8080
```
The admin password is printed in the terminal and saved to `.airflow/standalone_admin_password.txt`.

Optional Airflow Variables (UI → Admin → Variables, or CLI in a 2nd terminal after `source scripts/airflow_env.sh`):

| Variable | Default | Purpose |
|---|---|---|
| `ml_min_accuracy` | 0.93 | model gate |
| `ml_min_roc_auc` | 0.97 | model gate |
| `ml_min_malignant_recall` | 0.90 | model gate |
| `ml_deploy_with_sam` | false | `true` → `deploy_model` also runs `sam build && sam deploy` |
| `ml_inference_api_url` | *(empty)* | endpoint for `smoke_test` (e.g. `http://127.0.0.1:3000/predict` while `sam local start-api` runs, or the AWS `PredictUrl`) |

```bash
airflow variables set ml_inference_api_url http://127.0.0.1:3000/predict
```

**Successful run:** in the UI, unpause `breast_cancer_serverless_pipeline` and click ▶ *Trigger DAG*. You can also use the CLI:
```bash
airflow dags trigger breast_cancer_serverless_pipeline
```
📸 **Screenshots:** the Graph view with all tasks green, the `evaluate_model` log showing the metrics, and the `smoke_test` log.

**Quality-gate run:** trigger with config `{"inject_bad_data": true}`:
```bash
airflow dags trigger breast_cancer_serverless_pipeline --conf '{"inject_bad_data": true}'
```
📸 **Screenshots:** the Graph view with `validate_raw_data` red and downstream tasks `upstream_failed`, plus that task's log listing the failed expectations.

The GX report for each run is saved in `data/<run_id>/quality_report.json`, and model metrics in `artifacts/<run_id>/metrics.json`.

## Design notes
* **Idempotent:** outputs are keyed by the Airflow `run_id`, so a retry overwrites the same files.
* **Atomic:** one responsibility per task. Files are written to a temp file and then renamed.
* **No hard-coded secrets:** thresholds and URLs come from Airflow Variables, and AWS credentials come from the AWS profile.
* **No train/serve skew:** the scaler is saved inside the model pipeline, and both environments pin `numpy==1.26.4` and `scikit-learn==1.5.2`.

## Troubleshooting
* `sam build` on an Intel Mac or Windows PC: change `Architectures` in `template.yaml` to `x86_64`.
* Lambda error `No such file ... model.joblib`: run step 4 (or the DAG) before `sam build`.
* Airflow cannot import `pipeline`: you did not `source scripts/airflow_env.sh` in that terminal.

## References
* Dataset: Wolberg, W. et al. *Breast Cancer Wisconsin (Diagnostic)*, UCI ML Repository, accessed via `sklearn.datasets.load_breast_cancer`.
* AWS SAM, AWS Lambda container images, Apache Airflow TaskFlow API, Great Expectations GX Core 1.x documentation.
# DAM405_Assignment_04
