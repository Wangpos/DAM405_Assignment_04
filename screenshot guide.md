# Screenshot Guide: DAM405 Assignment 4

Take the screenshots in this order. Each one shows the command to run, what must be visible in the shot, and where it goes in `REPORT.md`.

- **Required** screenshots are the evidence the brief asks for: a successful function invocation and the DAG running.
- **Optional** screenshots strengthen the report. Take them if you have time or an AWS account.

Tips:
- Save each file under the exact **Save as** name shown below, in a `screenshots/` folder in the repo. The number prefix keeps them in order.
- Make the terminal font large enough to read and crop out unrelated windows.
- Keep the command visible together with its output in the same shot.

---

## Part A: Tests and model training

### 📸 01: Unit tests pass (Required)
**Save as:** `screenshots/01_pytest_passed.png`
```bash
pytest -v
```
**Must show:** the list of test names and the final line `N passed`.
**Report:** §7.1 Tests

### 📸 02: Local pipeline run with metrics (Required)
**Save as:** `screenshots/02_local_pipeline_metrics.png`
```bash
python scripts/run_pipeline_local.py
```
**Must show:** the log lines (Extracted, Trained, Metrics, Promoted) and the printed JSON with `accuracy`, `roc_auc` and `malignant_recall`.
**Report:** §7.2. Copy these numbers into the metrics table.

### 📸 03: Quality gate blocks bad data, local run (Optional)
**Save as:** `screenshots/03_local_quality_gate_fail.png`
```bash
python scripts/run_pipeline_local.py --run-id bad --inject-bad-data
```
**Must show:** `Pipeline stopped by quality gate: ... expectations failed: ...`
**Report:** §4 or §7.4, as supporting evidence for the gate.

### 📸 04: Model metrics file (Optional)
**Save as:** `screenshots/04_metrics_json.png`
Open `artifacts/local/metrics.json` in VS Code.
**Must show:** accuracy, roc_auc, malignant_recall and the confusion matrix.
**Report:** §7.2. Copy the confusion matrix into the table.

---

## Part B: Serverless inference (Lambda + SAM)

### 📸 05: SAM template is valid (Optional)
**Save as:** `screenshots/05_sam_validate.png`
```bash
sam validate --lint
```
**Must show:** `template.yaml is a valid SAM Template`.
**Report:** §3.3 Deployment as code

### 📸 06: SAM build succeeds (Required)
**Save as:** `screenshots/06_sam_build.png`
```bash
sam build
```
**Must show:** the end of the Docker image build and `Build Succeeded`.
**Report:** §3.3 Deployment as code

### 📸 07: Direct Lambda invocation, cold start (Required)
**Save as:** `screenshots/07_invoke_direct_cold_start.png`
```bash
sam local invoke InferenceFunction --event events/direct_event.json
```
**Must show:**
- the JSON response with `"statusCode": 200` and `"label": "malignant"`
- the line `REPORT RequestId: ... Init Duration: ... ms  Duration: ... ms`
- `"cold_start": true` in the response

**Report:** §7.3. Write the Init Duration and Duration into the cold/warm table.

### 📸 08: API Gateway event invocation (Required)
**Save as:** `screenshots/08_invoke_api_event.png`
```bash
sam local invoke InferenceFunction --event events/api_event.json
```
**Must show:** `"statusCode": 200` and `"label": "benign"`.
**Report:** §7.3. This proves the handler works with an API Gateway event.

### 📸 09: Local API running (Required)
**Save as:** `screenshots/09_sam_local_start_api.png`
Terminal 1:
```bash
sam local start-api
```
**Must show:** `Mounting InferenceFunction at http://127.0.0.1:3000/predict [POST]`.
**Report:** §3.2 Trigger

### 📸 10: curl call to the API, two predictions (Required)
**Save as:** `screenshots/10_curl_predictions.png`
Terminal 2, while 09 is still running:
```bash
curl -s -X POST http://127.0.0.1:3000/predict \
  -H "Content-Type: application/json" \
  -d @events/request.json | python -m json.tool
```
**Must show:** two predictions (`malignant`, `benign`) with their probabilities, plus `model_version` and `inference_ms`.
**Report:** §7.3. This is the main "successful invocation" evidence.

### 📸 11: Warm invocation (Required)
**Save as:** `screenshots/11_warm_invocation.png`
Run the same curl command a second time.
**Must show:** `"cold_start": false` in the response. In terminal 1, also capture the `REPORT ... Duration` line, which has no Init Duration this time. `inference_ms` only times the prediction itself, so it stays tiny in both calls. The cold-start cost shows up in the REPORT lines (Init Duration and Duration), not in `inference_ms`.
**Report:** §7.3 cold vs warm table, and §6 cold-start discussion.

### 📸 12: Bad request returns 400 (Optional)
**Save as:** `screenshots/12_bad_request_400.png`
```bash
curl -s -X POST http://127.0.0.1:3000/predict \
  -H "Content-Type: application/json" -d '{"features": {"mean_radius": 10}}'
```
**Must show:** an `"error": "Instance 0 is missing 29 feature(s) ..."` response.
**Report:** §3.1 Handler (input validation)

---

## Part C: Real AWS deploy (only if you have an AWS account)

Skip this part if you only use SAM local, and state that in the report header.

### 📸 13: sam deploy outputs
**Save as:** `screenshots/13_aws_sam_deploy_outputs.png`
```bash
sam deploy --guided
```
**Must show:** `Successfully created/updated stack` and the **Outputs** box with `PredictUrl`.

### 📸 14: curl to the real AWS endpoint
**Save as:** `screenshots/14_aws_curl_endpoint.png`
```bash
curl -s -X POST "<PredictUrl>" -H "Content-Type: application/json" -d @events/request.json
```
**Must show:** the predictions JSON from the `execute-api.ap-south-1.amazonaws.com` URL.

### 📸 15: Lambda console
**Save as:** `screenshots/15_aws_lambda_console.png`
In the AWS Console, open Lambda → `dam405-bc-inference-dev`.
**Must show:** the function overview with the API Gateway trigger, and the **Monitor** tab with invocations and duration graphs.

### 📸 16: CloudWatch logs
**Save as:** `screenshots/16_aws_cloudwatch_logs.png`
In the AWS Console, open CloudWatch → Log groups → `/aws/lambda/dam405-bc-inference-dev`.
**Must show:** the `"event": "prediction"` JSON log line and the `REPORT` line with Init Duration.

### 📸 17: CloudFormation stack resources
**Save as:** `screenshots/17_aws_cloudformation_stack.png`
**Must show:** the stack in `CREATE_COMPLETE` state, and the Resources tab listing the Lambda function, HTTP API, IAM role and log group.
**Report:** §5 service-to-role mapping

After taking these, delete the stack to avoid charges:
```bash
sam delete --stack-name dam405-bc-inference
```

---

## Part D: Airflow DAG

> `airflow standalone` crashes on macOS 26 (gunicorn workers `SIGSEGV`), so use three terminals instead.
> Run `cd ~/Desktop/DAM405_Assignment4 && source .venv/bin/activate && source scripts/airflow_env.sh` in terminals 1 and 3 first.

| Terminal | Command | Keep it open? |
|---|---|---|
| 1 (UI) | `airflow webserver --debug --port 8080` | yes |
| 2 (API) | `sam local start-api` | yes |
| 3 (runs) | `airflow dags reserialize` once, then the `airflow dags test ...` commands below | no |

Open http://localhost:8080 and log in as `admin`. The password is in `.airflow/standalone_admin_password.txt`.

```bash
# terminal 3
airflow variables set ml_inference_api_url http://127.0.0.1:3000/predict
airflow dags test breast_cancer_serverless_pipeline                                    # clean run (20-23)
airflow dags test breast_cancer_serverless_pipeline --conf '{"inject_bad_data": true}' # bad-data run (24-25)
```

### 📸 18: DAG list (Required)
**Save as:** `screenshots/18_airflow_dag_list.png`
**Must show:** `breast_cancer_serverless_pipeline` in the DAG list, unpaused, with its tags `dam405 mlops serverless`.
**Report:** §4 Airflow DAG

### 📸 19: Airflow Variables (Optional)
**Save as:** `screenshots/19_airflow_variables.png`
Open Admin → Variables.
**Must show:** the `ml_*` variables, e.g. `ml_inference_api_url`. This is evidence that the config is not hard-coded.
**Report:** §4 (best practice: no secrets in code)

### 📸 20: Successful DAG run, Graph view (Required)
**Save as:** `screenshots/20_airflow_dag_success_graph.png`
Run the clean `airflow dags test` command in terminal 3, with `sam local start-api` running so `smoke_test` can call it. Then open the run in the UI (DAG → Graph, pick the run).
**Must show:** all 7 tasks green: extract, validate_raw_data, transform, train, evaluate_model, deploy_model, smoke_test.
**Report:** §7.4. Also good to use next to the diagram in §2.

### 📸 21: validate_raw_data log, gate passed (Required)
**Save as:** `screenshots/21_airflow_validate_log_pass.png`
> The UI **Logs** tab is empty for `airflow dags test` runs (the log server crashes on macOS), so 21–23 and 25 use the saved run logs in `evidence/`. Run from the project folder and screenshot the command with its output.
```bash
grep -nE "task_id=validate_raw_data|quality_report.json" evidence/dag_run_success.log | grep -E "Returned value|SUCCESS" | cut -c1-200
```
**Must show:** `Done. Returned value was: .../quality_report.json` and `Marking task as SUCCESS ... task_id=validate_raw_data`.
**Report:** §4 data-quality gate

### 📸 22: evaluate_model log (Required)
**Save as:** `screenshots/22_airflow_evaluate_model_log.png`
```bash
grep -nE "Metrics:|task_id=evaluate_model" evidence/dag_run_success.log | cut -c1-200
```
**Must show:** the `Metrics: {'accuracy': ..., 'roc_auc': ..., 'malignant_recall': ...}` line.
**Report:** §7.2 / §4 model gate

### 📸 23: smoke_test log (Required)
**Save as:** `screenshots/23_airflow_smoke_test_log.png`
```bash
grep -nE "Smoke test OK|task_id=smoke_test|DagRun Finished" evidence/dag_run_success.log | cut -c1-220
```
**Must show:** `Smoke test OK: {... predictions ...}`. This proves the DAG reached the deployed endpoint.
**Report:** §7.4

### 📸 24: Failed DAG run caused by bad data, Graph view (Required)
**Save as:** `screenshots/24_airflow_dag_failed_graph.png`
Run the bad-data command in terminal 3:
```bash
airflow dags test breast_cancer_serverless_pipeline --conf '{"inject_bad_data": true}'
```
**Must show:** `extract` green, `validate_raw_data` **red**, and every task after it `upstream_failed` (orange).
**Report:** §7.4. This is the key evidence that the quality gate works.

### 📸 25: validate_raw_data log, gate failed (Required)
**Save as:** `screenshots/25_airflow_validate_log_fail.png`
```bash
grep -nE "AirflowFailException|Immediate failure|unrunnable|DagRun failed" evidence/dag_run_bad_data.log | head -5 | cut -c1-220
```
**Must show:** `AirflowFailException: N/65 expectations failed: expect_column_values_to_not_be_null(mean radius), expect_column_values_to_be_between(mean area), expect_column_values_to_be_in_set(target) ...`
**Report:** §7.4 discussion

### 📸 26: Grid view / run history (Optional)
**Save as:** `screenshots/26_airflow_grid_history.png`
**Must show:** both runs side by side (one success, one failed).
**Report:** §7.4

### 📸 27: Quality report file (Optional)
**Save as:** `screenshots/27_quality_report_json.png`
Open `data/<bad run id>/quality_report.json`.
**Must show:** the `"success": false` result and its `failed` list.
**Report:** §4 data-quality gate

---

## Part E: Repository

### 📸 28: Repository page (Required)
**Save as:** `screenshots/28_repository_page.png`
**Must show:** your repo on the HoD-allocated version-control system, with the README rendered and the folders `lambda/`, `dags/`, `pipeline/`, `template.yaml` visible.
**Report:** header (repository link)

---

## Checklist

| # | File name | Screenshot | Required | Taken |
|---|---|---|---|---|
| 01 | `01_pytest_passed.png` | pytest passed | ✅ | ☐ |
| 02 | `02_local_pipeline_metrics.png` | local pipeline metrics | ✅ | ☐ |
| 03 | `03_local_quality_gate_fail.png` | local quality gate fail | | ☐ |
| 04 | `04_metrics_json.png` | metrics.json | | ☐ |
| 05 | `05_sam_validate.png` | sam validate | | ☐ |
| 06 | `06_sam_build.png` | sam build | ✅ | ☐ |
| 07 | `07_invoke_direct_cold_start.png` | invoke direct event (cold) | ✅ | ☐ |
| 08 | `08_invoke_api_event.png` | invoke API event | ✅ | ☐ |
| 09 | `09_sam_local_start_api.png` | sam local start-api | ✅ | ☐ |
| 10 | `10_curl_predictions.png` | curl 2 predictions | ✅ | ☐ |
| 11 | `11_warm_invocation.png` | warm invocation | ✅ | ☐ |
| 12 | `12_bad_request_400.png` | 400 bad request | | ☐ |
| 13 | `13_aws_sam_deploy_outputs.png` | sam deploy outputs | if AWS | ☐ |
| 14 | `14_aws_curl_endpoint.png` | curl real endpoint | if AWS | ☐ |
| 15 | `15_aws_lambda_console.png` | Lambda console | if AWS | ☐ |
| 16 | `16_aws_cloudwatch_logs.png` | CloudWatch logs | if AWS | ☐ |
| 17 | `17_aws_cloudformation_stack.png` | CloudFormation stack | if AWS | ☐ |
| 18 | `18_airflow_dag_list.png` | DAG list | ✅ | ☐ |
| 19 | `19_airflow_variables.png` | Variables | | ☐ |
| 20 | `20_airflow_dag_success_graph.png` | successful DAG graph | ✅ | ☐ |
| 21 | `21_airflow_validate_log_pass.png` | validate log (pass) | ✅ | ☐ |
| 22 | `22_airflow_evaluate_model_log.png` | evaluate_model log | ✅ | ☐ |
| 23 | `23_airflow_smoke_test_log.png` | smoke_test log | ✅ | ☐ |
| 24 | `24_airflow_dag_failed_graph.png` | failed DAG graph (bad data) | ✅ | ☐ |
| 25 | `25_airflow_validate_log_fail.png` | validate log (fail) | ✅ | ☐ |
| 26 | `26_airflow_grid_history.png` | grid view history | | ☐ |
| 27 | `27_quality_report_json.png` | quality_report.json | | ☐ |
| 28 | `28_repository_page.png` | repository page | ✅ | ☐ |

**Numbers to copy into REPORT.md as you go:**
- §7.2: accuracy, ROC-AUC, malignant recall and the confusion matrix (from 02/04)
- §7.3: Init Duration and Duration for the cold call (07), and Duration for the warm call (11)
- §5: whether each AWS service was "used" or "emulated by SAM local"
