# Container-image Lambda: scikit-learn + numpy + scipy exceed the 250 MB zip limit,
# while an image may be up to 10 GB.
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY app.py ${LAMBDA_TASK_ROOT}/
# Promoted by the Airflow DAG (deploy_model task) or scripts/run_pipeline_local.py
COPY model/ ${LAMBDA_TASK_ROOT}/model/

CMD ["app.lambda_handler"]
