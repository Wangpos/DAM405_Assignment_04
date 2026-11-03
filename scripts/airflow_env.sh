# Usage (from the repo root):  source scripts/airflow_env.sh
if [ ! -d "dags" ] || [ ! -d "pipeline" ]; then
  echo "Run this from the repository root" >&2
  return 1
fi

export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # lets tasks import the `pipeline` package

# macOS: avoid crashes when Airflow forks task processes
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export NO_PROXY="*"

echo "AIRFLOW_HOME=$AIRFLOW_HOME"
