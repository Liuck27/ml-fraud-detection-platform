# python3.11 variant: the default 2.7.3 image ships Python 3.8, which is too
# old for the training pins below (matplotlib 3.8.x needs >=3.9). Using 3.11
# keeps the DAG's training environment on the same package versions as
# training/.venv, so models trained via Airflow match host-trained ones.
FROM apache/airflow:2.7.3-python3.11

# Extending the official Airflow image: install extra packages using Airflow's
# own constraints file so pip doesn't fight the pre-resolved dependency graph.
# See: https://airflow.apache.org/docs/docker-stack/build.html
#
# The ML packages (pinned to match training/requirements.txt) let the retrain
# DAG run training/train_xgboost.py in-process via subprocess. Torch is NOT
# installed: the DAG only retrains XGBoost. In production these would live in
# a dedicated training image launched via DockerOperator/KubernetesPodOperator
# instead of being baked into the orchestrator (see retrain_dag.py docstring).
USER airflow
RUN PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") \
 && pip install --no-cache-dir --timeout 120 \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-${PYTHON_VER}.txt" \
    apache-airflow==2.7.3 \
    pandas \
    pyarrow \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    imbalanced-learn==0.11.0 \
    mlflow==2.9.2 \
    matplotlib==3.8.2
