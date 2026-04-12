FROM apache/airflow:2.7.3

# Extending the official Airflow image: install extra packages using Airflow's
# own constraints file so pip doesn't fight the pre-resolved dependency graph.
# See: https://airflow.apache.org/docs/docker-stack/build.html
USER airflow
RUN PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") \
 && pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-${PYTHON_VER}.txt" \
    apache-airflow==2.7.3 \
    pandas \
    pyarrow
