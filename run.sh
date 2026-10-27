set -e

.venv/bin/python -m pip install -e '.[workflow]'
mkdir -p work/dagster
export DAGSTER_HOME="$PWD/work/dagster"
export ICU_RAW_DATA="${ICU_RAW_DATA:-$PWD/data}"
.venv/bin/python -m dagster dev -f src/workflow.py --host 127.0.0.1 --port 3000
