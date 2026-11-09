#!/bin/sh
# Start the ImageToSplat local service on Linux: linux-service.sh <port>
#
# On first run this creates a Python environment with uv and installs PyTorch and the other packages
# (several GB). Lines starting with "SETUP" are shown in the app while it waits.
#
#   IMAGETOSPLAT_DATA_DIR   where the environment, models and outputs live
#                           (default: $XDG_DATA_HOME/NekoLegends/ImageToSplat, i.e. ~/.local/share/...)
#   IMAGETOSPLAT_ENV_DIR    the Python environment (default: <data dir>/env)
#   IMAGETOSPLAT_PYTHON     use this Python as it is, and install nothing
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE
[ -f "$ROOT/run_gradio.py" ] || ROOT=$(cd "$HERE/.." && pwd)
PORT=${1:-17340}
DATA=${IMAGETOSPLAT_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/NekoLegends/ImageToSplat}
mkdir -p "$DATA"

PY=${IMAGETOSPLAT_PYTHON:-}
if [ -z "$PY" ]; then
  ENV_DIR=${IMAGETOSPLAT_ENV_DIR:-$DATA/env}
  PY=$ENV_DIR/bin/python
  WANT=$(cksum < "$ROOT/requirements-app.txt")
  if [ ! -x "$PY" ] || [ "$(cat "$ENV_DIR/.requirements" 2>/dev/null || true)" != "$WANT" ]; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "SETUP-ERROR uv was not found. Install uv (https://docs.astral.sh/uv/), or set IMAGETOSPLAT_PYTHON to a Python that has the packages in requirements-app.txt."
      exit 3
    fi
    if [ ! -x "$PY" ]; then
      echo "SETUP Creating the Python environment"
      uv venv --python 3.12 "$ENV_DIR"
    fi
    echo "SETUP Installing PyTorch and other packages (first run only: several GB, a few minutes)"
    uv pip install --python "$PY" -r "$ROOT/requirements-app.txt"
    echo "$WANT" > "$ENV_DIR/.requirements"
  fi
fi

echo "SETUP Starting the local service"
export IMAGETOSPLAT_DATA_DIR="$DATA"
export IMAGETOSPLAT_SCENE_PYTHON="${IMAGETOSPLAT_SCENE_PYTHON:-$PY}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
exec "$PY" run_gradio.py --serve-agent-api --agent-api-port "$PORT"
