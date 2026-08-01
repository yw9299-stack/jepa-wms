#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${PI_LTC_JEPA_WM_EVAL_ENV:-/root/autodl-tmp/jepa_wms_native_eval_py310}"
ASSET_ROOT="${PI_LTC_JEPA_WM_EVAL_ASSETS:-/root/autodl-tmp/jepa_wms_native_eval_assets}"
MUJOCO_ROOT="${MUJOCO_PY_MUJOCO_PATH:-$ASSET_ROOT/mujoco210}"
CONDA_BIN="${PI_LTC_CONDA_BIN:-/root/miniconda3/bin/conda}"
PYTHON_BIN="$ENV_PREFIX/bin/python"
UV_BIN="$ENV_PREFIX/bin/uv"
READY_STAMP="$ENV_PREFIX/.pi_ltc_native_pointmaze_ready_v2"
POINTMAZE_REQUIREMENTS="$REPO_ROOT/requirements/pointmaze-native-eval.txt"

fail() {
  echo "[STOP] $*"
  exit 1
}

echo "================================================================"
echo "[JEPA-WM native PointMaze evaluation environment]"
echo "repository=$REPO_ROOT"
echo "environment=$ENV_PREFIX"
echo "mujoco_2_1=$MUJOCO_ROOT"
echo "python=3.10 (required by the upstream JEPA-WM repository)"
echo "================================================================"

test -x "$CONDA_BIN" || fail "conda executable not found: $CONDA_BIN"
test -f "$POINTMAZE_REQUIREMENTS" || fail "missing PointMaze requirements: $POINTMAZE_REQUIREMENTS"

# AutoDL commonly places the uv cache and conda prefix on different mounts.
# Copy mode is intentional here and avoids a harmless hardlink warning.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [ ! -x "$PYTHON_BIN" ]; then
  if [ -e "$ENV_PREFIX" ]; then
    RESOLVED_ENV="$(readlink -m -- "$ENV_PREFIX")"
    DEFAULT_ENV="/root/autodl-tmp/jepa_wms_native_eval_py310"
    if [ "$RESOLVED_ENV" != "$DEFAULT_ENV" ]; then
      fail "incomplete custom environment already exists: $RESOLVED_ENV; preserve/inspect it before retrying"
    fi
    echo "[cleanup] removing incomplete conda prefix from the failed setup: $RESOLVED_ENV"
    rm -rf -- "$RESOLVED_ENV"
  fi
  # Ignore the host's .condarc: some AutoDL images still list the retired
  # pkgs/free repository, which returns invalid repodata. Native evaluation
  # does not save video, so only Python and pip are needed from conda here.
  "$CONDA_BIN" create \
    --prefix "$ENV_PREFIX" \
    --override-channels \
    --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    python=3.10 pip \
    -y
  CREATE_RC=$?
  test "$CREATE_RC" -eq 0 || fail "conda environment creation status=$CREATE_RC"
fi

mkdir -p "$ASSET_ROOT"
if [ ! -d "$MUJOCO_ROOT/bin" ]; then
  ARCHIVE="$ASSET_ROOT/mujoco210-linux-x86_64.tar.gz"
  if [ ! -f "$ARCHIVE" ]; then
    wget --tries=5 --timeout=30 \
      -O "$ARCHIVE" \
      https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    DOWNLOAD_RC=$?
    test "$DOWNLOAD_RC" -eq 0 || fail "MuJoCo 2.1 download status=$DOWNLOAD_RC"
  fi
  tar -xzf "$ARCHIVE" -C "$ASSET_ROOT"
  EXTRACT_RC=$?
  test "$EXTRACT_RC" -eq 0 || fail "MuJoCo 2.1 extraction status=$EXTRACT_RC"
fi
test -d "$MUJOCO_ROOT/bin" || fail "invalid MuJoCo 2.1 installation: $MUJOCO_ROOT"

"$PYTHON_BIN" -m pip install --disable-pip-version-check --no-input "uv>=0.8,<0.10"
UV_RC=$?
test "$UV_RC" -eq 0 || fail "uv installation status=$UV_RC"
test -x "$UV_BIN" || fail "uv executable not found after installation: $UV_BIN"

# RTX 5090 requires a Blackwell-capable PyTorch wheel. Install it explicitly
# before resolving the repository dependencies so the generic torch>=2.7
# requirement cannot select an unsuitable CUDA build.
"$UV_BIN" pip install --python "$PYTHON_BIN" \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1" "torchvision==0.22.1"
TORCH_RC=$?
test "$TORCH_RC" -eq 0 || fail "CUDA 12.8 PyTorch installation status=$TORCH_RC"

# The legacy Gym/D4RL/MuJoCo-py stack is not NumPy-2/Cython-3 compatible.
"$UV_BIN" pip install --python "$PYTHON_BIN" \
  "numpy<2" "Cython<3" "setuptools<70" "wheel"
LEGACY_RC=$?
test "$LEGACY_RC" -eq 0 || fail "legacy compatibility dependency status=$LEGACY_RC"

echo "[dependencies] installing the PointMaze-only evaluation runtime"
"$UV_BIN" pip install --python "$PYTHON_BIN" \
  --requirements "$POINTMAZE_REQUIREMENTS"
POINTMAZE_RC=$?
test "$POINTMAZE_RC" -eq 0 || fail "PointMaze-only dependency installation status=$POINTMAZE_RC"

# Do not resolve pyproject.toml here. Its complete research environment also
# installs MetaWorld, PushT, Wall, and other simulators that this evaluation
# neither imports nor executes.
"$UV_BIN" pip install --python "$PYTHON_BIN" --no-deps --editable "$REPO_ROOT"
REPO_RC=$?
test "$REPO_RC" -eq 0 || fail "JEPA-WM no-dependency editable installation status=$REPO_RC"

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_ROOT"
export LD_LIBRARY_PATH="$MUJOCO_ROOT/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"[STOP] expected Python 3.10, found {sys.version}")

import gym
import d4rl
import mujoco_py
import torch

# Import the exact local evaluation entry point now, before generating any
# output or starting an expensive CEM arm. This catches incomplete runtime
# dependencies in one deterministic preflight.
from evals.simu_env_planning.eval import main as _run_native_eval

if not torch.cuda.is_available():
    raise SystemExit("[STOP] CUDA is unavailable in the isolated evaluation environment")

print(f"[dependency preflight] Python={sys.version.split()[0]}")
print(f"[dependency preflight] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
print(f"[dependency preflight] gym={gym.__version__}")
print(f"[dependency preflight] mujoco_py={mujoco_py.__version__}")
print("[dependency preflight] native evaluation entry point=importable")
print("[dependency preflight] PASS")
PY
IMPORT_RC=$?
if [ "$IMPORT_RC" -ne 0 ]; then
  echo "[hint] MuJoCo-py also needs libGL, GLEW, and OSMesa development libraries."
  echo "[hint] If the error names one of those libraries, run:"
  echo "       apt-get update && apt-get install -y libgl1-mesa-dev libglew-dev libosmesa6-dev patchelf"
  fail "native PointMaze dependency preflight status=$IMPORT_RC"
fi

printf '%s\n' \
  "protocol=jepa_wm_native_pointmaze_eval_env_v2" \
  "python=$PYTHON_BIN" \
  "mujoco=$MUJOCO_ROOT" \
  > "$READY_STAMP"

echo "================================================================"
echo "[success] isolated native PointMaze evaluation environment is ready"
echo "[python] $PYTHON_BIN"
echo "[terminal remains open]"
echo "================================================================"
