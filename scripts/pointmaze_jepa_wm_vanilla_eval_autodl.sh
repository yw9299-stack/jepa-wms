#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-screen}"
case "$MODE" in
  screen) EPISODES=24; LABEL=screen ;;
  official) EPISODES=96; LABEL=official ;;
  *) echo "Usage: bash $0 {screen|official}"; exit 2 ;;
esac

EVAL_SEED="${PI_LTC_EVAL_SEED:-1}"
EVAL_ENV_PREFIX="${PI_LTC_JEPA_WM_EVAL_ENV:-/root/autodl-tmp/jepa_wms_native_eval_py310}"
EVAL_PYTHON="${PI_LTC_JEPA_WM_EVAL_PYTHON:-$EVAL_ENV_PREFIX/bin/python}"
EVAL_READY_STAMP="$EVAL_ENV_PREFIX/.pi_ltc_native_pointmaze_ready_v2"
MUJOCO_ROOT="${MUJOCO_PY_MUJOCO_PATH:-/root/autodl-tmp/jepa_wms_native_eval_assets/mujoco210}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms}"
CHECKPOINT_DIR="${JEPA_WM_VANILLA_CHECKPOINT_DIR:-$JEPAWM_LOGS/pi_ltc_cross_model/pointmaze_jepa_wm_vanilla_5pass_seed3072}"
CHECKPOINT="${JEPA_WM_VANILLA_CHECKPOINT:-jepa-latest.pth.tar}"
TRAINING_CONFIG="$CHECKPOINT_DIR/vanilla_training_config.yaml"
OUTPUT_ROOT="$CHECKPOINT_DIR/native_pointmaze_cem30_${LABEL}_seed${EVAL_SEED}_ep${EPISODES}"
PI_ROOT="${PI_LTC_JEPA_WM_PI_EVAL_ROOT:-$JEPAWM_LOGS/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass_seed3072/native_pointmaze_cem30_scale_${LABEL}_seed${EVAL_SEED}_ep${EPISODES}_v3}"
TAG=native_cem30_s300_k10_h6_nas6_ctxt2

export PI_LTC_POINTMAZE_SOURCE="${PI_LTC_POINTMAZE_SOURCE:-/root/autodl-tmp/lewm_data/ogbench/temporal_pointmaze_medium_topdown.h5}"
export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_ROOT"
export LD_LIBRARY_PATH="$MUJOCO_ROOT/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ -n "${HDF5_PLUGIN_PATH:-}" ] && [ ! -d "$HDF5_PLUGIN_PATH" ]; then
  unset HDF5_PLUGIN_PATH
fi
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo 2>/dev/null || true
fi

fail() {
  echo "[STOP] $*"
  exit 1
}

test -f "$CHECKPOINT_DIR/$CHECKPOINT" || fail "missing vanilla checkpoint: $CHECKPOINT_DIR/$CHECKPOINT"
test -f "$TRAINING_CONFIG" || fail "missing generated vanilla training config: $TRAINING_CONFIG"
test -f "$PI_LTC_POINTMAZE_SOURCE" || fail "missing source: $PI_LTC_POINTMAZE_SOURCE"
test -f "$PI_ROOT/run_manifest.json" || fail "missing completed PI comparison run: $PI_ROOT"

if [ ! -x "$EVAL_PYTHON" ] || [ ! -f "$EVAL_READY_STAMP" ]; then
  echo "[setup] isolated Python 3.10 native-eval environment is absent or incomplete"
  bash scripts/setup_pointmaze_native_eval_env_autodl.sh
  SETUP_RC=$?
  test "$SETUP_RC" -eq 0 || fail "native PointMaze environment setup status=$SETUP_RC"
fi
test -x "$EVAL_PYTHON" || fail "evaluation Python not found: $EVAL_PYTHON"

"$EVAL_PYTHON" scripts/generate_pointmaze_jepa_wm_vanilla_eval.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --source-h5 "$PI_LTC_POINTMAZE_SOURCE" \
  --training-config "$TRAINING_CONFIG" \
  --episodes "$EPISODES" \
  --eval-seed "$EVAL_SEED" \
  --label "$LABEL"
GEN_RC=$?
test "$GEN_RC" -eq 0 || fail "vanilla eval generation status=$GEN_RC"

"$EVAL_PYTHON" scripts/preflight_pointmaze_native_eval.py \
  --config "$OUTPUT_ROOT/configs/vanilla.yaml" \
  --audit "$OUTPUT_ROOT/native_environment_preflight.json"
ENV_RC=$?
test "$ENV_RC" -eq 0 || fail "native PointMaze preflight status=$ENV_RC"

echo "================================================================"
echo "[JEPA-WM PointMaze matched vanilla evaluation]"
echo "training=false mode=$MODE episodes=$EPISODES eval_seed=$EVAL_SEED"
echo "checkpoint=$CHECKPOINT_DIR/$CHECKPOINT"
echo "single arm=vanilla_5pass (completed PI learned outcomes are reused)"
echo "native planner=CEM30 samples300 elites10 horizon6 action_step6"
echo "expected runtime=about one arm (~32 minutes), not four arms (~2 hours)"
echo "output=$OUTPUT_ROOT"
echo "================================================================"

WORK_DIR="$OUTPUT_ROOT/arms/vanilla/simu_env_planning/$TAG"
OUTCOMES="$WORK_DIR/episode_outcomes.csv"
if "$EVAL_PYTHON" - "$OUTCOMES" "$EPISODES" <<'PY'
import csv
import sys
from pathlib import Path

path, expected = Path(sys.argv[1]), int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
with path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
raise SystemExit(0 if len(rows) == expected else 1)
PY
then
  echo "[skip complete] vanilla rows=$EPISODES"
  EVAL_RC=0
else
  LOG="$OUTPUT_ROOT/logs/vanilla.log"
  "$EVAL_PYTHON" scripts/run_simu_env_eval_local.py \
    --config "$OUTPUT_ROOT/configs/vanilla.yaml" 2>&1 | tee "$LOG"
  EVAL_RC=${PIPESTATUS[0]}
  echo "[vanilla arm returned] status=$EVAL_RC"
fi

if [ "$EVAL_RC" -eq 0 ]; then
  "$EVAL_PYTHON" scripts/summarize_pointmaze_jepa_wm_vanilla_comparison.py \
    --vanilla-root "$OUTPUT_ROOT" \
    --pi-root "$PI_ROOT"
  SUMMARY_RC=$?
else
  SUMMARY_RC=1
fi

echo "================================================================"
echo "[vanilla evaluation returned] eval_status=$EVAL_RC summary_status=$SUMMARY_RC"
echo "[output] $OUTPUT_ROOT"
echo "[terminal remains open]"
echo "================================================================"
test "$EVAL_RC" -eq 0 && test "$SUMMARY_RC" -eq 0
