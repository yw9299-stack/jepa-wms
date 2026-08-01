#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EVAL_ENV_PREFIX="${PI_LTC_JEPA_WM_EVAL_ENV:-/root/autodl-tmp/jepa_wms_native_eval_py310}"
EVAL_PYTHON="${PI_LTC_JEPA_WM_EVAL_PYTHON:-$EVAL_ENV_PREFIX/bin/python}"
EVAL_READY_STAMP="$EVAL_ENV_PREFIX/.pi_ltc_native_pointmaze_ready_v2"
MUJOCO_ROOT="${MUJOCO_PY_MUJOCO_PATH:-/root/autodl-tmp/jepa_wms_native_eval_assets/mujoco210}"

MODE="${1:-screen}"
case "$MODE" in
  screen)
    EPISODES=24
    LABEL=screen
    ;;
  official)
    EPISODES=96
    LABEL=official
    ;;
  *)
    echo "Usage: bash $0 {screen|official}"
    exit 2
    ;;
esac

EVAL_SEED="${PI_LTC_EVAL_SEED:-1}"
CHECKPOINT_DIR="${PI_LTC_JEPA_WM_CHECKPOINT_DIR:-/root/autodl-tmp/lewm_data/jepa_wms/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass_seed3072}"
CHECKPOINT="${PI_LTC_JEPA_WM_CHECKPOINT:-jepa-latest.pth.tar}"
OUTPUT_ROOT="$CHECKPOINT_DIR/native_pointmaze_cem30_scale_${LABEL}_seed${EVAL_SEED}_ep${EPISODES}_v3"
TAG=native_cem30_s300_k10_h6_nas6_ctxt2

export PI_LTC_POINTMAZE_SOURCE="${PI_LTC_POINTMAZE_SOURCE:-/root/autodl-tmp/lewm_data/ogbench/temporal_pointmaze_medium_topdown.h5}"
export PI_LTC_POINTMAZE_SIDECAR="${PI_LTC_POINTMAZE_SIDECAR:-/root/autodl-tmp/lewm_data/ogbench/pointmaze_medium_counterfactual_cem0_dinocam_v4_replaystable_seed3072.h5}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms}"
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

cleanup_failed_preflight_output() {
  if [ ! -d "$OUTPUT_ROOT" ]; then
    return 0
  fi
  if [ -f "$OUTPUT_ROOT/native_environment_preflight.json" ]; then
    return 0
  fi
  if find "$OUTPUT_ROOT" -type f -name episode_outcomes.csv -print -quit | grep -q .; then
    fail "incomplete output contains episode outcomes; preserve and inspect it: $OUTPUT_ROOT"
  fi

  local resolved_checkpoint
  local resolved_output
  resolved_checkpoint="$(readlink -m -- "$CHECKPOINT_DIR")"
  resolved_output="$(readlink -m -- "$OUTPUT_ROOT")"
  case "$resolved_output" in
    "$resolved_checkpoint"/native_pointmaze_cem30_scale_*) ;;
    *) fail "refusing to clean unexpected output path: $resolved_output" ;;
  esac
  echo "[cleanup] removing failed preflight-only output: $resolved_output"
  rm -rf -- "$resolved_output"
}

test -f "$CHECKPOINT_DIR/$CHECKPOINT" || fail "missing checkpoint: $CHECKPOINT_DIR/$CHECKPOINT"
test -f "$PI_LTC_POINTMAZE_SOURCE" || fail "missing source: $PI_LTC_POINTMAZE_SOURCE"

if [ ! -x "$EVAL_PYTHON" ] || [ ! -f "$EVAL_READY_STAMP" ]; then
  echo "[setup] isolated Python 3.10 native-eval environment is absent or incomplete"
  bash scripts/setup_pointmaze_native_eval_env_autodl.sh
  SETUP_RC=$?
  test "$SETUP_RC" -eq 0 || fail "native PointMaze environment setup status=$SETUP_RC"
fi
test -x "$EVAL_PYTHON" || fail "evaluation Python not found: $EVAL_PYTHON"

cleanup_failed_preflight_output

"$EVAL_PYTHON" -m unittest tests.models.test_planner_identified_scale
UNIT_RC=$?
test "$UNIT_RC" -eq 0 || fail "scale intervention unit preflight status=$UNIT_RC"

"$EVAL_PYTHON" scripts/generate_pointmaze_pi_ltc_intervention_eval.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --source-h5 "$PI_LTC_POINTMAZE_SOURCE" \
  --episodes "$EPISODES" \
  --eval-seed "$EVAL_SEED" \
  --label "$LABEL"
GEN_RC=$?
test "$GEN_RC" -eq 0 || fail "eval generation status=$GEN_RC"

"$EVAL_PYTHON" scripts/preflight_pointmaze_native_eval.py \
  --config "$OUTPUT_ROOT/configs/learned.yaml" \
  --audit "$OUTPUT_ROOT/native_environment_preflight.json"
ENV_RC=$?
test "$ENV_RC" -eq 0 || fail "native PointMaze construction/render/step preflight status=$ENV_RC"

echo "================================================================"
echo "[JEPA-WM PointMaze same-checkpoint scale intervention]"
echo "training=false"
echo "mode=$MODE episodes=$EPISODES eval_seed=$EVAL_SEED"
echo "checkpoint=$CHECKPOINT_DIR/$CHECKPOINT"
echo "evaluation_python=$EVAL_PYTHON"
echo "mujoco_2_1=$MUJOCO_ROOT"
echo "arms=learned,identity,fixed06,fixed04"
echo "native planner=CEM30 samples300 elites10 horizon6 action_step6"
echo "pairing=same task, episode index, environment seed, and initial planner RNG"
echo "output=$OUTPUT_ROOT"
echo "================================================================"

FAILED=0
for ARM in learned identity fixed06 fixed04; do
  CONFIG="$OUTPUT_ROOT/configs/$ARM.yaml"
  WORK_DIR="$OUTPUT_ROOT/arms/$ARM/simu_env_planning/$TAG"
  OUTCOMES="$WORK_DIR/episode_outcomes.csv"
  AUDIT="$WORK_DIR/planner_scale_audit.json"
  if "$EVAL_PYTHON" - "$OUTCOMES" "$AUDIT" "$EPISODES" <<'PY'
import csv
import json
import sys
from pathlib import Path

outcomes, audit, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if not outcomes.is_file() or not audit.is_file():
    raise SystemExit(1)
with outcomes.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
json.loads(audit.read_text())
raise SystemExit(0 if len(rows) == expected else 1)
PY
  then
    echo "[skip complete] arm=$ARM rows=$EPISODES"
    continue
  fi

  mkdir -p "$OUTPUT_ROOT/logs"
  LOG="$OUTPUT_ROOT/logs/$ARM.log"
  "$EVAL_PYTHON" scripts/run_simu_env_eval_local.py --config "$CONFIG" 2>&1 | tee "$LOG"
  ARM_RC=${PIPESTATUS[0]}
  echo "[arm returned] arm=$ARM status=$ARM_RC"
  if [ "$ARM_RC" -ne 0 ]; then
    FAILED=$((FAILED + 1))
    break
  fi
done

if [ "$FAILED" -eq 0 ]; then
  "$EVAL_PYTHON" scripts/summarize_pointmaze_pi_ltc_intervention.py --output-root "$OUTPUT_ROOT"
  SUMMARY_RC=$?
else
  SUMMARY_RC=1
fi

echo "================================================================"
echo "[JEPA-WM intervention returned] failed_arms=$FAILED summary_status=$SUMMARY_RC"
echo "[output] $OUTPUT_ROOT"
echo "[terminal remains open]"
echo "================================================================"
test "$FAILED" -eq 0 && test "$SUMMARY_RC" -eq 0
