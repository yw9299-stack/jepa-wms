#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EPISODES=24
EVAL_SEED="${PI_LTC_EVAL_SEED:-1}"
EVAL_ENV_PREFIX="${PI_LTC_JEPA_WM_EVAL_ENV:-/root/autodl-tmp/jepa_wms_native_eval_py310}"
EVAL_PYTHON="${PI_LTC_JEPA_WM_EVAL_PYTHON:-$EVAL_ENV_PREFIX/bin/python}"
EVAL_READY_STAMP="$EVAL_ENV_PREFIX/.pi_ltc_native_pointmaze_ready_v2"
MUJOCO_ROOT="${MUJOCO_PY_MUJOCO_PATH:-/root/autodl-tmp/jepa_wms_native_eval_assets/mujoco210}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms}"

OFFICIAL_DIR="${JEPA_WM_OFFICIAL_CHECKPOINT_DIR:-$JEPAWM_LOGS/pi_ltc_cross_model/pointmaze_jepa_wm_official_checkpoint}"
OFFICIAL_CHECKPOINT="mz_jepa-wm.pth.tar"
OFFICIAL_URL="https://dl.fbaipublicfiles.com/jepa-wms/mz_jepa-wm.pth.tar"
VANILLA_DIR="${JEPA_WM_VANILLA_CHECKPOINT_DIR:-$JEPAWM_LOGS/pi_ltc_cross_model/pointmaze_jepa_wm_vanilla_5pass_seed3072}"
TRAINING_CONFIG="$VANILLA_DIR/vanilla_training_config.yaml"
PI_ROOT="${PI_LTC_JEPA_WM_PI_EVAL_ROOT:-$JEPAWM_LOGS/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass_seed3072/native_pointmaze_cem30_scale_screen_seed${EVAL_SEED}_ep${EPISODES}_v3}"
VANILLA_ROOT="${JEPA_WM_VANILLA_EVAL_ROOT:-$VANILLA_DIR/native_pointmaze_cem30_screen_seed${EVAL_SEED}_ep${EPISODES}}"
OUTPUT_ROOT="$OFFICIAL_DIR/native_pointmaze_cem30_official_checkpoint_screen_seed${EVAL_SEED}_ep${EPISODES}"
TAG=native_cem30_s300_k10_h6_nas6_ctxt2

export PI_LTC_POINTMAZE_SOURCE="${PI_LTC_POINTMAZE_SOURCE:-/root/autodl-tmp/lewm_data/ogbench/temporal_pointmaze_medium_topdown.h5}"
export JEPAWM_DSET="${JEPAWM_DSET:-/root/autodl-tmp/lewm_data/jepa_wms/datasets}"
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

test -f "$TRAINING_CONFIG" || fail "missing matched vanilla config: $TRAINING_CONFIG"
test -f "$PI_LTC_POINTMAZE_SOURCE" || fail "missing source: $PI_LTC_POINTMAZE_SOURCE"
test -f "$PI_ROOT/run_manifest.json" || fail "missing PI screen results: $PI_ROOT"
test -f "$VANILLA_ROOT/run_manifest.json" || fail "missing matched vanilla screen results: $VANILLA_ROOT"

if [ ! -x "$EVAL_PYTHON" ] || [ ! -f "$EVAL_READY_STAMP" ]; then
  echo "[setup] isolated Python 3.10 native-eval environment is absent or incomplete"
  bash scripts/setup_pointmaze_native_eval_env_autodl.sh
  SETUP_RC=$?
  test "$SETUP_RC" -eq 0 || fail "native PointMaze environment setup status=$SETUP_RC"
fi
test -x "$EVAL_PYTHON" || fail "evaluation Python not found: $EVAL_PYTHON"

mkdir -p "$OFFICIAL_DIR"
if [ ! -f "$OFFICIAL_DIR/$OFFICIAL_CHECKPOINT" ]; then
  PARTIAL="$OFFICIAL_DIR/$OFFICIAL_CHECKPOINT.partial"
  echo "[download] official Meta PointMaze JEPA-WM checkpoint"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 8 --retry-delay 3 --retry-all-errors \
      --continue-at - --output "$PARTIAL" "$OFFICIAL_URL" || \
      fail "official checkpoint download failed; partial file retained for resume"
  else
    wget --tries=8 --continue --output-document="$PARTIAL" "$OFFICIAL_URL" || \
      fail "official checkpoint download failed; partial file retained for resume"
  fi
  mv "$PARTIAL" "$OFFICIAL_DIR/$OFFICIAL_CHECKPOINT"
fi

AUDIT_ARGS=(
  --source-h5 "$PI_LTC_POINTMAZE_SOURCE"
  --current-config configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml
  --official-config configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml
  --output "$OFFICIAL_DIR/data_alignment_audit.json"
  --samples 128
)
if [ -f "$JEPAWM_DSET/point_maze/actions.pth" ]; then
  AUDIT_ARGS+=(--official-dataset "$JEPAWM_DSET/point_maze")
fi
"$EVAL_PYTHON" scripts/audit_pointmaze_jepa_wm_data_alignment.py "${AUDIT_ARGS[@]}"
AUDIT_RC=$?
test "$AUDIT_RC" -eq 0 || fail "data/preprocessing alignment audit status=$AUDIT_RC"

"$EVAL_PYTHON" scripts/generate_pointmaze_jepa_wm_official_checkpoint_eval.py \
  --checkpoint-dir "$OFFICIAL_DIR" \
  --checkpoint "$OFFICIAL_CHECKPOINT" \
  --source-h5 "$PI_LTC_POINTMAZE_SOURCE" \
  --training-config "$TRAINING_CONFIG" \
  --episodes "$EPISODES" \
  --eval-seed "$EVAL_SEED" \
  --label official_checkpoint_screen
GEN_RC=$?
test "$GEN_RC" -eq 0 || fail "official-checkpoint eval generation status=$GEN_RC"

"$EVAL_PYTHON" scripts/preflight_pointmaze_native_eval.py \
  --config "$OUTPUT_ROOT/configs/official.yaml" \
  --audit "$OUTPUT_ROOT/native_environment_preflight.json"
ENV_RC=$?
test "$ENV_RC" -eq 0 || fail "native PointMaze preflight status=$ENV_RC"

echo "================================================================"
echo "[JEPA-WM official-checkpoint sanity test]"
echo "training=false episodes=$EPISODES eval_seed=$EVAL_SEED"
echo "controlled variable=predictor checkpoint only"
echo "same native environment, starts, planner, H5 normalization, and preprocessing"
echo "checkpoint=$OFFICIAL_DIR/$OFFICIAL_CHECKPOINT"
echo "expected runtime=one arm, about 32 minutes"
echo "output=$OUTPUT_ROOT"
echo "================================================================"

WORK_DIR="$OUTPUT_ROOT/arms/official/simu_env_planning/$TAG"
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
  echo "[skip complete] official checkpoint rows=$EPISODES"
  EVAL_RC=0
else
  LOG="$OUTPUT_ROOT/logs/official.log"
  "$EVAL_PYTHON" scripts/run_simu_env_eval_local.py \
    --config "$OUTPUT_ROOT/configs/official.yaml" 2>&1 | tee "$LOG"
  EVAL_RC=${PIPESTATUS[0]}
  echo "[official arm returned] status=$EVAL_RC"
fi

if [ "$EVAL_RC" -eq 0 ]; then
  "$EVAL_PYTHON" scripts/summarize_pointmaze_jepa_wm_official_sanity.py \
    --official-root "$OUTPUT_ROOT" \
    --pi-root "$PI_ROOT" \
    --vanilla-root "$VANILLA_ROOT"
  SUMMARY_RC=$?
else
  SUMMARY_RC=1
fi

echo "================================================================"
echo "[official checkpoint sanity returned] eval_status=$EVAL_RC summary_status=$SUMMARY_RC"
echo "[output] $OUTPUT_ROOT"
echo "[terminal remains open]"
echo "================================================================"
test "$EVAL_RC" -eq 0 && test "$SUMMARY_RC" -eq 0
