#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export PI_LTC_POINTMAZE_SOURCE="${PI_LTC_POINTMAZE_SOURCE:-/root/autodl-tmp/lewm_data/ogbench/temporal_pointmaze_medium_topdown.h5}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

OUTPUT_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/pointmaze_jepa_wm_vanilla_5pass_seed3072"
CONFIG="$OUTPUT_DIR/vanilla_training_config.yaml"

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

test -f "$PI_LTC_POINTMAZE_SOURCE" || fail "missing source: $PI_LTC_POINTMAZE_SOURCE"

python scripts/generate_pointmaze_jepa_wm_vanilla_training_config.py \
  --output-dir "$OUTPUT_DIR"
GEN_RC=$?
test "$GEN_RC" -eq 0 || fail "vanilla config generation status=$GEN_RC"

python - "$PI_LTC_POINTMAZE_SOURCE" "$CONFIG" <<'PY'
import sys
import h5py
import numpy as np
import yaml

source_path, config_path = sys.argv[1:]
config = yaml.safe_load(open(config_path))
if config["model"]["predictor"].get("planner_identified_input_scale") is not False:
    raise SystemExit("[STOP] vanilla predictor still has PI input scale enabled")
if config.get("planner_identified") != {"enabled": False}:
    raise SystemExit("[STOP] vanilla planner-identification objective is not disabled")
with h5py.File(source_path, "r", swmr=True) as source:
    required = {"pixels", "action", "observation", "ep_idx", "ep_len", "ep_offset"}
    missing = sorted(required.difference(source.keys()))
    if missing:
        raise SystemExit(f"[STOP] source missing keys: {missing}")
    rows = len(source["pixels"])
    lengths = np.asarray(source["ep_len"][:], dtype=np.int64)
    offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
    starts = np.flatnonzero(np.arange(rows, dtype=np.int64) == offsets)
    if len(starts) != 4000 or not np.all(lengths[starts] == 100):
        raise SystemExit("[STOP] canonical source requires 4000 episodes of 100 rows")
    clips = int(0.9 * len(starts)) * (100 - 4 * 5 + 1)
    microbatches = clips // 16 // 8 * 8
    steps = microbatches // 8
    if (clips, microbatches, steps) != (291600, 18224, 2278):
        raise SystemExit(f"[STOP] schedule mismatch: clips={clips} microbatches={microbatches} steps={steps}")
print(f"[preflight] rows={rows} episodes={len(starts)} optimizer_steps/pass={steps}")
PY
PREFLIGHT_RC=$?
test "$PREFLIGHT_RC" -eq 0 || fail "vanilla data/config preflight status=$PREFLIGHT_RC"

if python - "$OUTPUT_DIR/jepa-latest.pth.tar" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
predictor = checkpoint.get("predictor") or {}
scale_keys = [key for key in predictor if key.endswith("planner_input_scale.log_scale")]
if scale_keys:
    raise SystemExit(f"[STOP] vanilla checkpoint unexpectedly contains scale keys: {scale_keys}")
raise SystemExit(0 if int(checkpoint.get("epoch", -1)) >= 5 else 1)
PY
then
  echo "[skip complete] vanilla checkpoint already reached five passes: $OUTPUT_DIR/jepa-latest.pth.tar"
  echo "[terminal remains open]"
  exit 0
fi

mkdir -p training_logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="training_logs/pointmaze_jepa_wm_vanilla_5pass_${STAMP}.log"

echo "================================================================"
echo "[PointMaze JEPA-WM matched vanilla baseline]"
echo "source=$PI_LTC_POINTMAZE_SOURCE"
echo "config=$CONFIG"
echo "schedule=5 passes x 2278 optimizer steps = 11390 steps"
echo "batch=micro16 accumulation8 effective_global128 workers=12"
echo "difference from PI-LTC=global input scale and landscape sidecar objective disabled"
echo "checkpoint=native latest + per-epoch files; automatic resume enabled"
echo "log=$LOG"
echo "================================================================"

python -m app.main \
  --fname "$CONFIG" \
  --devices cuda:0 \
  --debug \
  2>&1 | tee "$LOG"
TRAIN_RC=${PIPESTATUS[0]}

echo "================================================================"
echo "[vanilla training returned] status=$TRAIN_RC"
echo "[log] $LOG"
echo "[output] $OUTPUT_DIR"
echo "[terminal remains open]"
echo "================================================================"
exit "$TRAIN_RC"
