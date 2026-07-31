#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${1:-}"
case "$MODEL" in
  jepa_wm)
    CONFIG="configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml"
    RUN_NAME="pointmaze_jepa_wm_pi_ltc_5pass"
    ;;
  dino_wm)
    CONFIG="configs/pi_ltc_cross_model/pointmaze_dino_wm_pi_ltc_5pass.yaml"
    RUN_NAME="pointmaze_dino_wm_pi_ltc_5pass"
    ;;
  *)
    echo "Usage: bash $0 {jepa_wm|dino_wm}"
    exit 2
    ;;
esac

export PI_LTC_POINTMAZE_SOURCE="${PI_LTC_POINTMAZE_SOURCE:-/root/autodl-tmp/lewm_data/ogbench/temporal_pointmaze_medium_topdown.h5}"
export PI_LTC_POINTMAZE_SIDECAR="${PI_LTC_POINTMAZE_SIDECAR:-/root/autodl-tmp/lewm_data/ogbench/pointmaze_medium_counterfactual_cem0_dinocam_v4_replaystable_seed3072.h5}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

if [ -n "${HDF5_PLUGIN_PATH:-}" ] && [ ! -d "$HDF5_PLUGIN_PATH" ]; then
  unset HDF5_PLUGIN_PATH
fi

if [ -f /etc/network_turbo ]; then
  # Needed the first time torch.hub downloads the frozen DINOv2 encoder.
  source /etc/network_turbo 2>/dev/null || true
fi

fail() {
  echo "[STOP] $*"
  exit 1
}

test -f "$CONFIG" || fail "missing config: $CONFIG"
test -f "$PI_LTC_POINTMAZE_SOURCE" || fail "missing source: $PI_LTC_POINTMAZE_SOURCE"
test -f "$PI_LTC_POINTMAZE_SIDECAR" || fail "missing sidecar: $PI_LTC_POINTMAZE_SIDECAR"

python - "$PI_LTC_POINTMAZE_SOURCE" "$PI_LTC_POINTMAZE_SIDECAR" <<'PY'
import sys
import h5py
import numpy as np

source_path, sidecar_path = sys.argv[1:]
with h5py.File(source_path, "r", swmr=True) as source:
    required = {
        "pixels",
        "action",
        "observation",
        "ep_idx",
        "ep_len",
        "ep_offset",
    }
    missing = sorted(required.difference(source.keys()))
    if missing:
        raise SystemExit(f"[STOP] source missing keys: {missing}")
    rows = len(source["pixels"])
    row_lengths = np.asarray(source["ep_len"][:], dtype=np.int64)
    row_offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
    row_episode_ids = np.asarray(source["ep_idx"][:], dtype=np.int64)
    if not (
        row_lengths.shape
        == row_offsets.shape
        == row_episode_ids.shape
        == (rows,)
    ):
        raise SystemExit("[STOP] episode metadata is not a per-row source schema")
    episode_starts = np.flatnonzero(
        np.arange(rows, dtype=np.int64) == row_offsets
    )
    episode_lengths = row_lengths[episode_starts]
    episode_ids = row_episode_ids[episode_starts]
    episodes = len(episode_starts)
    if episodes != 4000 or not np.all(episode_lengths == 100):
        raise SystemExit(
            "[STOP] canonical schedule requires 4000 physical episodes "
            "of exactly 100 rows"
        )
    if (
        not np.array_equal(episode_starts, row_offsets[episode_starts])
        or not np.array_equal(
            episode_ids, np.arange(episodes, dtype=np.int64)
        )
    ):
        raise SystemExit("[STOP] physical episode IDs/offsets are inconsistent")
    train_episodes = int(0.9 * episodes)
    clips = train_episodes * (100 - 4 * 5 + 1)
    loader_microbatches = clips // 16
    used_microbatches = loader_microbatches // 8 * 8
    optimizer_steps = used_microbatches // 8
    if (clips, loader_microbatches, used_microbatches, optimizer_steps) != (
        291600,
        18225,
        18224,
        2278,
    ):
        raise SystemExit("[STOP] computed schedule differs from the canonical schedule")
with h5py.File(sidecar_path, "r", swmr=True) as sidecar:
    protocol = sidecar.attrs.get("protocol", "")
    if isinstance(protocol, bytes):
        protocol = protocol.decode()
    if protocol != "pointmaze_planner_counterfactual_v1":
        raise SystemExit(f"[STOP] sidecar protocol={protocol!r}")
    if not bool(sidecar.attrs.get("complete", False)):
        raise SystemExit("[STOP] sidecar is incomplete")
    groups = len(sidecar["selected_context_rows"])
    branches = int(sidecar.attrs["branches_per_state"])
    if groups != 10000 or branches != 4:
        raise SystemExit(
            f"[STOP] canonical sidecar requires 10000x4 groups, got {groups}x{branches}"
        )
print(
    f"[preflight] source episodes={episodes} rows={rows}; "
    f"sidecar groups={groups} branches/group={branches}; "
    f"clips={clips} microbatches={used_microbatches} optimizer_steps/pass={optimizer_steps}"
)
PY
PREFLIGHT_RC=$?
test "$PREFLIGHT_RC" -eq 0 || exit "$PREFLIGHT_RC"

python -m unittest \
  tests.models.test_planner_identified_scale \
  tests.models.test_planner_landscape_loss \
  tests.datasets.test_stablewm_pi_ltc_h5
UNIT_RC=$?
test "$UNIT_RC" -eq 0 || fail "PI-LTC unit preflight failed with status=$UNIT_RC"

python scripts/smoke_planner_identified_cross_model.py \
  --model "${MODEL}_pointmaze" \
  --device cuda
SMOKE_RC=$?
test "$SMOKE_RC" -eq 0 || fail "real-checkpoint PI-LTC smoke failed with status=$SMOKE_RC"

mkdir -p training_logs "$JEPAWM_LOGS"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="training_logs/${RUN_NAME}_${STAMP}.log"

echo "================================================================"
echo "[PointMaze cross-model PI-LTC]"
echo "model=$MODEL"
echo "config=$CONFIG"
echo "source=$PI_LTC_POINTMAZE_SOURCE"
echo "sidecar=$PI_LTC_POINTMAZE_SIDECAR"
echo "schedule=5 passes x 2278 optimizer steps = 11390 steps"
echo "batch=micro16 accumulation8 effective_global128"
echo "workers=12 ordinary + 4 landscape"
echo "gradient ownership:"
echo "  ordinary MSE -> predictor/action/proprio (scale detached)"
echo "  held-out landscape -> global log_input_scale only"
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
echo "[training returned] model=$MODEL status=$TRAIN_RC"
echo "[log] $LOG"
echo "[output root] $JEPAWM_LOGS/pi_ltc_cross_model"
echo "[terminal remains open]"
echo "================================================================"
exit "$TRAIN_RC"
