#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASK="${1:-}"
case "$TASK" in pusht|cube) ;; *) echo "Usage: bash $0 {pusht|cube}"; exit 2 ;; esac

EXPECTED_COMMIT="${PI_LTC_EXPECTED_COMMIT:?set PI_LTC_EXPECTED_COMMIT}"
LEWM_REPO="${LEWM_REPO:-/root/autodl-tmp/lewm_figure6_crossmodel_clean}"
EXPECTED_LEWM_COMMIT="${PI_LTC_EXPECTED_LEWM_COMMIT:-2f4b66934b844a2cacc9454106b9ab163e20c4be}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms}"

fail() { echo "[STOP] $*"; exit 1; }
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" || fail "JEPA commit mismatch"
git diff --quiet || fail "JEPA tracked worktree is dirty"
git diff --cached --quiet || fail "JEPA index is dirty"
test "$(git -C "$LEWM_REPO" rev-parse HEAD)" = "$EXPECTED_LEWM_COMMIT" || fail "LEWM commit mismatch"
git -C "$LEWM_REPO" diff --quiet || fail "LEWM tracked worktree is dirty"
git -C "$LEWM_REPO" diff --cached --quiet || fail "LEWM index is dirty"

if [ "$TASK" = pusht ]; then
  SOURCE="${PI_LTC_PUSHT_SOURCE:-/root/autodl-tmp/lewm_data/pusht_expert_train.h5}"
  SOURCE_SHA="${PI_LTC_PUSHT_SOURCE_SHA256:?set PI_LTC_PUSHT_SOURCE_SHA256}"
  SIDECAR_SHA="${PI_LTC_PUSHT_SIDECAR_SHA256:?set PI_LTC_PUSHT_SIDECAR_SHA256}"
  PI_RUN=pusht_jepa_wm_pi_ltc_step111464_v4_seed3072
  VANILLA_RUN=pusht_jepa_wm_vanilla_step111464_v4_seed3072
else
  SOURCE="${PI_LTC_CUBE_SOURCE:-/root/autodl-tmp/lewm_data/ogbench/cube_single_expert.h5}"
  SOURCE_SHA="${PI_LTC_CUBE_SOURCE_SHA256:?set PI_LTC_CUBE_SOURCE_SHA256}"
  SIDECAR_SHA="${PI_LTC_CUBE_SIDECAR_SHA256:?set PI_LTC_CUBE_SIDECAR_SHA256}"
  PI_RUN=cube_jepa_wm_pi_ltc_step51184_v4_seed3072
  VANILLA_RUN=cube_jepa_wm_vanilla_step51184_v4_seed3072
fi

PI_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$PI_RUN"
VANILLA_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$VANILLA_RUN"
PI_CONFIG="$PI_DIR/pi_training_config.yaml"
VANILLA_CONFIG="$VANILLA_DIR/vanilla_training_config.yaml"
PI_CHECKPOINT="$PI_DIR/jepa-latest.pth.tar"
VANILLA_CHECKPOINT="$VANILLA_DIR/jepa-latest.pth.tar"
DATA_AUDIT="$JEPAWM_LOGS/pi_ltc_cross_model/preflight/$TASK/$EXPECTED_COMMIT/data_training_preflight.json"
OUTPUT_ROOT="$JEPAWM_LOGS/pi_ltc_cross_model/stablewm_exact_multiseed_v1"

for required in "$SOURCE" "$PI_CONFIG" "$VANILLA_CONFIG" "$PI_CHECKPOINT" "$VANILLA_CHECKPOINT" "$DATA_AUDIT"; do
  test -f "$required" || fail "missing required artifact: $required"
done

run_arm() {
  local seed="$1" owner="$2" arm="$3" config="$4" directory="$5"
  python scripts/eval_stablewm_task_protocol.py \
    --task "$TASK" \
    --lewm-repo "$LEWM_REPO" \
    --source-h5 "$SOURCE" \
    --source-sha256 "$SOURCE_SHA" \
    --sidecar-sha256 "$SIDECAR_SHA" \
    --preflight-audit "$DATA_AUDIT" \
    --training-config "$config" \
    --checkpoint-dir "$directory" \
    --checkpoint jepa-latest.pth.tar \
    --owner "$owner" \
    --arm "$arm" \
    --output-root "$OUTPUT_ROOT" \
    --eval-seed "$seed" \
    --episodes 50 \
    --candidate-chunk-size 32 \
    --expected-commit "$EXPECTED_COMMIT" \
    --expected-lewm-commit "$EXPECTED_LEWM_COMMIT" \
    --device cuda \
    --reuse-complete
}

for seed in 42 43 44; do
  run_arm "$seed" pi learned "$PI_CONFIG" "$PI_DIR"
  run_arm "$seed" pi identity "$PI_CONFIG" "$PI_DIR"
  run_arm "$seed" vanilla vanilla_stepmatched "$VANILLA_CONFIG" "$VANILLA_DIR"
done

if [ "${RUN_OPTIONAL_FIXED:-0}" = 1 ]; then
  run_arm 42 pi fixed06 "$PI_CONFIG" "$PI_DIR"
  run_arm 42 pi fixed04 "$PI_CONFIG" "$PI_DIR"
fi

SUMMARY="$OUTPUT_ROOT/$TASK/multiseed_summary.json"
python scripts/summarize_stablewm_task_multiseed.py \
  --input-root "$OUTPUT_ROOT" \
  --task "$TASK" \
  --output "$SUMMARY" \
  --bootstrap-draws 10000 \
  --bootstrap-seed 20260815
echo "[evaluation complete] task=$TASK summary=$SUMMARY"
