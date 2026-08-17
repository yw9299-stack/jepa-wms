#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXPECTED_EVALUATOR_COMMIT="${PI_LTC_EXPECTED_COMMIT:?set PI_LTC_EXPECTED_COMMIT}"
EXPECTED_TRAINING_COMMIT="${PI_LTC_EXPECTED_TRAINING_COMMIT:-90bb004351a1e4c42f0e99d20601fb69dcd22fdf}"
LEWM_REPO="${LEWM_REPO:-/root/autodl-tmp/lewm_figure6_crossmodel_clean}"
EXPECTED_LEWM_COMMIT="${PI_LTC_EXPECTED_LEWM_COMMIT:-2f4b66934b844a2cacc9454106b9ab163e20c4be}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms_noscan}"

fail() { echo "[STOP] $*" >&2; exit 1; }
test "$(git rev-parse HEAD)" = "$EXPECTED_EVALUATOR_COMMIT" || fail "JEPA evaluator commit mismatch"
git diff --quiet || fail "JEPA tracked worktree is dirty"
git diff --cached --quiet || fail "JEPA index is dirty"
test -d "$LEWM_REPO/.git" -o -f "$LEWM_REPO/.git" || fail "missing LEWM worktree: $LEWM_REPO"
test "$(git -C "$LEWM_REPO" rev-parse HEAD)" = "$EXPECTED_LEWM_COMMIT" || fail "LEWM commit mismatch"
git -C "$LEWM_REPO" diff --quiet || fail "LEWM tracked worktree is dirty"
git -C "$LEWM_REPO" diff --cached --quiet || fail "LEWM index is dirty"

SOURCE="${PI_LTC_PUSHT_SOURCE:-/root/autodl-tmp/lewm_data/pusht_expert_train.h5}"
SIDECAR="${PI_LTC_PUSHT_SIDECAR:-/root/autodl-tmp/lewm_data/pusht_planner_counterfactual_cem0_seed3072.h5}"
PI_RUN=pusht_jepa_wm_pi_ltc_step111464_v4_seed3072
VANILLA_RUN=pusht_jepa_wm_vanilla_step111464_v4_seed3072
PI_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$PI_RUN"
VANILLA_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$VANILLA_RUN"
PI_CONFIG="$PI_DIR/pi_training_config.yaml"
VANILLA_CONFIG="$VANILLA_DIR/vanilla_training_config.yaml"
PI_CHECKPOINT="$PI_DIR/jepa-e1.pth.tar"
VANILLA_CHECKPOINT="$VANILLA_DIR/jepa-e1.pth.tar"
RELAY_SUMMARY="$JEPAWM_LOGS/pi_ltc_cross_model/pusht_second_pass_relay/relay_summary.json"
OUTPUT_ROOT="$JEPAWM_LOGS/pi_ltc_cross_model/pusht_twopass_seed42_clean_v1"
PREFLIGHT_AUDIT="$OUTPUT_ROOT/preflight.json"
SUMMARY="$OUTPUT_ROOT/pusht/seed42/pi_vs_vanilla_summary.json"

for required in "$SOURCE" "$SIDECAR" "$PI_CONFIG" "$VANILLA_CONFIG" "$PI_CHECKPOINT" "$VANILLA_CHECKPOINT" "$RELAY_SUMMARY"; do
  test -f "$required" || fail "missing required artifact: $required"
done

python scripts/preflight_pusht_one_pass_seed42_eval.py \
  --relay-summary "$RELAY_SUMMARY" \
  --pi-config "$PI_CONFIG" \
  --vanilla-config "$VANILLA_CONFIG" \
  --pi-checkpoint "$PI_CHECKPOINT" \
  --vanilla-checkpoint "$VANILLA_CHECKPOINT" \
  --source-h5 "$SOURCE" \
  --sidecar-h5 "$SIDECAR" \
  --output "$PREFLIGHT_AUDIT" \
  --expected-completed-passes 2

mapfile -t PROVENANCE < <(python - "$PREFLIGHT_AUDIT" <<'PY'
import json
import sys
audit = json.load(open(sys.argv[1]))
print(audit["source"]["sha256"])
print(audit["planner"]["sidecar_sha256"])
PY
)
test "${#PROVENANCE[@]}" -eq 2 || fail "could not read preflight provenance"
SOURCE_SHA="${PROVENANCE[0]}"
SIDECAR_SHA="${PROVENANCE[1]}"

run_arm() {
  local owner="$1" arm="$2" config="$3" directory="$4"
  python scripts/eval_stablewm_task_protocol.py \
    --task pusht \
    --lewm-repo "$LEWM_REPO" \
    --source-h5 "$SOURCE" \
    --source-sha256 "$SOURCE_SHA" \
    --sidecar-sha256 "$SIDECAR_SHA" \
    --preflight-audit "$PREFLIGHT_AUDIT" \
    --training-config "$config" \
    --checkpoint-dir "$directory" \
    --checkpoint jepa-e1.pth.tar \
    --owner "$owner" \
    --arm "$arm" \
    --output-root "$OUTPUT_ROOT" \
    --eval-seed 42 \
    --episodes 50 \
    --candidate-chunk-size 32 \
    --expected-commit "$EXPECTED_EVALUATOR_COMMIT" \
    --expected-training-commit "$EXPECTED_TRAINING_COMMIT" \
    --expected-lewm-commit "$EXPECTED_LEWM_COMMIT" \
    --expected-completed-passes 2 \
    --device cuda \
    --reuse-complete
}

echo "[seed42 clean pass-2] evaluating PI-LTC learned checkpoint"
run_arm pi learned "$PI_CONFIG" "$PI_DIR"
echo "[seed42 clean pass-2] evaluating matched vanilla checkpoint"
run_arm vanilla vanilla_stepmatched "$VANILLA_CONFIG" "$VANILLA_DIR"

python scripts/summarize_stablewm_task_seed_pair.py \
  --input-root "$OUTPUT_ROOT" \
  --task pusht \
  --eval-seed 42 \
  --output "$SUMMARY" \
  --bootstrap-draws 10000 \
  --bootstrap-seed 20260817

echo "[seed42 clean pass-2 complete] summary=$SUMMARY"
