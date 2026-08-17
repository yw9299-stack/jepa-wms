#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

TASK="${1:-}"
ARM="${2:-}"
case "$TASK/$ARM" in
  pusht/pi|pusht/vanilla|cube/pi|cube/vanilla) ;;
  *) echo "Usage: bash $0 {pusht|cube} {pi|vanilla}"; exit 2 ;;
esac

EXPECTED_COMMIT="${PI_LTC_EXPECTED_COMMIT:?set PI_LTC_EXPECTED_COMMIT to the detached full SHA}"
LEWM_REPO="${LEWM_REPO:-/root/autodl-tmp/lewm_figure6_crossmodel_clean}"
EXPECTED_LEWM_COMMIT="${PI_LTC_EXPECTED_LEWM_COMMIT:-2f4b66934b844a2cacc9454106b9ab163e20c4be}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms}"
NO_FILE_SCAN="${NO_FILE_SCAN:-0}"
CANARY_OPTIMIZER_STEPS="${CANARY_OPTIMIZER_STEPS:-0}"
test "$NO_FILE_SCAN" = 0 -o "$NO_FILE_SCAN" = 1 || { echo "[STOP] NO_FILE_SCAN must be 0 or 1"; exit 1; }
[[ "$CANARY_OPTIMIZER_STEPS" =~ ^[0-9]+$ ]] || { echo "[STOP] CANARY_OPTIMIZER_STEPS must be a non-negative integer"; exit 1; }
if [ "$CANARY_OPTIMIZER_STEPS" -gt 0 ] && [ "$ARM" != pi ]; then
  echo "[STOP] canary mode is PI-only"
  exit 1
fi

fail() { echo "[STOP] $*"; exit 1; }

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" || fail "JEPA HEAD is not $EXPECTED_COMMIT"
git diff --quiet || fail "JEPA tracked worktree is dirty"
git diff --cached --quiet || fail "JEPA index is dirty"
if [ "$NO_FILE_SCAN" = 0 ]; then
  test -d "$LEWM_REPO/.git" -o -f "$LEWM_REPO/.git" || fail "missing clean LEWM worktree: $LEWM_REPO"
  test "$(git -C "$LEWM_REPO" rev-parse HEAD)" = "$EXPECTED_LEWM_COMMIT" || fail "LEWM commit mismatch"
  git -C "$LEWM_REPO" diff --quiet || fail "LEWM tracked worktree is dirty"
  git -C "$LEWM_REPO" diff --cached --quiet || fail "LEWM index is dirty"
fi

if [ "$TASK" = pusht ]; then
  export PI_LTC_PUSHT_SOURCE="${PI_LTC_PUSHT_SOURCE:-/root/autodl-tmp/lewm_data/pusht_expert_train.h5}"
  export PI_LTC_PUSHT_SIDECAR="${PI_LTC_PUSHT_SIDECAR:-/root/autodl-tmp/lewm_data/pusht_planner_counterfactual_cem0_seed3072.h5}"
  SOURCE="$PI_LTC_PUSHT_SOURCE"
  SIDECAR="$PI_LTC_PUSHT_SIDECAR"
  EXPECTED_SOURCE_BYTES=46300921856
  EXPECTED_SIDECAR_BYTES=192116437
  PI_RUN="pusht_jepa_wm_pi_ltc_step111464_v5_seed3072"
  VANILLA_RUN="pusht_jepa_wm_vanilla_step111464_v5_seed3072"
  OPTIMIZER_STEP_BUDGET=111464
  LEWM_REFERENCE_PASSES=8
  DRIVER_EPOCHS=9
  DEFAULT_TARGET_COMPLETE_PASSES=1
else
  export PI_LTC_CUBE_SOURCE="${PI_LTC_CUBE_SOURCE:-/root/autodl-tmp/lewm_data/ogbench/cube_single_expert.h5}"
  export PI_LTC_CUBE_SIDECAR="${PI_LTC_CUBE_SIDECAR:-/root/autodl-tmp/lewm_data/ogbench/cube_counterfactual_cem0_seed3072.h5}"
  SOURCE="$PI_LTC_CUBE_SOURCE"
  SIDECAR="$PI_LTC_CUBE_SIDECAR"
  EXPECTED_SOURCE_BYTES=101942558720
  EXPECTED_SIDECAR_BYTES=1282198884
  PI_RUN="cube_jepa_wm_pi_ltc_step51184_v5_seed3072"
  VANILLA_RUN="cube_jepa_wm_vanilla_step51184_v5_seed3072"
  OPTIMIZER_STEP_BUDGET=51184
  LEWM_REFERENCE_PASSES=4
  DRIVER_EPOCHS=4
  DEFAULT_TARGET_COMPLETE_PASSES=4
fi

TARGET_COMPLETE_PASSES="${TARGET_COMPLETE_PASSES:-$DEFAULT_TARGET_COMPLETE_PASSES}"
[[ "$TARGET_COMPLETE_PASSES" =~ ^[1-9][0-9]*$ ]] || fail "TARGET_COMPLETE_PASSES must be a positive integer"
test "$TARGET_COMPLETE_PASSES" -le "$LEWM_REFERENCE_PASSES" || fail "TARGET_COMPLETE_PASSES exceeds the configured complete-pass horizon"
echo "[configured schedule] task=$TASK reference_passes=$LEWM_REFERENCE_PASSES optimizer_steps=$OPTIMIZER_STEP_BUDGET driver_epochs=$DRIVER_EPOCHS"
echo "[runtime budget] stop cleanly after complete_passes=$TARGET_COMPLETE_PASSES (PushT default=1; set 2 to resume both matched arms through pass 2)"

if [ "$NO_FILE_SCAN" = 1 ]; then
  # SHA-256 of the literal marker, not of either data file.  Provenance records
  # the mode explicitly so this cannot be mistaken for content verification.
  SOURCE_SHA=f259ec4020fbbe9ab7765c4ca9dc2d67c6c21b7c5b11b8b6bab564fb77c5a5ba
  SIDECAR_SHA="$SOURCE_SHA"
  export PI_LTC_CONTENT_HASH_MODE=not_scanned_user_confirmed
else
  if [ "$TASK" = pusht ]; then
    SOURCE_SHA="${PI_LTC_PUSHT_SOURCE_SHA256:?set PI_LTC_PUSHT_SOURCE_SHA256}"
    SIDECAR_SHA="${PI_LTC_PUSHT_SIDECAR_SHA256:?set PI_LTC_PUSHT_SIDECAR_SHA256}"
  else
    SOURCE_SHA="${PI_LTC_CUBE_SOURCE_SHA256:?set PI_LTC_CUBE_SOURCE_SHA256}"
    SIDECAR_SHA="${PI_LTC_CUBE_SIDECAR_SHA256:?set PI_LTC_CUBE_SIDECAR_SHA256}"
  fi
  export PI_LTC_CONTENT_HASH_MODE=sha256
fi

export PI_LTC_REPOSITORY_COMMIT="$EXPECTED_COMMIT"
export PI_LTC_SOURCE_SHA256="$SOURCE_SHA"
export PI_LTC_SIDECAR_SHA256="$SIDECAR_SHA"

test -f "$SOURCE" || fail "missing source: $SOURCE"
test -f "$SIDECAR" || fail "missing sidecar: $SIDECAR"
test "$(stat -c %s "$SOURCE")" = "$EXPECTED_SOURCE_BYTES" || fail "source byte size differs"
test "$(stat -c %s "$SIDECAR")" = "$EXPECTED_SIDECAR_BYTES" || fail "sidecar byte size differs"
test "${#SOURCE_SHA}" -eq 64 || fail "source SHA-256 must have 64 characters"
test "${#SIDECAR_SHA}" -eq 64 || fail "sidecar SHA-256 must have 64 characters"

python scripts/generate_stablewm_task_training_configs.py \
  --task "$TASK" \
  --output-root "$JEPAWM_LOGS" \
  --source-sha256 "$SOURCE_SHA" \
  --sidecar-sha256 "$SIDECAR_SHA"

PI_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$PI_RUN"
VANILLA_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$VANILLA_RUN"
PI_CONFIG="$PI_DIR/pi_training_config.yaml"
VANILLA_CONFIG="$VANILLA_DIR/vanilla_training_config.yaml"
PREFLIGHT_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/preflight/$TASK/$EXPECTED_COMMIT"
DATA_AUDIT="$PREFLIGHT_DIR/data_training_preflight.json"
MODEL_AUDIT="$PREFLIGHT_DIR/model_preflight.json"
PI_SMOKE_CKPT="$PREFLIGHT_DIR/pi_preflight.pth.tar"
VANILLA_SMOKE_CKPT="$PREFLIGHT_DIR/vanilla_preflight.pth.tar"

if [ "$NO_FILE_SCAN" = 1 ]; then
  echo "[located only; no content scan] task=$TASK"
  echo "[source] path=$(realpath "$SOURCE") bytes=$(stat -c %s "$SOURCE")"
  echo "[sidecar] path=$(realpath "$SIDECAR") bytes=$(stat -c %s "$SIDECAR")"
  echo "[user override] skipping content hashes, standalone data preflight, unit/model preflight, and LEWM smoke"
  python - "$TASK" "$EXPECTED_COMMIT" "$SOURCE" "$SIDECAR" "$SOURCE_SHA" "$SIDECAR_SHA" "$PI_CONFIG" "$VANILLA_CONFIG" "$DATA_AUDIT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

(
    task,
    commit,
    source_text,
    sidecar_text,
    source_sha,
    sidecar_sha,
    pi_config_text,
    vanilla_config_text,
    output_text,
) = sys.argv[1:]
source = Path(source_text).resolve()
sidecar = Path(sidecar_text).resolve()
pi_config = Path(pi_config_text).resolve()
vanilla_config = Path(vanilla_config_text).resolve()
output = Path(output_text)

def sha256_small(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

pi = yaml.safe_load(pi_config.read_text(encoding="utf-8"))
steps_per_pass = int(
    pi["optimization"]["transition_model"][
        "expected_optimizer_steps_per_epoch"
    ]
)
audit = {
    "status": "PASS",
    "verification_mode": "not_scanned_user_confirmed",
    "task": task,
    "repository_commit": commit,
    "training_configs": {
        "pi": {"path": str(pi_config), "sha256": sha256_small(pi_config)},
        "vanilla": {
            "path": str(vanilla_config),
            "sha256": sha256_small(vanilla_config),
        },
    },
    "source": {
        "path": str(source),
        "bytes": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "sha256": source_sha,
        "content_not_read": True,
    },
    "split": {
        "optimizer_steps_per_pass": steps_per_pass,
        "optimizer_step_budget": int(
            pi["optimization"]["transition_model"]["total_optimizer_steps"]
        ),
    },
    "planner": {
        "sidecar_path": str(sidecar),
        "sidecar_bytes": sidecar.stat().st_size,
        "sidecar_mtime_ns": sidecar.stat().st_mtime_ns,
        "sidecar_sha256": sidecar_sha,
        "content_not_read": True,
    },
}
payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists() and output.read_text(encoding="utf-8") != payload:
    raise SystemExit(f"[STOP] immutable located-only audit differs: {output}")
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(payload, encoding="utf-8")
os.replace(temporary, output)
print(f"[located-only audit PASS] {output}")
PY
else
reuse_data_preflight=0
if [ -f "$DATA_AUDIT" ]; then
  if python - "$DATA_AUDIT" "$SOURCE" "$SIDECAR" "$SOURCE_SHA" "$SIDECAR_SHA" "$EXPECTED_COMMIT" "$PI_CONFIG" "$VANILLA_CONFIG" <<'PY'
import hashlib, json, os, sys
audit_path, source, sidecar, source_sha, sidecar_sha, commit, pi_config, vanilla_config = sys.argv[1:]
audit = json.load(open(audit_path))
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
ok = (
    audit.get("status") == "PASS"
    and audit.get("repository_commit") == commit
    and audit.get("source", {}).get("path") == os.path.realpath(source)
    and audit.get("source", {}).get("bytes") == os.stat(source).st_size
    and audit.get("source", {}).get("mtime_ns") == os.stat(source).st_mtime_ns
    and audit.get("source", {}).get("sha256") == source_sha
    and audit.get("planner", {}).get("sidecar_path") == os.path.realpath(sidecar)
    and audit.get("planner", {}).get("sidecar_bytes") == os.stat(sidecar).st_size
    and audit.get("planner", {}).get("sidecar_mtime_ns") == os.stat(sidecar).st_mtime_ns
    and audit.get("planner", {}).get("sidecar_sha256") == sidecar_sha
    and audit.get("training_configs", {}).get("pi", {}).get("path") == os.path.realpath(pi_config)
    and audit.get("training_configs", {}).get("pi", {}).get("sha256") == sha256(pi_config)
    and audit.get("training_configs", {}).get("vanilla", {}).get("path") == os.path.realpath(vanilla_config)
    and audit.get("training_configs", {}).get("vanilla", {}).get("sha256") == sha256(vanilla_config)
)
raise SystemExit(0 if ok else 1)
PY
  then reuse_data_preflight=1; fi
fi

if [ "$reuse_data_preflight" -eq 1 ]; then
  echo "[reuse complete] $DATA_AUDIT"
else
  python scripts/preflight_stablewm_task_training.py \
    --task "$TASK" \
    --source "$SOURCE" \
    --sidecar "$SIDECAR" \
    --source-sha256 "$SOURCE_SHA" \
    --sidecar-sha256 "$SIDECAR_SHA" \
    --pi-config "$PI_CONFIG" \
    --vanilla-config "$VANILLA_CONFIG" \
    --expected-commit "$EXPECTED_COMMIT" \
    --output "$DATA_AUDIT"
fi

python -m unittest \
  tests.models.test_planner_identified_scale \
  tests.models.test_planner_landscape_loss \
  tests.datasets.test_stablewm_pi_ltc_h5 \
  tests.scripts.test_stablewm_task_training_protocol \
  tests.scripts.test_stablewm_task_summary

python scripts/preflight_stablewm_task_model.py \
  --task "$TASK" \
  --device cuda \
  --checkpoint-output "$PI_SMOKE_CKPT" \
  --vanilla-checkpoint-output "$VANILLA_SMOKE_CKPT" \
  --output "$MODEL_AUDIT"

if [ "$TASK" = pusht ]; then
  python scripts/smoke_planner_identified_cross_model.py \
    --model jepa_wm_pusht \
    --device cuda
fi

SMOKE_ROOT="$PREFLIGHT_DIR/evaluator"
for smoke_arm in learned vanilla_stepmatched; do
  if [ "$smoke_arm" = learned ]; then
    smoke_owner=pi
    smoke_config="$PI_CONFIG"
    smoke_checkpoint="$(basename "$PI_SMOKE_CKPT")"
  else
    smoke_owner=vanilla
    smoke_config="$VANILLA_CONFIG"
    smoke_checkpoint="$(basename "$VANILLA_SMOKE_CKPT")"
  fi
  python scripts/eval_stablewm_task_protocol.py \
    --task "$TASK" \
    --lewm-repo "$LEWM_REPO" \
    --source-h5 "$SOURCE" \
    --source-sha256 "$SOURCE_SHA" \
    --sidecar-sha256 "$SIDECAR_SHA" \
    --preflight-audit "$DATA_AUDIT" \
    --training-config "$smoke_config" \
    --checkpoint-dir "$PREFLIGHT_DIR" \
    --checkpoint "$smoke_checkpoint" \
    --owner "$smoke_owner" \
    --arm "$smoke_arm" \
    --output-root "$SMOKE_ROOT" \
    --eval-seed 42 \
    --episodes 1 \
    --candidate-chunk-size 32 \
    --expected-commit "$EXPECTED_COMMIT" \
    --expected-lewm-commit "$EXPECTED_LEWM_COMMIT" \
    --device cuda \
    --smoke \
    --reuse-complete
done

echo "[preflight PASS] task=$TASK data=$DATA_AUDIT model=$MODEL_AUDIT evaluator=$SMOKE_ROOT"
fi
if [ "${PREFLIGHT_ONLY:-1}" = 1 ]; then
  echo "[preflight only] Set PREFLIGHT_ONLY=0 only after reviewing all audits."
  exit 0
fi

if [ "$CANARY_OPTIMIZER_STEPS" -gt 0 ]; then
  CANARY_RUN="${PI_RUN}_canary${CANARY_OPTIMIZER_STEPS}_${EXPECTED_COMMIT:0:12}"
  CANARY_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$CANARY_RUN"
  CANARY_CONFIG="$CANARY_DIR/pi_canary_config.yaml"
  python - "$PI_CONFIG" "$CANARY_CONFIG" "$CANARY_DIR" "$CANARY_OPTIMIZER_STEPS" <<'PY'
import os
import sys
from pathlib import Path

import yaml

source, destination, folder, optimizer_steps = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
config["folder"] = folder
config["meta"]["canary_optimizer_steps"] = int(optimizer_steps)
config["meta"]["load_checkpoint"] = False
config["meta"]["read_checkpoint"] = None
config["meta"]["pretrained_path"] = None
payload = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
path = Path(destination)
if path.exists():
    if path.read_bytes() != payload:
        raise SystemExit(f"[STOP] immutable canary config differs: {path}")
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
print(f"[canary config] {path}")
PY
  mkdir -p "$CANARY_DIR/logs"
  CANARY_LOG="$CANARY_DIR/logs/${TASK}_pi_canary_$(date +%Y%m%d_%H%M%S).log"
  echo "[canary launch] task=$TASK optimizer_steps=$CANARY_OPTIMIZER_STEPS config=$CANARY_CONFIG"
  set +e
  PI_LTC_STOP_AFTER_COMPLETE_PASSES=0 \
    python -m app.main --fname "$CANARY_CONFIG" --devices cuda:0 --debug 2>&1 | tee "$CANARY_LOG"
  CANARY_RC=${PIPESTATUS[0]}
  set -e
  test "$CANARY_RC" -eq 0 || fail "canary returned status=$CANARY_RC"
  test -f "$CANARY_DIR/canary_summary.json" || fail "canary summary is missing"
  python - "$CANARY_DIR/canary_summary.json" <<'PY' || fail "canary safety gate rejected this configuration"
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if summary.get("canary_gate_passed") is True else 1)
PY
  echo "[canary complete] summary=$CANARY_DIR/canary_summary.json"
  exit 0
fi

if [ "$ARM" = pi ]; then
  RUN_DIR="$PI_DIR"
  CONFIG="$PI_CONFIG"
  OWNER=pi
else
  RUN_DIR="$VANILLA_DIR"
  CONFIG="$VANILLA_CONFIG"
  OWNER=vanilla
fi
LATEST="$RUN_DIR/jepa-latest.pth.tar"

checkpoint_status() {
  python - "$LATEST" "$OWNER" "$EXPECTED_COMMIT" "$SOURCE_SHA" "$SIDECAR_SHA" "$DATA_AUDIT" "$NO_FILE_SCAN" "$PI_LTC_CONTENT_HASH_MODE" "$OPTIMIZER_STEP_BUDGET" "$LEWM_REFERENCE_PASSES" "$DRIVER_EPOCHS" "$TARGET_COMPLETE_PASSES" <<'PY'
import json, sys
from pathlib import Path
import torch
(
    path,
    owner,
    commit,
    source_sha,
    sidecar_sha,
    audit_path,
    no_file_scan,
    hash_mode,
    budget_text,
    reference_passes_text,
    driver_epochs_text,
    target_passes_text,
) = sys.argv[1:]
budget = int(budget_text)
reference_passes = int(reference_passes_text)
driver_epochs = int(driver_epochs_text)
target_passes = int(target_passes_text)
path = Path(path)
if not path.is_file():
    raise SystemExit(1)
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
predictor = checkpoint.get("predictor") or {}
scale = [key for key in predictor if key.endswith("planner_input_scale.log_scale")]
if (owner == "pi" and len(scale) != 1) or (owner == "vanilla" and scale):
    raise SystemExit(2)
provenance = checkpoint.get("training_provenance") or {}
if (
    provenance.get("repository_commit") != commit
    or provenance.get("content_hash_mode") != hash_mode
    or provenance.get("source_h5", {}).get("sha256") != source_sha
    or provenance.get("sidecar_h5", {}).get("sha256") != sidecar_sha
):
    raise SystemExit(2)
if no_file_scan == "1":
    steps = int(checkpoint.get("optimizer_steps_per_epoch", 0))
else:
    steps = json.load(open(audit_path))["split"]["optimizer_steps_per_pass"]
if steps < 1 or checkpoint.get("optimizer_steps_per_epoch") != steps:
    raise SystemExit(2)
initialization = checkpoint.get("common_trainable_initialization_sha256")
if (
    not isinstance(initialization, str)
    or len(initialization) != 64
    or any(character not in "0123456789abcdef" for character in initialization)
):
    raise SystemExit(2)
epoch = int(checkpoint.get("epoch", -1))
total_steps = int(checkpoint.get("total_optimizer_steps", -1))
step_in_epoch = int(checkpoint.get("optimizer_step_in_epoch", 0))
if checkpoint.get("optimizer_step_budget") != budget:
    raise SystemExit(2)
expected_full_passes, expected_tail_steps = divmod(budget, steps)
if expected_full_passes != reference_passes:
    raise SystemExit(2)
if checkpoint.get("training_complete") is True:
    if (
        total_steps != budget
        or epoch != expected_full_passes
        or step_in_epoch != expected_tail_steps
    ):
        raise SystemExit(2)
    print(
        f"[complete checkpoint] lewm_reference_passes={reference_passes} "
        f"full_jepa_passes={epoch} tail_steps={step_in_epoch} "
        f"total_steps={total_steps} scale_keys={scale}"
    )
    raise SystemExit(0)
if (
    target_passes <= epoch <= expected_full_passes
    and step_in_epoch == 0
    and total_steps == epoch * steps
    and total_steps < budget
):
    print(
        f"[requested pass boundary complete] requested={target_passes} "
        f"completed_passes={epoch} steps/pass={steps} total_steps={total_steps}"
    )
    raise SystemExit(0)
if (
    0 <= epoch < target_passes
    and step_in_epoch == 0
    and total_steps == epoch * steps
    and total_steps < budget
):
    print(
        f"[strict resume] completed_passes={epoch} "
        f"steps/pass={steps} total_steps={total_steps}/{budget}"
    )
    raise SystemExit(1)
raise SystemExit(2)
PY
}

set +e
checkpoint_status
CHECKPOINT_RC=$?
set -e
case "$CHECKPOINT_RC" in
  0) echo "[skip complete] $LATEST"; exit 0 ;;
  1) ;;
  *) fail "existing checkpoint failed strict provenance/schedule validation" ;;
esac

mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/${TASK}_${ARM}_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] task=$TASK arm=$ARM target_complete_passes=$TARGET_COMPLETE_PASSES commit=$EXPECTED_COMMIT config=$CONFIG log=$LOG"
set +e
PI_LTC_STOP_AFTER_COMPLETE_PASSES="$TARGET_COMPLETE_PASSES" \
  python -m app.main --fname "$CONFIG" --devices cuda:0 --debug 2>&1 | tee "$LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e
test "$TRAIN_RC" -eq 0 || fail "training returned status=$TRAIN_RC"

set +e
checkpoint_status
CHECKPOINT_RC=$?
set -e
test "$CHECKPOINT_RC" -eq 0 || fail "training returned without a complete strict checkpoint"
echo "[training target complete] task=$TASK arm=$ARM complete_passes=$TARGET_COMPLETE_PASSES checkpoint=$LATEST"
