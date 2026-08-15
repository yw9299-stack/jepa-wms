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

fail() { echo "[STOP] $*"; exit 1; }

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" || fail "JEPA HEAD is not $EXPECTED_COMMIT"
git diff --quiet || fail "JEPA tracked worktree is dirty"
git diff --cached --quiet || fail "JEPA index is dirty"
test -d "$LEWM_REPO/.git" -o -f "$LEWM_REPO/.git" || fail "missing clean LEWM worktree: $LEWM_REPO"
test "$(git -C "$LEWM_REPO" rev-parse HEAD)" = "$EXPECTED_LEWM_COMMIT" || fail "LEWM commit mismatch"
git -C "$LEWM_REPO" diff --quiet || fail "LEWM tracked worktree is dirty"
git -C "$LEWM_REPO" diff --cached --quiet || fail "LEWM index is dirty"

if [ "$TASK" = pusht ]; then
  export PI_LTC_PUSHT_SOURCE="${PI_LTC_PUSHT_SOURCE:-/root/autodl-tmp/lewm_data/pusht_expert_train.h5}"
  export PI_LTC_PUSHT_SIDECAR="${PI_LTC_PUSHT_SIDECAR:-/root/autodl-tmp/lewm_data/pusht_planner_counterfactual_cem0_seed3072.h5}"
  SOURCE="$PI_LTC_PUSHT_SOURCE"
  SIDECAR="$PI_LTC_PUSHT_SIDECAR"
  SOURCE_SHA="${PI_LTC_PUSHT_SOURCE_SHA256:?set PI_LTC_PUSHT_SOURCE_SHA256}"
  SIDECAR_SHA="${PI_LTC_PUSHT_SIDECAR_SHA256:?set PI_LTC_PUSHT_SIDECAR_SHA256}"
  PI_RUN="pusht_jepa_wm_pi_ltc_5pass_seed3072"
  VANILLA_RUN="pusht_jepa_wm_vanilla_5pass_seed3072"
else
  export PI_LTC_CUBE_SOURCE="${PI_LTC_CUBE_SOURCE:-/root/autodl-tmp/lewm_data/ogbench/cube_single_expert.h5}"
  export PI_LTC_CUBE_SIDECAR="${PI_LTC_CUBE_SIDECAR:-/root/autodl-tmp/lewm_data/ogbench/cube_counterfactual_cem0_seed3072.h5}"
  SOURCE="$PI_LTC_CUBE_SOURCE"
  SIDECAR="$PI_LTC_CUBE_SIDECAR"
  SOURCE_SHA="${PI_LTC_CUBE_SOURCE_SHA256:?set PI_LTC_CUBE_SOURCE_SHA256}"
  SIDECAR_SHA="${PI_LTC_CUBE_SIDECAR_SHA256:?set PI_LTC_CUBE_SIDECAR_SHA256}"
  PI_RUN="cube_jepa_wm_pi_ltc_5pass_seed3072"
  VANILLA_RUN="cube_jepa_wm_vanilla_5pass_seed3072"
fi

export PI_LTC_REPOSITORY_COMMIT="$EXPECTED_COMMIT"
export PI_LTC_SOURCE_SHA256="$SOURCE_SHA"
export PI_LTC_SIDECAR_SHA256="$SIDECAR_SHA"

test -f "$SOURCE" || fail "missing source: $SOURCE"
test -f "$SIDECAR" || fail "missing sidecar: $SIDECAR"
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
for smoke_arm in learned vanilla_5pass; do
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
if [ "${PREFLIGHT_ONLY:-1}" = 1 ]; then
  echo "[preflight only] Set PREFLIGHT_ONLY=0 only after reviewing all audits."
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
  python - "$LATEST" "$OWNER" "$EXPECTED_COMMIT" "$SOURCE_SHA" "$SIDECAR_SHA" "$DATA_AUDIT" <<'PY'
import json, sys
from pathlib import Path
import torch
path, owner, commit, source_sha, sidecar_sha, audit_path = sys.argv[1:]
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
    or provenance.get("source_h5", {}).get("sha256") != source_sha
    or provenance.get("sidecar_h5", {}).get("sha256") != sidecar_sha
):
    raise SystemExit(2)
steps = json.load(open(audit_path))["split"]["optimizer_steps_per_pass"]
if checkpoint.get("optimizer_steps_per_epoch") != steps:
    raise SystemExit(2)
initialization = checkpoint.get("common_trainable_initialization_sha256")
if (
    not isinstance(initialization, str)
    or len(initialization) != 64
    or any(character not in "0123456789abcdef" for character in initialization)
):
    raise SystemExit(2)
epoch = int(checkpoint.get("epoch", -1))
if epoch == 5:
    if checkpoint.get("total_optimizer_steps") != 5 * steps:
        raise SystemExit(2)
    print(f"[complete checkpoint] epoch={epoch} steps={5 * steps} scale_keys={scale}")
    raise SystemExit(0)
if 0 <= epoch < 5:
    print(f"[strict resume] epoch={epoch} steps/pass={steps}")
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
echo "[launch] task=$TASK arm=$ARM commit=$EXPECTED_COMMIT config=$CONFIG log=$LOG"
set +e
python -m app.main --fname "$CONFIG" --devices cuda:0 --debug 2>&1 | tee "$LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e
test "$TRAIN_RC" -eq 0 || fail "training returned status=$TRAIN_RC"

set +e
checkpoint_status
CHECKPOINT_RC=$?
set -e
test "$CHECKPOINT_RC" -eq 0 || fail "training returned without a complete strict checkpoint"
echo "[training complete] task=$TASK arm=$ARM checkpoint=$LATEST"
