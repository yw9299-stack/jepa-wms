#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

fail() { echo "[STOP] $*"; exit 1; }

EXPECTED_COMMIT="${PI_LTC_EXPECTED_COMMIT:?set PI_LTC_EXPECTED_COMMIT to the detached full SHA}"
CANARY_TRAINING_COMMIT="${CANARY_TRAINING_COMMIT:-3afd2173f244426cd82af001add337f80f5ebe37}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms_noscan}"
CANARY_DIR="${CANARY_DIR:-$JEPAWM_LOGS/pi_ltc_cross_model/pusht_jepa_wm_pi_ltc_step111464_v6_seed3072_canary2000_3afd2173f244}"
CANARY_SUMMARY="$CANARY_DIR/canary_summary.json"
CANARY_CONFIG="$CANARY_DIR/pi_canary_config.yaml"
SWEEP_DIR="$CANARY_DIR/extended_scale_sweep_${EXPECTED_COMMIT:0:12}"
SWEEP_CONFIG="$SWEEP_DIR/planner_scale_sweep_only.yaml"
SWEEP_ARTIFACT="$SWEEP_DIR/planner_scale_sweep_only.json"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" || fail "JEPA HEAD is not $EXPECTED_COMMIT"
git diff --quiet || fail "JEPA tracked worktree is dirty"
git diff --cached --quiet || fail "JEPA index is dirty"
test -f "$CANARY_SUMMARY" || fail "missing canary summary: $CANARY_SUMMARY"
test -f "$CANARY_CONFIG" || fail "missing immutable canary config: $CANARY_CONFIG"

mapfile -t CHECKPOINT_AUDIT < <(
  python - "$CANARY_SUMMARY" "$CANARY_TRAINING_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

summary_path = Path(sys.argv[1]).resolve()
expected_training_commit = sys.argv[2]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") not in {"CANARY_COMPLETE", "CANARY_REJECTED"}:
    raise SystemExit("[STOP] source canary summary is incomplete")
audit = summary.get("canary_checkpoint") or {}
checkpoint_path = Path(audit.get("path", "")).resolve()
if not checkpoint_path.is_file():
    raise SystemExit(f"[STOP] missing canary checkpoint: {checkpoint_path}")

digest = hashlib.sha256()
with checkpoint_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
checkpoint_sha256 = digest.hexdigest()
if checkpoint_sha256 != audit.get("sha256"):
    raise SystemExit("[STOP] canary checkpoint SHA-256 differs from its summary")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
for key, expected in (
    ("checkpoint_role", "canary_diagnostic"),
    ("total_optimizer_steps", 2000),
    ("training_complete", False),
):
    if checkpoint.get(key) != expected or audit.get(key) != expected:
        raise SystemExit(
            f"[STOP] invalid canary checkpoint metadata {key}: "
            f"checkpoint={checkpoint.get(key)!r} summary={audit.get(key)!r}"
        )
predictor = checkpoint.get("predictor") or {}
scale_keys = [
    key for key in predictor if key.endswith("planner_input_scale.log_scale")
]
if len(scale_keys) != 1:
    raise SystemExit("[STOP] canary checkpoint has no unique PI-LTC scale parameter")
provenance = checkpoint.get("training_provenance") or {}
if provenance.get("repository_commit") != expected_training_commit:
    raise SystemExit("[STOP] canary training commit differs")
if provenance.get("content_hash_mode") != "not_scanned_user_confirmed":
    raise SystemExit("[STOP] this launcher requires no-scan dataset provenance")
source = provenance.get("source_h5") or {}
sidecar = provenance.get("sidecar_h5") or {}
for label, record in (("source", source), ("sidecar", sidecar)):
    if not Path(record.get("path", "")).is_file():
        raise SystemExit(f"[STOP] located {label} file is missing")
    marker = str(record.get("sha256", ""))
    if len(marker) != 64:
        raise SystemExit(f"[STOP] invalid {label} no-scan marker")

print(checkpoint_path)
print(checkpoint_sha256)
print(source["path"])
print(source["bytes"])
print(source["sha256"])
print(sidecar["path"])
print(sidecar["bytes"])
print(sidecar["sha256"])
PY
)
test "${#CHECKPOINT_AUDIT[@]}" -eq 8 || fail "checkpoint audit did not return eight fields"

CHECKPOINT_PATH="${CHECKPOINT_AUDIT[0]}"
CHECKPOINT_SHA256="${CHECKPOINT_AUDIT[1]}"
SOURCE_PATH="${CHECKPOINT_AUDIT[2]}"
SOURCE_BYTES="${CHECKPOINT_AUDIT[3]}"
SOURCE_MARKER="${CHECKPOINT_AUDIT[4]}"
SIDECAR_PATH="${CHECKPOINT_AUDIT[5]}"
SIDECAR_BYTES="${CHECKPOINT_AUDIT[6]}"
SIDECAR_MARKER="${CHECKPOINT_AUDIT[7]}"

# The dataset files are only located and size-checked.  They are not hashed or
# sampled by this launcher; the explicit no-scan markers come from the source
# checkpoint's immutable training provenance.
test -f "$SOURCE_PATH" || fail "missing source HDF5: $SOURCE_PATH"
test -f "$SIDECAR_PATH" || fail "missing planner sidecar: $SIDECAR_PATH"
test "$(stat -c %s "$SOURCE_PATH")" = "$SOURCE_BYTES" || fail "source byte size differs"
test "$(stat -c %s "$SIDECAR_PATH")" = "$SIDECAR_BYTES" || fail "sidecar byte size differs"
echo "[located only; no dataset scan] source=$SOURCE_PATH bytes=$SOURCE_BYTES"
echo "[located only; no dataset scan] sidecar=$SIDECAR_PATH bytes=$SIDECAR_BYTES"

export PI_LTC_PUSHT_SOURCE="$SOURCE_PATH"
export PI_LTC_PUSHT_SIDECAR="$SIDECAR_PATH"
export PI_LTC_REPOSITORY_COMMIT="$EXPECTED_COMMIT"
export PI_LTC_SOURCE_SHA256="$SOURCE_MARKER"
export PI_LTC_SIDECAR_SHA256="$SIDECAR_MARKER"
export PI_LTC_CONTENT_HASH_MODE=not_scanned_user_confirmed

python - \
  "$CANARY_CONFIG" \
  "$SWEEP_CONFIG" \
  "$SWEEP_DIR" \
  "$CHECKPOINT_PATH" \
  "$CHECKPOINT_SHA256" \
  "$CANARY_SUMMARY" <<'PY'
import os
import sys
from pathlib import Path

import yaml

source, destination, folder, checkpoint, checkpoint_sha256, summary = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
config["folder"] = folder
config["checkpoint_folder"] = folder
meta = config["meta"]
meta.update(
    {
        "planner_scale_sweep_only": True,
        "planner_scale_sweep_checkpoint_sha256": checkpoint_sha256,
        "planner_scale_sweep_expected_checkpoint_role": "canary_diagnostic",
        "planner_scale_sweep_expected_optimizer_steps": 2000,
        "planner_scale_sweep_expected_training_complete": False,
        "planner_scale_sweep_parent_summary": str(Path(summary).resolve()),
        "canary_optimizer_steps": 0,
        "load_checkpoint": True,
        "load_opt_scale_epoch": False,
        "read_checkpoint": None,
        "pretrained_path": str(Path(checkpoint).resolve()),
        "plan_only_eval_mode": False,
        "unroll_decode_eval_only_mode": False,
        "light_eval_only_mode": False,
        "skip_planning_eval": True,
        "save_every_freq": -1,
    }
)
config["planner_identified"]["scale_sweep_values"] = [
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    8.0,
]
wandb = config.setdefault("logging", {}).setdefault("wandb", {})
wandb["use_wandb"] = False
wandb["disable_wandb_media"] = True
wandb["log_media_locally"] = False

payload = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
path = Path(destination)
if path.exists():
    if path.read_bytes() != payload:
        raise SystemExit(f"[STOP] immutable scale-sweep config differs: {path}")
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
print(f"[scale-sweep config] {path}")
PY

print_result() {
  python - "$SWEEP_ARTIFACT" "$CHECKPOINT_SHA256" <<'PY'
import json
import math
import sys
from pathlib import Path

artifact = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_sha256 = sys.argv[2]
if artifact.get("status") != "PLANNER_SCALE_SWEEP_ONLY_COMPLETE":
    raise SystemExit("[STOP] extended scale sweep is incomplete")
if artifact.get("optimizer_steps_executed") != 0:
    raise SystemExit("[STOP] scale-sweep-only mode executed optimizer steps")
if artifact.get("model_parameters_updated") is not False:
    raise SystemExit("[STOP] scale-sweep-only mode does not certify frozen parameters")
if artifact.get("source_checkpoint", {}).get("sha256") != expected_sha256:
    raise SystemExit("[STOP] output source checkpoint SHA-256 differs")
sweep = artifact.get("planner_scale_sweep") or {}
if sweep.get("status") != "complete" or sweep.get("learned_parameter_unchanged") is not True:
    raise SystemExit("[STOP] scale intervention audit is incomplete")
records = sweep.get("records") or []
fixed = [record for record in records if record.get("mode") == "fixed"]
if len(fixed) != 13 or not math.isclose(max(row["scale"] for row in fixed), 8.0):
    raise SystemExit("[STOP] extended scale grid is incomplete")

print("mode      scale       loss      sign_acc   response_ratio")
for record in records:
    metrics = record["metrics"]
    print(
        f"{record['mode']:<8} {record['scale']:7.4f} "
        f"{metrics['planner_landscape_loss']:10.6f} "
        f"{metrics['planner_valid_pairwise_sign_accuracy']:10.6f} "
        f"{metrics['planner_predicted_to_real_cost_std_ratio']:14.6f}"
    )
best = sweep["best_observed"]
print(
    "[best observed] "
    f"mode={best['mode']} scale={best['scale']:.6g} "
    f"loss={best['planner_landscape_loss']:.6g}"
)
if math.isclose(best["scale"], max(row["scale"] for row in fixed)):
    print("[diagnostic] best remains at the upper boundary; extend once more before tuning optimization")
PY
}

if test -f "$SWEEP_ARTIFACT"; then
  echo "[reuse complete] $SWEEP_ARTIFACT"
  print_result
  exit 0
fi

mkdir -p "$SWEEP_DIR/logs"
LOG="$SWEEP_DIR/logs/extended_scale_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "[launch checkpoint-only sweep] checkpoint=$CHECKPOINT_PATH config=$SWEEP_CONFIG log=$LOG"
PI_LTC_STOP_AFTER_COMPLETE_PASSES=0 \
  python -m app.main --fname "$SWEEP_CONFIG" --devices cuda:0 --debug 2>&1 | tee "$LOG"
test -f "$SWEEP_ARTIFACT" || fail "scale-sweep artifact is missing"
print_result
echo "[complete] artifact=$SWEEP_ARTIFACT"
