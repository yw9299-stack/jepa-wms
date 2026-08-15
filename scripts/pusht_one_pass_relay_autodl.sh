#!/usr/bin/env bash
set -euo pipefail

# Stop the already-running PushT PI-LTC job at its first native pass boundary,
# then run the matched vanilla variant to the same boundary.  This script is
# intentionally self-contained so it can be extracted with `git show` and run
# without checking out a new commit underneath the active training process.

fail() {
  echo "[STOP] $*" >&2
  exit 1
}

JEPA_REPO="${JEPA_REPO:-/root/autodl-tmp/jepa-wms}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-/root/autodl-tmp/lewm_data/jepa_wms_noscan}"
PI_RUN="${PI_RUN:-pusht_jepa_wm_pi_ltc_step111464_v4_seed3072}"
VANILLA_RUN="${VANILLA_RUN:-pusht_jepa_wm_vanilla_step111464_v4_seed3072}"
POLL_SECONDS="${POLL_SECONDS:-2}"
INT_GRACE_SECONDS="${INT_GRACE_SECONDS:-60}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-60}"
EXPECTED_OPTIMIZER_STEPS_PER_PASS=13923
EXPECTED_MICROBATCHES_PER_PASS=111384
EXPECTED_GRADIENT_ACCUMULATION=8
EXPECTED_LONG_HORIZON=111464

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "POLL_SECONDS must be a positive integer"
[[ "$INT_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "INT_GRACE_SECONDS must be a positive integer"
[[ "$TERM_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "TERM_GRACE_SECONDS must be a positive integer"
test -d "$JEPA_REPO" || fail "missing JEPA repository: $JEPA_REPO"
cd "$JEPA_REPO"
test -d .git -o -f .git || fail "not a git worktree: $JEPA_REPO"
git diff --quiet || fail "tracked JEPA worktree is dirty"
git diff --cached --quiet || fail "JEPA index is dirty"

export PYTHONPATH="$JEPA_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export PI_LTC_PUSHT_SOURCE="${PI_LTC_PUSHT_SOURCE:-/root/autodl-tmp/lewm_data/pusht_expert_train.h5}"
export PI_LTC_PUSHT_SIDECAR="${PI_LTC_PUSHT_SIDECAR:-/root/autodl-tmp/lewm_data/pusht_planner_counterfactual_cem0_seed3072.h5}"

PI_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$PI_RUN"
VANILLA_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/$VANILLA_RUN"
PI_CONFIG="$PI_DIR/pi_training_config.yaml"
VANILLA_CONFIG="$VANILLA_DIR/vanilla_training_config.yaml"
PI_PASS1="$PI_DIR/jepa-e0.pth.tar"
VANILLA_PASS1="$VANILLA_DIR/jepa-e0.pth.tar"
RELAY_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/pusht_one_pass_relay"
RELAY_LOG="$RELAY_DIR/relay_$(date +%Y%m%d_%H%M%S).log"
RELAY_SUMMARY="$RELAY_DIR/relay_summary.json"

mkdir -p "$RELAY_DIR"
exec > >(tee -a "$RELAY_LOG") 2>&1

echo "[relay] current variant=PI-LTC config=$PI_CONFIG"
echo "[relay] second variant=vanilla config=$VANILLA_CONFIG"
echo "[relay] accepted boundary=1 PushT pass = $EXPECTED_OPTIMIZER_STEPS_PER_PASS optimizer updates = $EXPECTED_MICROBATCHES_PER_PASS microbatches"
echo "[relay] matched schedule horizon=$EXPECTED_LONG_HORIZON optimizer updates; accumulation=$EXPECTED_GRADIENT_ACCUMULATION"

test -f "$PI_CONFIG" || fail "missing current PI config: $PI_CONFIG"
test -f "$VANILLA_CONFIG" || fail "missing generated vanilla config: $VANILLA_CONFIG"
test -f "$PI_LTC_PUSHT_SOURCE" || fail "missing PushT source: $PI_LTC_PUSHT_SOURCE"
test -f "$PI_LTC_PUSHT_SIDECAR" || fail "missing PushT sidecar: $PI_LTC_PUSHT_SIDECAR"

python - "$PI_CONFIG" "$VANILLA_CONFIG" "$PI_DIR" "$VANILLA_DIR" <<'PY'
import sys
from copy import deepcopy
from pathlib import Path

import yaml

pi_path, vanilla_path, pi_dir, vanilla_dir = sys.argv[1:]
pi = yaml.safe_load(Path(pi_path).read_text(encoding="utf-8"))
vanilla = yaml.safe_load(Path(vanilla_path).read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(f"[STOP] config audit: {message}")

require(pi["folder"].endswith(Path(pi_dir).name), "PI folder differs")
require(vanilla["folder"].endswith(Path(vanilla_dir).name), "vanilla folder differs")
require(pi["model"]["predictor"].get("planner_identified_input_scale") is True, "current arm is not PI-LTC")
require(pi.get("planner_identified", {}).get("enabled") is True, "PI planner loss is not enabled")
require(vanilla["model"]["predictor"].get("planner_identified_input_scale") is False, "second arm is not vanilla")
require(vanilla.get("planner_identified") == {"enabled": False}, "vanilla planner loss is not disabled")

for label, config in (("PI", pi), ("vanilla", vanilla)):
    optimization = config["optimization"]["transition_model"]
    require(config["data"]["loader"]["batch_size"] == 16, f"{label} microbatch is not 16")
    require(optimization["gradient_accumulation_steps"] == 8, f"{label} accumulation is not 8")
    require(optimization["expected_optimizer_steps_per_epoch"] == 13923, f"{label} steps/pass differ")
    require(optimization["total_optimizer_steps"] == 111464, f"{label} long-horizon scheduler differs")

restored = deepcopy(vanilla)
restored["folder"] = pi["folder"]
restored["model"]["predictor"]["planner_identified_input_scale"] = True
restored["planner_identified"] = deepcopy(pi["planner_identified"])
require(restored == pi, "PI and vanilla configs differ outside the intended PI fields")
print("[config audit PASS] matched PI/vanilla; batch=16 accumulation=8 steps/pass=13923")
PY

# Prints checkpoint metadata as tab-separated values after a strict load.  The
# e0 checkpoint is atomically renamed by train.py, so a visible, valid file is
# the completed pass boundary rather than a partially written checkpoint.
checkpoint_metadata() {
  local checkpoint="$1"
  local owner="$2"
  python - "$checkpoint" "$owner" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
owner = sys.argv[2]
if not path.is_file():
    raise SystemExit(3)
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
predictor = checkpoint.get("predictor") or {}
scale_keys = [key for key in predictor if key.endswith("planner_input_scale.log_scale")]
expected_scale_count = 1 if owner == "pi" else 0
if len(scale_keys) != expected_scale_count:
    raise SystemExit(f"[STOP] {owner} checkpoint scale ownership differs: {scale_keys}")
expected = {
    "epoch": 1,
    "optimizer_steps_per_epoch": 13923,
    "gradient_accumulation_steps": 8,
    "total_optimizer_steps": 13923,
    "optimizer_step_in_epoch": 0,
    "optimizer_step_budget": 111464,
    "training_complete": False,
}
for key, value in expected.items():
    if checkpoint.get(key) != value:
        raise SystemExit(
            f"[STOP] {owner} pass-1 checkpoint {key}={checkpoint.get(key)!r}, expected {value!r}"
        )
provenance = checkpoint.get("training_provenance") or {}
commit = provenance.get("repository_commit")
hash_mode = provenance.get("content_hash_mode")
source_sha = (provenance.get("source_h5") or {}).get("sha256")
sidecar_sha = (provenance.get("sidecar_h5") or {}).get("sha256")
initialization = checkpoint.get("common_trainable_initialization_sha256")
if not isinstance(commit, str) or len(commit) != 40:
    raise SystemExit(f"[STOP] invalid repository commit in {owner} checkpoint: {commit!r}")
for label, value in (
    ("source SHA", source_sha),
    ("sidecar SHA", sidecar_sha),
    ("common initialization", initialization),
):
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"[STOP] invalid {label} in {owner} checkpoint: {value!r}")
if not isinstance(hash_mode, str) or not hash_mode:
    raise SystemExit(f"[STOP] invalid content hash mode in {owner} checkpoint")
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
print("\t".join((commit, hash_mode, source_sha, sidecar_sha, initialization, digest.hexdigest())))
PY
}

find_training_pid() {
  local config="$1"
  python - "$config" <<'PY'
import os
import sys

expected = os.path.realpath(sys.argv[1])
matches = []
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    try:
        raw = open(f"/proc/{entry.name}/cmdline", "rb").read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    args = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    if "-m" not in args or "app.main" not in args or "--fname" not in args:
        continue
    try:
        candidate = os.path.realpath(args[args.index("--fname") + 1])
    except (ValueError, IndexError):
        continue
    if candidate == expected:
        matches.append(int(entry.name))
if len(matches) > 1:
    raise SystemExit(f"[STOP] multiple training processes use {expected}: {matches}")
if matches:
    print(matches[0])
PY
}

wait_for_pass_checkpoint() {
  local checkpoint="$1"
  local owner="$2"
  local pid="$3"
  echo "[wait] owner=$owner pid=${pid:-none} checkpoint=$checkpoint"
  while true; do
    set +e
    checkpoint_metadata "$checkpoint" "$owner" > "$RELAY_DIR/${owner}_pass1_metadata.tsv.tmp"
    local checkpoint_rc=$?
    set -e
    if [ "$checkpoint_rc" -eq 0 ]; then
      mv "$RELAY_DIR/${owner}_pass1_metadata.tsv.tmp" "$RELAY_DIR/${owner}_pass1_metadata.tsv"
      echo "[pass-1 checkpoint PASS] owner=$owner path=$checkpoint"
      return 0
    fi
    rm -f "$RELAY_DIR/${owner}_pass1_metadata.tsv.tmp"
    if [ "$checkpoint_rc" -ne 3 ]; then
      fail "$owner checkpoint exists but failed validation"
    fi
    test -n "$pid" || fail "no $owner process and no valid pass-1 checkpoint"
    kill -0 "$pid" 2>/dev/null || fail "$owner process $pid exited before pass-1 checkpoint"
    sleep "$POLL_SECONDS"
  done
}

stop_at_boundary() {
  local pid="$1"
  local owner="$2"
  local config="$3"
  test -n "$pid" || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[already stopped] owner=$owner pid=$pid"
    return 0
  fi
  local detected_pid
  detected_pid="$(find_training_pid "$config")"
  test "$detected_pid" = "$pid" || fail "refusing to signal pid=$pid; it no longer owns $config"
  echo "[intentional pass-boundary stop] owner=$owner pid=$pid signal=INT"
  kill -INT "$pid"
  for ((second = 0; second < INT_GRACE_SECONDS; second++)); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  echo "[pass-boundary stop escalation] owner=$owner pid=$pid signal=TERM"
  kill -TERM "$pid"
  for ((second = 0; second < TERM_GRACE_SECONDS; second++)); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  fail "$owner process $pid did not stop; vanilla will not be launched"
}

PI_PID="${PI_PID:-$(find_training_pid "$PI_CONFIG")}"
if [ -n "$PI_PID" ]; then
  detected_pi_pid="$(find_training_pid "$PI_CONFIG")"
  test "$detected_pi_pid" = "$PI_PID" || fail "PI_PID=$PI_PID does not own $PI_CONFIG"
fi
wait_for_pass_checkpoint "$PI_PASS1" pi "$PI_PID"
stop_at_boundary "$PI_PID" pi "$PI_CONFIG"

IFS=$'\t' read -r PI_COMMIT PI_HASH_MODE PI_SOURCE_SHA PI_SIDECAR_SHA PI_INITIALIZATION PI_CHECKPOINT_SHA < "$RELAY_DIR/pi_pass1_metadata.tsv"
test "$(git rev-parse HEAD)" = "$PI_COMMIT" || fail "JEPA HEAD differs from the PI training commit $PI_COMMIT"
export PI_LTC_REPOSITORY_COMMIT="$PI_COMMIT"
export PI_LTC_CONTENT_HASH_MODE="$PI_HASH_MODE"
export PI_LTC_SOURCE_SHA256="$PI_SOURCE_SHA"
export PI_LTC_SIDECAR_SHA256="$PI_SIDECAR_SHA"

VANILLA_PID="$(find_training_pid "$VANILLA_CONFIG")"
if [ -f "$VANILLA_PASS1" ]; then
  wait_for_pass_checkpoint "$VANILLA_PASS1" vanilla "$VANILLA_PID"
  stop_at_boundary "$VANILLA_PID" vanilla "$VANILLA_CONFIG"
else
  test -z "$VANILLA_PID" || fail "vanilla is already running without a pass-1 checkpoint; refusing a duplicate launch"
  mkdir -p "$VANILLA_DIR/logs"
  VANILLA_LOG="$VANILLA_DIR/logs/pusht_vanilla_one_pass_relay_$(date +%Y%m%d_%H%M%S).log"
  echo "[launch vanilla] config=$VANILLA_CONFIG log=$VANILLA_LOG"
  python -m app.main --fname "$VANILLA_CONFIG" --devices cuda:0 --debug > >(tee -a "$VANILLA_LOG") 2>&1 &
  VANILLA_PID=$!
  echo "$VANILLA_PID" > "$RELAY_DIR/vanilla.pid"
  wait_for_pass_checkpoint "$VANILLA_PASS1" vanilla "$VANILLA_PID"
  stop_at_boundary "$VANILLA_PID" vanilla "$VANILLA_CONFIG"
  set +e
  wait "$VANILLA_PID"
  VANILLA_RC=$?
  set -e
  echo "[expected boundary-stop exit] owner=vanilla status=$VANILLA_RC"
fi

IFS=$'\t' read -r VANILLA_COMMIT VANILLA_HASH_MODE VANILLA_SOURCE_SHA VANILLA_SIDECAR_SHA VANILLA_INITIALIZATION VANILLA_CHECKPOINT_SHA < "$RELAY_DIR/vanilla_pass1_metadata.tsv"
test "$VANILLA_COMMIT" = "$PI_COMMIT" || fail "PI and vanilla repository commits differ"
test "$VANILLA_HASH_MODE" = "$PI_HASH_MODE" || fail "PI and vanilla hash modes differ"
test "$VANILLA_SOURCE_SHA" = "$PI_SOURCE_SHA" || fail "PI and vanilla source provenance differs"
test "$VANILLA_SIDECAR_SHA" = "$PI_SIDECAR_SHA" || fail "PI and vanilla sidecar provenance differs"
test "$VANILLA_INITIALIZATION" = "$PI_INITIALIZATION" || fail "PI and vanilla common initialization differs"

python - "$RELAY_SUMMARY" "$PI_PASS1" "$VANILLA_PASS1" "$PI_COMMIT" "$PI_CHECKPOINT_SHA" "$VANILLA_CHECKPOINT_SHA" "$RELAY_LOG" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path, pi_path, vanilla_path, commit, pi_sha, vanilla_sha, log_path = sys.argv[1:]
summary = {
    "status": "PUSHT_MATCHED_ONE_PASS_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "repository_commit": commit,
    "budget_policy": "stop both long-horizon-matched arms at native pass-1 checkpoint",
    "optimizer_steps_per_arm": 13923,
    "microbatches_per_arm": 111384,
    "gradient_accumulation_steps": 8,
    "configured_scheduler_horizon_optimizer_steps": 111464,
    "pi": {"checkpoint": os.path.realpath(pi_path), "sha256": pi_sha},
    "vanilla": {"checkpoint": os.path.realpath(vanilla_path), "sha256": vanilla_sha},
    "relay_log": os.path.realpath(log_path),
    "intentional_boundary_stop": True,
}
payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
path = Path(summary_path)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(payload, encoding="utf-8")
os.replace(temporary, path)
print(f"[relay complete] summary={path}")
PY

echo "[done] PI and vanilla pass-1 checkpoints are complete and provenance-matched"
