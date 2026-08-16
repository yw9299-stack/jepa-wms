#!/usr/bin/env bash
set -euo pipefail

# Resume the already accepted matched PushT PI-LTC/vanilla checkpoints for one
# more native data pass.  Each arm resumes from its pass-1 latest checkpoint,
# retains the original long-horizon optimizer/scheduler configuration, and is
# stopped only after the atomic pass-2 checkpoint (jepa-e1.pth.tar) is valid.
#
# This file is self-contained so it can be extracted to /tmp, followed by a
# checkout of the original training commit before launch.

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
EXPECTED_TRAINING_COMMIT="${PI_LTC_EXPECTED_TRAINING_COMMIT:-90bb004351a1e4c42f0e99d20601fb69dcd22fdf}"
EXPECTED_PI_PASS1_SHA256="${EXPECTED_PI_PASS1_SHA256:-3933f39514ae6175c8186682b46f5da2430f8c6d96beff0a55fdca70a35b5869}"
EXPECTED_VANILLA_PASS1_SHA256="${EXPECTED_VANILLA_PASS1_SHA256:-2661e14b526a4e5098f5085d59ca664222edfdafafdb8ea15ca2c09dfb04f847}"
EXPECTED_OPTIMIZER_STEPS_PER_PASS=13923
EXPECTED_MICROBATCHES_PER_PASS=111384
EXPECTED_GRADIENT_ACCUMULATION=8
EXPECTED_LONG_HORIZON=111464
EXPECTED_COMPLETED_PASSES=2

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "POLL_SECONDS must be a positive integer"
[[ "$INT_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "INT_GRACE_SECONDS must be a positive integer"
[[ "$TERM_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "TERM_GRACE_SECONDS must be a positive integer"
test -d "$JEPA_REPO" || fail "missing JEPA repository: $JEPA_REPO"
cd "$JEPA_REPO"
test -d .git -o -f .git || fail "not a git worktree: $JEPA_REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_TRAINING_COMMIT" || \
  fail "checkout the original training commit $EXPECTED_TRAINING_COMMIT before launch"
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
PI_LATEST="$PI_DIR/jepa-latest.pth.tar"
VANILLA_LATEST="$VANILLA_DIR/jepa-latest.pth.tar"
PI_PASS2="$PI_DIR/jepa-e1.pth.tar"
VANILLA_PASS2="$VANILLA_DIR/jepa-e1.pth.tar"
PASS1_RELAY_SUMMARY="$JEPAWM_LOGS/pi_ltc_cross_model/pusht_one_pass_relay/relay_summary.json"
RELAY_DIR="$JEPAWM_LOGS/pi_ltc_cross_model/pusht_second_pass_relay"
RELAY_LOG="$RELAY_DIR/relay_$(date +%Y%m%d_%H%M%S).log"
RELAY_SUMMARY="$RELAY_DIR/relay_summary.json"

mkdir -p "$RELAY_DIR"
exec > >(tee -a "$RELAY_LOG") 2>&1

echo "[relay] first variant=PI-LTC config=$PI_CONFIG"
echo "[relay] second variant=vanilla config=$VANILLA_CONFIG"
echo "[relay] resume boundary=pass 1; target boundary=pass 2"
echo "[relay] per-pass=$EXPECTED_OPTIMIZER_STEPS_PER_PASS optimizer updates = $EXPECTED_MICROBATCHES_PER_PASS microbatches"
echo "[relay] pass-2 total=$((EXPECTED_COMPLETED_PASSES * EXPECTED_OPTIMIZER_STEPS_PER_PASS)) optimizer updates"
echo "[relay] original scheduler horizon=$EXPECTED_LONG_HORIZON optimizer updates; accumulation=$EXPECTED_GRADIENT_ACCUMULATION"

for required in \
  "$PI_CONFIG" "$VANILLA_CONFIG" \
  "$PI_PASS1" "$VANILLA_PASS1" \
  "$PI_LATEST" "$VANILLA_LATEST" \
  "$PASS1_RELAY_SUMMARY" \
  "$PI_LTC_PUSHT_SOURCE" "$PI_LTC_PUSHT_SIDECAR"; do
  test -f "$required" || fail "missing required artifact: $required"
done

python - \
  "$PI_CONFIG" "$VANILLA_CONFIG" "$PI_DIR" "$VANILLA_DIR" \
  "$PASS1_RELAY_SUMMARY" "$PI_PASS1" "$VANILLA_PASS1" \
  "$EXPECTED_TRAINING_COMMIT" "$EXPECTED_PI_PASS1_SHA256" \
  "$EXPECTED_VANILLA_PASS1_SHA256" <<'PY'
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import yaml

(
    pi_path,
    vanilla_path,
    pi_dir,
    vanilla_dir,
    pass1_summary_path,
    pi_pass1,
    vanilla_pass1,
    expected_commit,
    expected_pi_sha,
    expected_vanilla_sha,
) = sys.argv[1:]
pi = yaml.safe_load(Path(pi_path).read_text(encoding="utf-8"))
vanilla = yaml.safe_load(Path(vanilla_path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise SystemExit(f"[STOP] config/relay audit: {message}")


require(pi["folder"].endswith(Path(pi_dir).name), "PI folder differs")
require(vanilla["folder"].endswith(Path(vanilla_dir).name), "vanilla folder differs")
require(
    pi["model"]["predictor"].get("planner_identified_input_scale") is True,
    "PI arm does not own the learned scale",
)
require(pi.get("planner_identified", {}).get("enabled") is True, "PI planner loss is disabled")
require(
    vanilla["model"]["predictor"].get("planner_identified_input_scale") is False,
    "vanilla arm unexpectedly owns the learned scale",
)
require(vanilla.get("planner_identified") == {"enabled": False}, "vanilla planner loss is enabled")

for label, config in (("PI", pi), ("vanilla", vanilla)):
    meta = config["meta"]
    optimization = config["optimization"]["transition_model"]
    require(meta.get("load_checkpoint") is True, f"{label} checkpoint loading is disabled")
    require(meta.get("load_opt_scale_epoch") is True, f"{label} optimizer resume is disabled")
    require(meta.get("strict_provenance") is True, f"{label} strict provenance is disabled")
    require(meta.get("save_every_freq") == 1, f"{label} does not save every data pass")
    require(meta.get("skip_planning_eval") is True, f"{label} training would launch planning eval")
    require(config["data"]["loader"]["batch_size"] == 16, f"{label} microbatch is not 16")
    require(optimization["gradient_accumulation_steps"] == 8, f"{label} accumulation is not 8")
    require(optimization["expected_optimizer_steps_per_epoch"] == 13923, f"{label} steps/pass differ")
    require(optimization["total_optimizer_steps"] == 111464, f"{label} scheduler horizon differs")
    require(optimization["num_epochs"] == 9, f"{label} driver epoch count differs")

restored = deepcopy(vanilla)
restored["folder"] = pi["folder"]
restored["model"]["predictor"]["planner_identified_input_scale"] = True
restored["planner_identified"] = deepcopy(pi["planner_identified"])
require(restored == pi, "PI and vanilla configs differ outside the intended PI fields")

relay = json.loads(Path(pass1_summary_path).read_text(encoding="utf-8"))
require(relay.get("status") == "PUSHT_MATCHED_ONE_PASS_COMPLETE", "pass-1 relay is not complete")
require(relay.get("repository_commit") == expected_commit, "pass-1 training commit differs")
require(relay.get("optimizer_steps_per_arm") == 13923, "pass-1 optimizer boundary differs")
require(relay.get("microbatches_per_arm") == 111384, "pass-1 microbatch boundary differs")
require(relay.get("gradient_accumulation_steps") == 8, "pass-1 accumulation differs")
require(relay.get("configured_scheduler_horizon_optimizer_steps") == 111464, "pass-1 scheduler horizon differs")
require(os.path.realpath(relay["pi"]["checkpoint"]) == os.path.realpath(pi_pass1), "pass-1 PI path differs")
require(os.path.realpath(relay["vanilla"]["checkpoint"]) == os.path.realpath(vanilla_pass1), "pass-1 vanilla path differs")
require(relay["pi"]["sha256"] == expected_pi_sha, "pass-1 PI accepted SHA differs")
require(relay["vanilla"]["sha256"] == expected_vanilla_sha, "pass-1 vanilla accepted SHA differs")
print("[config/relay audit PASS] matched configs and accepted pass-1 boundary")
PY

checkpoint_metadata() {
  local checkpoint="$1"
  local owner="$2"
  local completed_passes="$3"
  local expected_sha="${4:-}"
  python - "$checkpoint" "$owner" "$completed_passes" "$expected_sha" <<'PY'
import hashlib
import math
import os
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
owner = sys.argv[2]
completed_passes = int(sys.argv[3])
expected_sha = sys.argv[4]
if not path.is_file():
    raise SystemExit(3)

digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual_sha = digest.hexdigest()
if expected_sha and actual_sha != expected_sha:
    raise SystemExit(
        f"[STOP] {owner} pass-{completed_passes} SHA={actual_sha}, expected {expected_sha}"
    )

checkpoint = torch.load(path, map_location="cpu", weights_only=False)
predictor = checkpoint.get("predictor") or {}
scale_keys = [key for key in predictor if key.endswith("planner_input_scale.log_scale")]
expected_scale_count = 1 if owner == "pi" else 0
if len(scale_keys) != expected_scale_count:
    raise SystemExit(f"[STOP] {owner} checkpoint scale ownership differs: {scale_keys}")
expected = {
    "epoch": completed_passes,
    "optimizer_steps_per_epoch": 13923,
    "gradient_accumulation_steps": 8,
    "total_optimizer_steps": completed_passes * 13923,
    "optimizer_step_in_epoch": 0,
    "optimizer_step_budget": 111464,
    "training_complete": False,
}
for key, value in expected.items():
    if checkpoint.get(key) != value:
        raise SystemExit(
            f"[STOP] {owner} pass-{completed_passes} checkpoint "
            f"{key}={checkpoint.get(key)!r}, expected {value!r}"
        )
optimizer = checkpoint.get("opt")
if not isinstance(optimizer, dict) or not optimizer.get("param_groups") or not optimizer.get("state"):
    raise SystemExit(f"[STOP] {owner} checkpoint has no resumable optimizer state")
provenance = checkpoint.get("training_provenance") or {}
commit = provenance.get("repository_commit")
hash_mode = provenance.get("content_hash_mode")
source_info = provenance.get("source_h5") or {}
sidecar_info = provenance.get("sidecar_h5") or {}
source_sha = source_info.get("sha256")
sidecar_sha = sidecar_info.get("sha256")
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

# This is only a path/stat audit.  It deliberately does not scan either HDF5
# file; strict resume will compare these same provenance fields before loading.
source_path = Path(os.environ["PI_LTC_PUSHT_SOURCE"]).resolve()
source_stat = source_path.stat()
if (
    source_info.get("path") != str(source_path)
    or source_info.get("bytes") != source_stat.st_size
    or source_info.get("mtime_ns") != source_stat.st_mtime_ns
):
    raise SystemExit(f"[STOP] {owner} source path/stat differs from pass-{completed_passes} provenance")
sidecar_path = Path(os.environ["PI_LTC_PUSHT_SIDECAR"]).resolve()
if owner == "pi":
    sidecar_stat = sidecar_path.stat()
    if (
        sidecar_info.get("path") != str(sidecar_path)
        or sidecar_info.get("bytes") != sidecar_stat.st_size
        or sidecar_info.get("mtime_ns") != sidecar_stat.st_mtime_ns
    ):
        raise SystemExit(f"[STOP] PI sidecar path/stat differs from pass-{completed_passes} provenance")
elif any(sidecar_info.get(key) is not None for key in ("path", "bytes", "mtime_ns")):
    raise SystemExit("[STOP] vanilla checkpoint unexpectedly owns a sidecar path/stat")

log_scale = "NA"
scale = "NA"
if owner == "pi":
    raw = predictor[scale_keys[0]]
    log_scale_value = float(raw.detach().cpu().reshape(-1)[0])
    if not math.isfinite(log_scale_value):
        raise SystemExit("[STOP] PI log scale is not finite")
    log_scale = repr(log_scale_value)
    scale = repr(math.exp(log_scale_value))
print(
    "\t".join(
        (
            commit,
            hash_mode,
            source_sha,
            sidecar_sha,
            initialization,
            actual_sha,
            log_scale,
            scale,
        )
    )
)
PY
}

assert_same_resume_state() {
  local latest="$1"
  local accepted="$2"
  local owner="$3"
  python - "$latest" "$accepted" "$owner" <<'PY'
import sys

import torch

latest_path, accepted_path, owner = sys.argv[1:]
latest = torch.load(latest_path, map_location="cpu", weights_only=False)
accepted = torch.load(accepted_path, map_location="cpu", weights_only=False)


def compare(left, right, path="checkpoint"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise SystemExit(f"[STOP] {owner} resume state type differs at {path}")
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise SystemExit(f"[STOP] {owner} resume tensor differs at {path}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            raise SystemExit(f"[STOP] {owner} resume state type differs at {path}")
        if set(left) != set(right):
            raise SystemExit(f"[STOP] {owner} resume keys differ at {path}")
        for key in sorted(left, key=str):
            compare(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise SystemExit(f"[STOP] {owner} resume sequence differs at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise SystemExit(f"[STOP] {owner} resume value differs at {path}")


compare(latest, accepted)
print(f"[resume-state PASS] owner={owner} latest is tensor-exact with accepted pass-1 checkpoint")
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

wait_for_pass2_checkpoint() {
  local checkpoint="$1"
  local owner="$2"
  local pid="$3"
  echo "[wait] owner=$owner pid=$pid target=$checkpoint"
  while [ ! -f "$checkpoint" ]; do
    kill -0 "$pid" 2>/dev/null || fail "$owner process $pid exited before pass-2 checkpoint"
    sleep "$POLL_SECONDS"
  done
  checkpoint_metadata "$checkpoint" "$owner" 2 > "$RELAY_DIR/${owner}_pass2_metadata.tsv.tmp" || \
    fail "$owner pass-2 checkpoint exists but failed validation"
  mv "$RELAY_DIR/${owner}_pass2_metadata.tsv.tmp" "$RELAY_DIR/${owner}_pass2_metadata.tsv"
  echo "[pass-2 checkpoint PASS] owner=$owner path=$checkpoint"
}

stop_at_boundary() {
  local pid="$1"
  local owner="$2"
  local config="$3"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[already stopped] owner=$owner pid=$pid"
    return 0
  fi
  local detected_pid
  detected_pid="$(find_training_pid "$config")"
  test "$detected_pid" = "$pid" || fail "refusing to signal pid=$pid; it no longer owns $config"
  echo "[intentional pass-2 boundary stop] owner=$owner pid=$pid signal=INT"
  kill -INT "$pid"
  for ((second = 0; second < INT_GRACE_SECONDS; second++)); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  echo "[pass-2 boundary stop escalation] owner=$owner pid=$pid signal=TERM"
  kill -TERM "$pid"
  for ((second = 0; second < TERM_GRACE_SECONDS; second++)); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  fail "$owner process $pid did not stop; the next arm will not be launched"
}

run_second_pass() {
  local owner="$1"
  local config="$2"
  local run_dir="$3"
  local accepted_pass1="$4"
  local latest="$5"
  local pass2="$6"
  local accepted_pass1_sha="$7"

  checkpoint_metadata "$accepted_pass1" "$owner" 1 "$accepted_pass1_sha" \
    > "$RELAY_DIR/${owner}_pass1_metadata.tsv"

  local existing_pid
  existing_pid="$(find_training_pid "$config")"
  test -z "$existing_pid" || fail "$owner training process already exists: pid=$existing_pid"

  if [ -f "$pass2" ]; then
    checkpoint_metadata "$pass2" "$owner" 2 > "$RELAY_DIR/${owner}_pass2_metadata.tsv" || \
      fail "$owner existing pass-2 checkpoint failed validation"
    echo "[skip completed arm] owner=$owner checkpoint=$pass2"
    return 0
  fi

  checkpoint_metadata "$latest" "$owner" 1 > "$RELAY_DIR/${owner}_latest_pass1_metadata.tsv" || \
    fail "$owner latest checkpoint is not the pass-1 resume boundary"
  assert_same_resume_state "$latest" "$accepted_pass1" "$owner"
  test ! -e "$pass2.tmp" || fail "$owner has a stale partial checkpoint: $pass2.tmp"

  mkdir -p "$run_dir/logs"
  local run_log="$run_dir/logs/pusht_${owner}_second_pass_relay_$(date +%Y%m%d_%H%M%S).log"
  echo "[launch] owner=$owner resume=$latest target=$pass2 log=$run_log"
  python -m app.main --fname "$config" --devices cuda:0 --debug >> "$run_log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RELAY_DIR/${owner}.pid"
  echo "$run_log" > "$RELAY_DIR/${owner}.log_path"

  wait_for_pass2_checkpoint "$pass2" "$owner" "$pid"
  stop_at_boundary "$pid" "$owner" "$config"
  set +e
  wait "$pid"
  local status=$?
  set -e
  echo "[expected boundary-stop exit] owner=$owner status=$status"
  checkpoint_metadata "$pass2" "$owner" 2 > "$RELAY_DIR/${owner}_pass2_metadata.tsv"
}

# Read the exact accepted provenance markers without touching HDF5 contents.
checkpoint_metadata "$PI_PASS1" pi 1 "$EXPECTED_PI_PASS1_SHA256" > "$RELAY_DIR/pi_pass1_metadata.tsv"
checkpoint_metadata "$VANILLA_PASS1" vanilla 1 "$EXPECTED_VANILLA_PASS1_SHA256" > "$RELAY_DIR/vanilla_pass1_metadata.tsv"
IFS=$'\t' read -r PI_COMMIT PI_HASH_MODE PI_SOURCE_SHA PI_SIDECAR_SHA PI_INITIALIZATION _ _ _ \
  < "$RELAY_DIR/pi_pass1_metadata.tsv"
IFS=$'\t' read -r VANILLA_COMMIT VANILLA_HASH_MODE VANILLA_SOURCE_SHA VANILLA_SIDECAR_SHA VANILLA_INITIALIZATION _ _ _ \
  < "$RELAY_DIR/vanilla_pass1_metadata.tsv"

test "$PI_COMMIT" = "$EXPECTED_TRAINING_COMMIT" || fail "PI pass-1 training commit differs"
test "$VANILLA_COMMIT" = "$EXPECTED_TRAINING_COMMIT" || fail "vanilla pass-1 training commit differs"
test "$VANILLA_HASH_MODE" = "$PI_HASH_MODE" || fail "PI and vanilla hash modes differ"
test "$VANILLA_SOURCE_SHA" = "$PI_SOURCE_SHA" || fail "PI and vanilla source provenance differs"
test "$VANILLA_SIDECAR_SHA" = "$PI_SIDECAR_SHA" || fail "PI and vanilla sidecar provenance differs"
test "$VANILLA_INITIALIZATION" = "$PI_INITIALIZATION" || fail "PI and vanilla common initialization differs"

export PI_LTC_REPOSITORY_COMMIT="$PI_COMMIT"
export PI_LTC_CONTENT_HASH_MODE="$PI_HASH_MODE"
export PI_LTC_SOURCE_SHA256="$PI_SOURCE_SHA"
export PI_LTC_SIDECAR_SHA256="$PI_SIDECAR_SHA"

PI_PID="$(find_training_pid "$PI_CONFIG")"
VANILLA_PID="$(find_training_pid "$VANILLA_CONFIG")"
test -z "$PI_PID" || fail "PI training is already running: pid=$PI_PID"
test -z "$VANILLA_PID" || fail "vanilla training is already running: pid=$VANILLA_PID"

run_second_pass pi "$PI_CONFIG" "$PI_DIR" "$PI_PASS1" "$PI_LATEST" "$PI_PASS2" "$EXPECTED_PI_PASS1_SHA256"
run_second_pass vanilla "$VANILLA_CONFIG" "$VANILLA_DIR" "$VANILLA_PASS1" "$VANILLA_LATEST" "$VANILLA_PASS2" "$EXPECTED_VANILLA_PASS1_SHA256"

IFS=$'\t' read -r PI2_COMMIT PI2_HASH_MODE PI2_SOURCE_SHA PI2_SIDECAR_SHA PI2_INITIALIZATION PI2_SHA PI2_LOG_SCALE PI2_SCALE \
  < "$RELAY_DIR/pi_pass2_metadata.tsv"
IFS=$'\t' read -r VANILLA2_COMMIT VANILLA2_HASH_MODE VANILLA2_SOURCE_SHA VANILLA2_SIDECAR_SHA VANILLA2_INITIALIZATION VANILLA2_SHA _ _ \
  < "$RELAY_DIR/vanilla_pass2_metadata.tsv"
test "$PI2_COMMIT" = "$PI_COMMIT" || fail "PI pass-2 commit differs from pass 1"
test "$VANILLA2_COMMIT" = "$PI_COMMIT" || fail "vanilla pass-2 commit differs from pass 1"
test "$PI2_HASH_MODE" = "$PI_HASH_MODE" || fail "PI pass-2 hash mode differs"
test "$VANILLA2_HASH_MODE" = "$PI_HASH_MODE" || fail "vanilla pass-2 hash mode differs"
test "$PI2_SOURCE_SHA" = "$PI_SOURCE_SHA" || fail "PI pass-2 source provenance differs"
test "$VANILLA2_SOURCE_SHA" = "$PI_SOURCE_SHA" || fail "vanilla pass-2 source provenance differs"
test "$PI2_SIDECAR_SHA" = "$PI_SIDECAR_SHA" || fail "PI pass-2 sidecar provenance differs"
test "$VANILLA2_SIDECAR_SHA" = "$PI_SIDECAR_SHA" || fail "vanilla pass-2 sidecar provenance differs"
test "$PI2_INITIALIZATION" = "$PI_INITIALIZATION" || fail "PI pass-2 initialization differs"
test "$VANILLA2_INITIALIZATION" = "$PI_INITIALIZATION" || fail "vanilla pass-2 initialization differs"

python - \
  "$RELAY_SUMMARY" "$PASS1_RELAY_SUMMARY" "$PI_PASS1" "$VANILLA_PASS1" \
  "$PI_PASS2" "$VANILLA_PASS2" "$PI_COMMIT" "$EXPECTED_PI_PASS1_SHA256" \
  "$EXPECTED_VANILLA_PASS1_SHA256" "$PI2_SHA" "$VANILLA2_SHA" \
  "$PI2_LOG_SCALE" "$PI2_SCALE" "$RELAY_LOG" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    summary_path,
    parent_summary,
    pi_pass1,
    vanilla_pass1,
    pi_pass2,
    vanilla_pass2,
    commit,
    pi_pass1_sha,
    vanilla_pass1_sha,
    pi_pass2_sha,
    vanilla_pass2_sha,
    pi_log_scale,
    pi_scale,
    log_path,
) = sys.argv[1:]
summary = {
    "status": "PUSHT_MATCHED_TWO_PASS_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "repository_commit": commit,
    "budget_policy": "resume both accepted pass-1 arms and stop at native pass-2 checkpoint",
    "selected_completed_passes": 2,
    "optimizer_steps_per_pass": 13923,
    "optimizer_steps_per_arm": 27846,
    "microbatches_per_pass": 111384,
    "microbatches_per_arm": 222768,
    "gradient_accumulation_steps": 8,
    "configured_scheduler_horizon_optimizer_steps": 111464,
    "resume": {
        "parent_relay_summary": os.path.realpath(parent_summary),
        "pi_checkpoint": os.path.realpath(pi_pass1),
        "pi_sha256": pi_pass1_sha,
        "vanilla_checkpoint": os.path.realpath(vanilla_pass1),
        "vanilla_sha256": vanilla_pass1_sha,
        "model_optimizer_scaler_state_restored": True,
        "scheduler_advanced_to_optimizer_step": 13923,
    },
    "pi": {
        "checkpoint": os.path.realpath(pi_pass2),
        "sha256": pi_pass2_sha,
        "final_log_scale": float(pi_log_scale),
        "final_scale": float(pi_scale),
    },
    "vanilla": {
        "checkpoint": os.path.realpath(vanilla_pass2),
        "sha256": vanilla_pass2_sha,
    },
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

echo "[done] PI and vanilla pass-2 checkpoints are complete and provenance-matched"
