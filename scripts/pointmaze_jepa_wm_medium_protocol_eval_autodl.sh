#!/usr/bin/env bash
# Evaluate JEPA-WM PI-LTC and matched vanilla with LEWM's exact OGBench
# Temporal PointMaze-Medium environment, starts, CEM, and success protocol.

set -u
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/lewm_data}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
LEWM_REPO="${LEWM_REPO:-/root/autodl-tmp/lewm}"
SOURCE_H5="${PI_LTC_POINTMAZE_SOURCE:-$STABLEWM_HOME/ogbench/temporal_pointmaze_medium_topdown.h5}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml}"
PI_DIR="${PI_DIR:-$STABLEWM_HOME/jepa_wms/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass_seed3072}"
VANILLA_DIR="${VANILLA_DIR:-$STABLEWM_HOME/jepa_wms/pi_ltc_cross_model/pointmaze_jepa_wm_vanilla_5pass_seed3072}"
CHECKPOINT="${CHECKPOINT:-jepa-latest.pth.tar}"
EVAL_SEED="${EVAL_SEED:-42}"
EPISODES="${EPISODES:-50}"
OUT_ROOT="${OUT_ROOT:-$STABLEWM_HOME/jepa_wms/pi_ltc_cross_model/pointmaze_medium_stablewm_protocol_cem16_seed${EVAL_SEED}}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-100000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260802}"

for required in \
    "$PYTHON_BIN" \
    "$SOURCE_H5" \
    "$TRAINING_CONFIG" \
    "$PI_DIR/$CHECKPOINT" \
    "$VANILLA_DIR/$CHECKPOINT" \
    "$LEWM_REPO/eval_pointmaze_topdown.py" \
    "$LEWM_REPO/config/eval/pointmaze_topdown_medium.yaml"; do
    if [ ! -e "$required" ]; then
        echo "[STOP] missing required input: $required"
        echo "[terminal remains open]"
        return 1 2>/dev/null || exit 1
    fi
done

if ! "$PYTHON_BIN" scripts/eval_pointmaze_medium_stablewm_protocol.py --help >/dev/null; then
    echo "[STOP] Medium-protocol evaluator import preflight failed"
    echo "[terminal remains open]"
    return 1 2>/dev/null || exit 1
fi

mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" - \
    "$OUT_ROOT" \
    "$SOURCE_H5" \
    "$PI_DIR/$CHECKPOINT" \
    "$VANILLA_DIR/$CHECKPOINT" \
    "$TRAINING_CONFIG" \
    "$LEWM_REPO" \
    "$EVAL_SEED" \
    "$EPISODES" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

(
    root_text,
    source_text,
    pi_text,
    vanilla_text,
    config_text,
    lewm_text,
    seed_text,
    episodes_text,
) = sys.argv[1:]
root = Path(root_text).resolve()
source = Path(source_text).resolve()
pi = Path(pi_text).resolve()
vanilla = Path(vanilla_text).resolve()
config = Path(config_text).resolve()
lewm = Path(lewm_text).resolve()

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()

evaluation = {
    "task": "Temporal PointMaze-Medium",
    "eval_seed": int(seed_text),
    "episodes": int(episodes_text),
    "action_noise_std": 0.0,
    "eval_budget": 50,
    "goal_offset_steps": 25,
    "sample_remaining_min": 25,
    "sample_remaining_max": 50,
    "history_size": 1,
    "horizon": 6,
    "receding_horizon": 6,
    "action_block": 5,
    "cem_steps": 16,
    "cem_samples": 256,
    "cem_topk": 64,
    "maze_spec": "medium",
    "coordinate_offset": [1.2, 1.2],
    "render_transform": "flip_ud",
    "success_threshold": 0.5,
    "cost": "visual latent squared L2 endpoint",
}
manifest = {
    "schema_version": 1,
    "protocol": "ogbench_temporal_pointmaze_medium_stablewm_exact_v1",
    "training": False,
    "repository_commit": commit(Path.cwd()),
    "lewm_repository_commit": commit(lewm),
    "source_h5": {
        "path": str(source),
        "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
    },
    "training_config": {
        "path": str(config),
        "sha256": sha256(config),
    },
    "checkpoints": {
        "pi": {"path": str(pi), "size": pi.stat().st_size, "sha256": sha256(pi)},
        "vanilla": {
            "path": str(vanilla),
            "size": vanilla.stat().st_size,
            "sha256": sha256(vanilla),
        },
    },
    "evaluation": evaluation,
    "arms": {
        "learned": {"owner": "pi", "mode": "learned", "value": None},
        "identity": {"owner": "pi", "mode": "fixed", "value": 1.0},
        "fixed06": {"owner": "pi", "mode": "fixed", "value": 0.6},
        "fixed04": {"owner": "pi", "mode": "fixed", "value": 0.4},
        "vanilla_5pass": {"owner": "vanilla", "mode": None, "value": None},
    },
    "scientific_files": {
        "adapter": sha256(Path("app/vjepa_wm/modelcustom/stablewm_pointmaze_cost.py")),
        "runner": sha256(Path("scripts/eval_pointmaze_medium_stablewm_protocol.py")),
        "lewm_evaluator": sha256(lewm / "eval_pointmaze_topdown.py"),
        "lewm_eval_config": sha256(lewm / "config/eval/pointmaze_topdown_medium.yaml"),
    },
}
path = root / "run_manifest.json"
if path.exists():
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != manifest:
        raise SystemExit(
            "[STOP] existing Medium-protocol output has different provenance; "
            "choose a deliberate OUT_ROOT instead of mixing lineages"
        )
else:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
print(f"[provenance] JEPA={manifest['repository_commit']} LEWM={manifest['lewm_repository_commit']}")
print(f"[protocol] {evaluation}")
PY
MANIFEST_STATUS=$?
if [ "$MANIFEST_STATUS" -ne 0 ]; then
    echo "[terminal remains open]"
    return "$MANIFEST_STATUS" 2>/dev/null || exit "$MANIFEST_STATUS"
fi

echo "================================================================"
echo "[JEPA-WM on exact OGBench Temporal PointMaze-Medium protocol]"
echo "training=false"
echo "source=$SOURCE_H5"
echo "PI=$PI_DIR/$CHECKPOINT"
echo "vanilla=$VANILLA_DIR/$CHECKPOINT"
echo "protocol: seed=$EVAL_SEED episodes=$EPISODES H6/R6/block5 CEM16/256/64"
echo "arms: learned identity fixed06 fixed04 vanilla_5pass"
echo "output=$OUT_ROOT"
echo "================================================================"

validate_completed_arm() {
    local arm="$1"
    local owner="$2"
    local checkpoint_path="$3"
    "$PYTHON_BIN" - "$OUT_ROOT" "$arm" "$owner" "$checkpoint_path" "$SOURCE_H5" "$EVAL_SEED" "$EPISODES" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root_text, arm, owner, checkpoint_text, source_text, seed_text, episodes_text = sys.argv[1:]
audit_path = Path(root_text) / arm / "arm_audit.json"
result_path = Path(root_text) / arm / "result.txt"
if not audit_path.is_file() or not result_path.is_file():
    raise SystemExit(1)
audit = json.loads(audit_path.read_text(encoding="utf-8"))
checkpoint = Path(checkpoint_text).resolve()
source = Path(source_text).resolve()

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

valid = (
    audit.get("status") == "complete"
    and audit.get("training") is False
    and audit.get("arm") == arm
    and audit.get("owner") == owner
    and audit.get("protocol") == "ogbench_temporal_pointmaze_medium_stablewm_exact_v1"
    and audit.get("checkpoint", {}).get("path") == str(checkpoint)
    and audit.get("checkpoint", {}).get("sha256") == sha256(checkpoint)
    and audit.get("source_h5", {}).get("path") == str(source)
    and audit.get("source_h5", {}).get("size") == source.stat().st_size
    and audit.get("evaluation", {}).get("eval_seed") == int(seed_text)
    and audit.get("evaluation", {}).get("episodes") == int(episodes_text)
    and len(audit.get("results", {}).get("episode_successes", [])) == int(episodes_text)
)
raise SystemExit(0 if valid else 1)
PY
}

run_arm() {
    local arm="$1"
    local owner="$2"
    local checkpoint_dir="$3"
    local checkpoint_path="$checkpoint_dir/$CHECKPOINT"
    local arm_root="$OUT_ROOT/$arm"
    local arm_log="$arm_root/run.log"

    if validate_completed_arm "$arm" "$owner" "$checkpoint_path"; then
        echo "[skip strictly validated arm] $arm"
        return 0
    fi
    mkdir -p "$arm_root"
    : > "$arm_log"
    echo "================================================================"
    echo "[arm] $arm owner=$owner"
    echo "================================================================"
    "$PYTHON_BIN" scripts/eval_pointmaze_medium_stablewm_protocol.py \
        --lewm-repo "$LEWM_REPO" \
        --source-h5 "$SOURCE_H5" \
        --training-config "$TRAINING_CONFIG" \
        --checkpoint-dir "$checkpoint_dir" \
        --checkpoint "$CHECKPOINT" \
        --owner "$owner" \
        --arm "$arm" \
        --output-root "$OUT_ROOT" \
        --eval-seed "$EVAL_SEED" \
        --episodes "$EPISODES" \
        2>&1 | tee "$arm_log"
    local status="${PIPESTATUS[0]}"
    echo "[arm returned] arm=$arm status=$status"
    return "$status"
}

FAILED=0
if ! run_arm learned pi "$PI_DIR"; then
    FAILED=1
elif ! run_arm identity pi "$PI_DIR"; then
    FAILED=1
elif ! run_arm fixed06 pi "$PI_DIR"; then
    FAILED=1
elif ! run_arm fixed04 pi "$PI_DIR"; then
    FAILED=1
elif ! run_arm vanilla_5pass vanilla "$VANILLA_DIR"; then
    FAILED=1
fi

SUMMARY_STATUS=1
if [ "$FAILED" -eq 0 ]; then
    "$PYTHON_BIN" scripts/summarize_pointmaze_medium_stablewm_protocol.py \
        --root "$OUT_ROOT" \
        --eval-seed "$EVAL_SEED" \
        --episodes "$EPISODES" \
        --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
        --bootstrap-seed "$BOOTSTRAP_SEED"
    SUMMARY_STATUS=$?
else
    echo "[summary skipped] failed_arms=$FAILED"
fi

FINAL_STATUS=0
if [ "$FAILED" -ne 0 ] || [ "$SUMMARY_STATUS" -ne 0 ]; then
    FINAL_STATUS=1
fi
echo "================================================================"
echo "[JEPA-WM Medium-protocol launcher returned] status=$FINAL_STATUS"
echo "[failed arms] $FAILED"
echo "[summary status] $SUMMARY_STATUS"
echo "[output] $OUT_ROOT"
echo "[terminal remains open]"
echo "================================================================"
return "$FINAL_STATUS" 2>/dev/null || exit "$FINAL_STATUS"
