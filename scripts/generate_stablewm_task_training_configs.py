#!/usr/bin/env python3

"""Generate pinned PushT/Cube PI-LTC and matched-vanilla configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


BASE_CONFIG = Path("configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml")

TASKS = {
    "pusht": {
        "pi_run": "pusht_jepa_wm_pi_ltc_5pass_seed3072",
        "vanilla_run": "pusht_jepa_wm_vanilla_5pass_seed3072",
        "source_env": "PI_LTC_PUSHT_SOURCE",
        "sidecar_env": "PI_LTC_PUSHT_SIDECAR",
        "proprio_keys": ["proprio"],
        "action_dim": 2,
        "proprio_dim": 4,
        "rows": 2336736,
        "episodes": 18685,
        "total_clips": 1981721,
        "source_bytes": 46300921856,
        "sidecar_bytes": 192116437,
        "protocol": "pusht_planner_counterfactual_v1",
        "sidecar_config_sha256": "35b55bbd2cdfde01ae133fe86254dc720116885c69e31165bf3abdb08fc5659b",
    },
    "cube": {
        "pi_run": "cube_jepa_wm_pi_ltc_5pass_seed3072",
        "vanilla_run": "cube_jepa_wm_vanilla_5pass_seed3072",
        "source_env": "PI_LTC_CUBE_SOURCE",
        "sidecar_env": "PI_LTC_CUBE_SIDECAR",
        "proprio_keys": [
            "proprio_effector_pos",
            "proprio_effector_yaw",
            "proprio_gripper_contact",
            "proprio_gripper_opening",
            "proprio_gripper_vel",
            "proprio_joint_pos",
            "proprio_joint_vel",
        ],
        "action_dim": 5,
        "proprio_dim": 19,
        "rows": 2010000,
        "episodes": 10000,
        "total_clips": 1820000,
        "source_bytes": 101942558720,
        "sidecar_bytes": 1282198884,
        "protocol": "cube_planner_counterfactual_v1",
        "sidecar_config_sha256": "d62e7a70e6a65a5560e0556c7483fc2d075baa7fb01f4b4c70c6433b471bc143",
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            text=True,
        ).strip()
    )
    return commit, dirty


def derive_task_configs(base: dict, task: str) -> tuple[dict, dict]:
    spec = TASKS[task]
    pi = deepcopy(base)
    if pi["model"]["predictor"].get("planner_identified_input_scale") is not True:
        raise ValueError("base config does not enable planner-identified input scale")
    if pi["planner_identified"].get("enabled") is not True:
        raise ValueError("base config does not enable the planner objective")
    optimization = pi["optimization"]["transition_model"]
    if (
        int(optimization["num_epochs"]) != 5
        or int(optimization["gradient_accumulation_steps"]) != 8
        or int(pi["data"]["loader"]["batch_size"]) != 16
    ):
        raise ValueError("base config is not the canonical five-pass/effective-128 schedule")

    pi["folder"] = f"${{JEPAWM_LOGS}}/pi_ltc_cross_model/{spec['pi_run']}"
    pi["data"]["paths"] = [f"${{{spec['source_env']}}}"]
    custom = pi["data"]["custom"]
    custom.update(
        {
            "proprio_keys": list(spec["proprio_keys"]),
            "expected_action_dim": spec["action_dim"],
            "expected_proprio_dim": spec["proprio_dim"],
            "expected_row_count": spec["rows"],
            "expected_episode_count": spec["episodes"],
            "expected_total_clips": spec["total_clips"],
        }
    )
    pi["meta"]["light_eval_freq"] = 1000000000
    pi["meta"]["strict_provenance"] = True
    planner = pi["planner_identified"]
    planner.update(
        {
            "sidecar_h5": f"${{{spec['sidecar_env']}}}",
            "expected_protocol": spec["protocol"],
            "expected_groups": 10000,
            "expected_branches_per_group": 4,
            "expected_config_sha256": spec["sidecar_config_sha256"],
        }
    )

    vanilla = deepcopy(pi)
    vanilla["folder"] = f"${{JEPAWM_LOGS}}/pi_ltc_cross_model/{spec['vanilla_run']}"
    vanilla["model"]["predictor"]["planner_identified_input_scale"] = False
    vanilla["planner_identified"] = {"enabled": False}
    return pi, vanilla


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"[STOP] immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    args = parser.parse_args()

    for label, value in (
        ("source", args.source_sha256),
        ("sidecar", args.sidecar_sha256),
    ):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise SystemExit(f"[STOP] invalid {label} SHA-256: {value!r}")

    base_path = args.base_config.resolve()
    base = yaml.safe_load(base_path.read_text())
    pi, vanilla = derive_task_configs(base, args.task)
    pi_payload = yaml.safe_dump(pi, sort_keys=False).encode()
    vanilla_payload = yaml.safe_dump(vanilla, sort_keys=False).encode()
    if yaml.safe_load(pi_payload) != pi or yaml.safe_load(vanilla_payload) != vanilla:
        raise SystemExit("[STOP] generated config failed YAML round trip")

    commit, dirty = repository_state()
    if dirty:
        raise SystemExit("[STOP] tracked repository files are dirty")
    spec = TASKS[args.task]
    manifest = {
        "schema_version": 1,
        "protocol": f"{args.task}_jepa_wm_pi_ltc_vs_vanilla_5pass_v1",
        "repository_commit": commit,
        "repository_tracked_dirty": dirty,
        "content_hash_mode": os.environ.get("PI_LTC_CONTENT_HASH_MODE", "sha256"),
        "base_config": {"path": str(args.base_config), "sha256": sha256_file(base_path)},
        "source_sha256": args.source_sha256.lower(),
        "sidecar_sha256": args.sidecar_sha256.lower(),
        "task_spec": spec,
        "pi_config_sha256": sha256_bytes(pi_payload),
        "vanilla_config_sha256": sha256_bytes(vanilla_payload),
        "matched_ablation": {
            "folder": [pi["folder"], vanilla["folder"]],
            "model.predictor.planner_identified_input_scale": [True, False],
            "planner_identified": ["task sidecar objective", {"enabled": False}],
        },
        "schedule": {
            "passes": 5,
            "micro_batch": 16,
            "gradient_accumulation_steps": 8,
            "effective_batch": 128,
            "optimizer_steps_per_pass": "derived from the episode-level train split",
        },
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    for arm, config, payload in (
        ("pi", pi, pi_payload),
        ("vanilla", vanilla, vanilla_payload),
    ):
        output = args.output_root / "pi_ltc_cross_model" / spec[f"{arm}_run"]
        _write_immutable(output / f"{arm}_training_config.yaml", payload)
        _write_immutable(output / "training_manifest.json", manifest_payload)
        print(f"[{arm}] config={output / f'{arm}_training_config.yaml'} folder={config['folder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
