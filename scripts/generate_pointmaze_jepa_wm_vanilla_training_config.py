#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


RUN_NAME = "pointmaze_jepa_wm_vanilla_5pass_seed3072"
EXPECTED_BASE_FOLDER = "${JEPAWM_LOGS}/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass_seed3072"
VANILLA_FOLDER = f"${{JEPAWM_LOGS}}/pi_ltc_cross_model/{RUN_NAME}"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def derive_vanilla_config(base):
    config = deepcopy(base)
    if config.get("folder") != EXPECTED_BASE_FOLDER:
        raise ValueError(f"unexpected base output folder: {config.get('folder')!r}")
    predictor = config.get("model", {}).get("predictor", {})
    if predictor.get("planner_identified_input_scale") is not True:
        raise ValueError("base predictor does not enable planner_identified_input_scale")
    planner = config.get("planner_identified", {})
    if planner.get("enabled") is not True:
        raise ValueError("base config does not enable planner_identified training")
    if int(config["optimization"]["transition_model"]["num_epochs"]) != 5:
        raise ValueError("base config is not the canonical five-pass schedule")

    config["folder"] = VANILLA_FOLDER
    config["model"]["predictor"]["planner_identified_input_scale"] = False
    # Keep the training path genuinely vanilla: no sidecar is opened and no
    # landscape objective or scale-only gradient is constructed.
    config["planner_identified"] = {"enabled": False}
    return config


def write_atomic(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_path = args.base_config.resolve()
    if not base_path.is_file():
        raise SystemExit(f"[STOP] missing base config: {base_path}")
    base = yaml.safe_load(base_path.read_text())
    vanilla = derive_vanilla_config(base)
    serialized = yaml.safe_dump(vanilla, sort_keys=False)
    if yaml.safe_load(serialized) != vanilla:
        raise SystemExit("[STOP] vanilla training config failed YAML round-trip validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "vanilla_training_config.yaml"
    manifest_path = args.output_dir / "vanilla_training_manifest.json"
    manifest = {
        "schema_version": 1,
        "protocol": "pointmaze_jepa_wm_vanilla_matched_5pass_v1",
        "repository_commit": repository_commit(),
        "base_config": {"path": str(args.base_config), "sha256": sha256(base_path)},
        "allowed_scientific_differences": {
            "folder": {"from": EXPECTED_BASE_FOLDER, "to": VANILLA_FOLDER},
            "model.predictor.planner_identified_input_scale": {"from": True, "to": False},
            "planner_identified": {"from": "enabled PI-LTC sidecar objective", "to": {"enabled": False}},
        },
        "schedule": {
            "passes": 5,
            "optimizer_steps_per_pass": 2278,
            "optimizer_steps_total": 11390,
            "micro_batch": 16,
            "gradient_accumulation_steps": 8,
            "effective_batch": 128,
        },
        "generated_config_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise SystemExit("[STOP] existing vanilla training provenance differs; preserve the output directory")
    elif any(args.output_dir.iterdir()):
        unexpected = [path.name for path in args.output_dir.iterdir()]
        raise SystemExit(
            "[STOP] non-empty vanilla output has no matching manifest: " + ", ".join(sorted(unexpected)[:8])
        )

    write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_atomic(config_path, serialized)
    print(f"[generated] {config_path}")
    print("[ablation] same data/model/optimizer/seed/schedule; PI scale and landscape objective disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
