#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from app.vjepa_wm.utils import build_plan_eval_args, clean_state_dict
from src.utils.yaml_utils import convert_to_dict_recursive


ARMS = {
    "learned": ("learned", None),
    "identity": ("fixed", 1.0),
    "fixed06": ("fixed", 0.6),
    "fixed04": ("fixed", 0.4),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_checkpoint_audit(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    predictor = clean_state_dict(checkpoint.get("predictor", {}))
    scale_keys = [key for key in predictor if key.endswith("planner_input_scale.log_scale")]
    if len(scale_keys) != 1:
        raise SystemExit(f"[STOP] checkpoint scale keys={scale_keys}")
    log_scale = float(predictor[scale_keys[0]].detach().cpu())
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != 5:
        raise SystemExit(f"[STOP] checkpoint epoch={epoch}, expected 5")
    if not math.isfinite(log_scale):
        raise SystemExit("[STOP] checkpoint log scale is not finite")
    return epoch, scale_keys[0], log_scale


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="jepa-latest.pth.tar")
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml"),
    )
    parser.add_argument(
        "--eval-template",
        type=Path,
        default=Path("configs/online_plan_evals/mz/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2.yaml"),
    )
    args = parser.parse_args()

    checkpoint_path = (args.checkpoint_dir / args.checkpoint).resolve()
    source_h5 = args.source_h5.resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"[STOP] missing checkpoint: {checkpoint_path}")
    if not source_h5.is_file():
        raise SystemExit(f"[STOP] missing source HDF5: {source_h5}")
    if args.episodes <= 0:
        raise SystemExit("[STOP] episodes must be positive")
    epoch, scale_key, log_scale = load_checkpoint_audit(checkpoint_path)

    training = yaml.safe_load(args.training_config.read_text())
    output_root = args.checkpoint_dir / (
        f"native_pointmaze_cem30_scale_{args.label}_seed{args.eval_seed}_ep{args.episodes}_v3"
    )
    tag = "native_cem30_s300_k10_h6_nas6_ctxt2"
    protocol = {
        "task": "maze-base",
        "goal_source": "random_state",
        "success_definition": "simulator",
        "eval_seed": args.eval_seed,
        "episodes": args.episodes,
        "image_size": 224,
        "context_window": 2,
        "objective": "latent_L2_alpha0.1",
        "cem_iterations": 30,
        "cem_samples": 300,
        "cem_elites": 10,
        "horizon": 6,
        "actions_stepped": 6,
        "decode_each_iteration": False,
    }
    scientific_paths = [
        args.training_config,
        args.eval_template,
        Path("app/plan_common/models/planner_identified_scale.py"),
        Path("app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py"),
        Path("evals/simu_env_planning/eval.py"),
        Path("evals/simu_env_planning/planning/plan_evaluator.py"),
        Path("evals/utils.py"),
        Path("scripts/preflight_pointmaze_native_eval.py"),
        Path(__file__),
    ]
    manifest = {
        "schema_version": 1,
        "protocol": "jepa_wm_pointmaze_same_checkpoint_scale_intervention_v1",
        "repository_commit": repository_commit(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "size": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "epoch": epoch,
            "steps_per_pass": 2278,
            "scale_key": scale_key,
            "log_scale": log_scale,
            "learned_scale": math.exp(log_scale),
        },
        "data_source": {
            "path": str(source_h5),
            "size": source_h5.stat().st_size,
            "mtime_ns": source_h5.stat().st_mtime_ns,
            "purpose": "action/state normalization metadata only",
        },
        "evaluation": protocol,
        "arms": {
            arm: {"mode": mode, "value": value}
            for arm, (mode, value) in ARMS.items()
        },
        "scientific_file_sha256": {str(path): sha256(path) for path in scientific_paths},
    }

    manifest_path = output_root / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise SystemExit("[STOP] existing output provenance differs; use a new label")
    elif output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("[STOP] non-empty output has no matching manifest")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "configs").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    write_json_atomic(manifest_path, manifest)

    for arm, (mode, value) in ARMS.items():
        arm_folder = (output_root / "arms" / arm).resolve()
        _, _, configs, _ = build_plan_eval_args(
            app_name="vjepa_wm",
            folder=str(arm_folder),
            checkpoint=args.checkpoint,
            eval_cfg_paths=[str(args.eval_template)],
            cfgs_model=deepcopy(training["model"]),
            cfgs_data=deepcopy(training["data"]),
            cfgs_data_aug=deepcopy(training["data_aug"]),
            tag=tag,
            evals_decode=False,
            evals_obs="rgb_state",
            evals_alpha=0.1,
            eval_nodes=1,
            eval_tasks_per_node=1,
            eval_episodes=args.episodes,
            wrapper_kwargs={
                "ctxt_window": 2,
                "planner_identified_scale_intervention_mode": mode,
                "planner_identified_scale_intervention_value": value,
            },
            checkpoint_folder=str(args.checkpoint_dir.resolve()),
        )
        config = configs[0]
        config["nodes"] = 1
        config["tasks_per_node"] = 1
        config["meta"]["seed"] = args.eval_seed
        config["meta"]["eval_episodes"] = args.episodes
        config["tag"] = tag
        config["logging"]["exp_name"] = f"jepa_wm_pi_ltc_{arm}"
        config["logging"]["optional_plots"] = False
        config["logging"]["tqdm_silent"] = True
        config["logging"]["save_episode_csv"] = True
        config["planner"]["decode_each_iteration"] = False
        config["task_specification"]["obs"] = "rgb_state"
        config = convert_to_dict_recursive(config)
        serialized = yaml.safe_dump(config, sort_keys=False)
        if yaml.safe_load(serialized) != config:
            raise SystemExit(f"[STOP] generated config failed YAML round-trip validation: arm={arm}")
        config_path = output_root / "configs" / f"{arm}.yaml"
        config_path.write_text(serialized)

    print(f"[generated] {output_root}")
    print(f"[checkpoint] epoch={epoch} log_scale={log_scale:+.6f} scale={math.exp(log_scale):.6f}")
    print(f"[protocol] {protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
