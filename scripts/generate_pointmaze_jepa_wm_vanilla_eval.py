#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from app.vjepa_wm.utils import build_plan_eval_args, clean_state_dict
from src.utils.yaml_utils import convert_to_dict_recursive


TAG = "native_cem30_s300_k10_h6_nas6_ctxt2"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_atomic(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def load_checkpoint_audit(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    predictor = clean_state_dict(checkpoint.get("predictor", {}))
    scale_keys = [key for key in predictor if key.endswith("planner_input_scale.log_scale")]
    if scale_keys:
        raise SystemExit(f"[STOP] vanilla checkpoint contains PI scale keys: {scale_keys}")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != 5:
        raise SystemExit(f"[STOP] vanilla checkpoint epoch={epoch}, expected 5")
    if checkpoint.get("planner_identified") not in (None, {"enabled": False}):
        raise SystemExit("[STOP] checkpoint declares an enabled/non-vanilla planner-identification objective")
    return epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="jepa-latest.pth.tar")
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument(
        "--eval-template",
        type=Path,
        default=Path("configs/online_plan_evals/mz/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2.yaml"),
    )
    args = parser.parse_args()

    checkpoint_path = (args.checkpoint_dir / args.checkpoint).resolve()
    source_h5 = args.source_h5.resolve()
    training_config = args.training_config.resolve()
    for description, path in (
        ("checkpoint", checkpoint_path),
        ("source HDF5", source_h5),
        ("training config", training_config),
        ("evaluation template", args.eval_template),
    ):
        if not path.is_file():
            raise SystemExit(f"[STOP] missing {description}: {path}")
    if args.episodes <= 0:
        raise SystemExit("[STOP] episodes must be positive")

    epoch = load_checkpoint_audit(checkpoint_path)
    training = yaml.safe_load(training_config.read_text())
    if training["model"]["predictor"].get("planner_identified_input_scale") is not False:
        raise SystemExit("[STOP] vanilla evaluation config still enables the PI scale module")
    if training.get("planner_identified") != {"enabled": False}:
        raise SystemExit("[STOP] vanilla evaluation config still enables the landscape objective")
    frameskip = int(training["data"]["custom"]["frameskip"])

    output_root = args.checkpoint_dir / (
        f"native_pointmaze_cem30_{args.label}_seed{args.eval_seed}_ep{args.episodes}"
    )
    protocol = {
        "task": "maze-base",
        "goal_source": "random_state",
        "success_definition": "simulator",
        "eval_seed": args.eval_seed,
        "episodes": args.episodes,
        "image_size": 224,
        "context_window": 2,
        "frameskip": frameskip,
        "objective": "latent_L2_alpha0.1",
        "cem_iterations": 30,
        "cem_samples": 300,
        "cem_elites": 10,
        "horizon": 6,
        "actions_stepped": 6,
        "decode_each_iteration": False,
    }
    scientific_paths = [
        training_config,
        args.eval_template,
        Path("app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py"),
        Path("evals/simu_env_planning/eval.py"),
        Path("evals/simu_env_planning/planning/plan_evaluator.py"),
        Path("evals/utils.py"),
        Path("scripts/preflight_pointmaze_native_eval.py"),
        Path(__file__),
    ]
    manifest = {
        "schema_version": 1,
        "protocol": "jepa_wm_pointmaze_vanilla_matched_5pass_eval_v1",
        "repository_commit": repository_commit(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "size": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "epoch": epoch,
            "steps_per_pass": 2278,
            "planner_identified_scale": False,
        },
        "data_source": {
            "path": str(source_h5),
            "size": source_h5.stat().st_size,
            "mtime_ns": source_h5.stat().st_mtime_ns,
            "purpose": "action/state normalization metadata only",
        },
        "evaluation": protocol,
        "arm": "vanilla",
        "scientific_file_sha256": {str(path): sha256(path) for path in scientific_paths},
    }

    manifest_path = output_root / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit("[STOP] existing vanilla eval provenance differs; preserve the output directory")
    elif output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("[STOP] non-empty vanilla eval output has no matching manifest")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "configs").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    arm_folder = (output_root / "arms" / "vanilla").resolve()
    _, _, configs, _ = build_plan_eval_args(
        app_name="vjepa_wm",
        folder=str(arm_folder),
        checkpoint=args.checkpoint,
        eval_cfg_paths=[str(args.eval_template)],
        cfgs_model=deepcopy(training["model"]),
        cfgs_data=deepcopy(training["data"]),
        cfgs_data_aug=deepcopy(training["data_aug"]),
        tag=TAG,
        evals_decode=False,
        evals_obs="rgb_state",
        evals_alpha=0.1,
        eval_nodes=1,
        eval_tasks_per_node=1,
        eval_episodes=args.episodes,
        wrapper_kwargs={"ctxt_window": 2},
        checkpoint_folder=str(args.checkpoint_dir.resolve()),
    )
    config = configs[0]
    config["nodes"] = 1
    config["tasks_per_node"] = 1
    config["frameskip"] = frameskip
    config["meta"]["seed"] = args.eval_seed
    config["meta"]["eval_episodes"] = args.episodes
    config["tag"] = TAG
    config["logging"]["exp_name"] = "jepa_wm_vanilla_5pass"
    config["logging"]["optional_plots"] = False
    config["logging"]["tqdm_silent"] = True
    config["logging"]["save_episode_csv"] = True
    config["planner"]["decode_each_iteration"] = False
    config["task_specification"]["obs"] = "rgb_state"
    config = convert_to_dict_recursive(config)
    serialized = yaml.safe_dump(config, sort_keys=False)
    if yaml.safe_load(serialized) != config:
        raise SystemExit("[STOP] generated vanilla eval config failed YAML round-trip validation")
    write_atomic(output_root / "configs" / "vanilla.yaml", serialized)

    print(f"[generated] {output_root}")
    print(f"[checkpoint] epoch={epoch} planner_identified_scale=false")
    print(f"[protocol] {protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
