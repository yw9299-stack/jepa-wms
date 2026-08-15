#!/usr/bin/env python3

"""Hard-stop data, pairing, config, and schedule preflight for PushT/Cube."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import hdf5plugin  # noqa: F401 - registers the PushT Blosc filter
import torch
import yaml

from app.plan_common.datasets.planner_landscape_h5 import (
    PlannerLandscapeH5Dataset,
    split_planner_groups_by_physical_episode,
)
from app.plan_common.datasets.stablewm_h5_dset import load_stablewm_h5_train_val
from scripts.generate_stablewm_task_training_configs import (
    BASE_CONFIG,
    TASKS,
    derive_task_configs,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            text=True,
        ).strip()
    )
    return commit, dirty


def _load_config(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"[STOP] missing config: {path}")
    return yaml.safe_load(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--pi-config", type=Path, required=True)
    parser.add_argument("--vanilla-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = TASKS[args.task]
    commit, dirty = _git_state()
    if commit != args.expected_commit:
        raise SystemExit(f"[STOP] repository commit={commit}, expected {args.expected_commit}")
    if dirty:
        raise SystemExit("[STOP] tracked repository files are dirty")
    for label, path, expected_bytes, expected_hash in (
        ("source", args.source, spec["source_bytes"], args.source_sha256),
        ("sidecar", args.sidecar, spec["sidecar_bytes"], args.sidecar_sha256),
    ):
        if not path.is_file():
            raise SystemExit(f"[STOP] missing {label}: {path}")
        if path.stat().st_size != expected_bytes:
            raise SystemExit(
                f"[STOP] {label} bytes={path.stat().st_size}, expected {expected_bytes}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash.lower():
            raise SystemExit(
                f"[STOP] {label} SHA-256={actual_hash}, expected {expected_hash.lower()}"
            )

    expected_pi, expected_vanilla = derive_task_configs(
        yaml.safe_load(BASE_CONFIG.read_text()), args.task
    )
    pi = _load_config(args.pi_config)
    vanilla = _load_config(args.vanilla_config)
    if pi != expected_pi:
        raise SystemExit("[STOP] PI config differs from the task-derived canonical config")
    if vanilla != expected_vanilla:
        raise SystemExit("[STOP] vanilla config differs from the task-derived canonical config")

    custom = pi["data"]["custom"]
    datasets, metadata_by_split, train_episode_ids = load_stablewm_h5_train_val(
        args.source,
        transform=None,
        normalize_action=custom["normalize_action"],
        split_ratio=custom["split_ratio"],
        num_hist=custom["num_hist"],
        num_pred=custom["num_pred"],
        num_frames_val=pi["data"]["validation"]["num_frames_val"],
        frameskip=custom["frameskip"],
        action_skip=custom["action_skip"],
        random_seed=pi["data"]["seed"],
        proprio_keys=custom["proprio_keys"],
        expected_action_dim=custom["expected_action_dim"],
        expected_proprio_dim=custom["expected_proprio_dim"],
        expected_row_count=custom["expected_row_count"],
        expected_episode_count=custom["expected_episode_count"],
        expected_total_clips=custom["expected_total_clips"],
    )
    metadata = metadata_by_split["train"]
    train_clips = len(datasets["train"])
    valid_clips = len(datasets["valid"])
    micro_batch = int(pi["data"]["loader"]["batch_size"])
    accumulation = int(
        pi["optimization"]["transition_model"]["gradient_accumulation_steps"]
    )
    loader_microbatches = train_clips // micro_batch
    used_microbatches = loader_microbatches // accumulation * accumulation
    steps_per_pass = used_microbatches // accumulation
    if steps_per_pass < 1:
        raise SystemExit("[STOP] no complete optimizer step")

    sample_obs, sample_action, sample_state, _ = datasets["train"][0]
    expected_action_width = spec["action_dim"] * int(custom["frameskip"])
    if tuple(sample_obs["visual"].shape) != (4, 3, 224, 224):
        raise SystemExit(f"[STOP] visual sample shape={tuple(sample_obs['visual'].shape)}")
    if tuple(sample_obs["proprio"].shape) != (4, spec["proprio_dim"]):
        raise SystemExit(f"[STOP] proprio sample shape={tuple(sample_obs['proprio'].shape)}")
    if tuple(sample_action.shape) != (4, expected_action_width):
        raise SystemExit(f"[STOP] action sample shape={tuple(sample_action.shape)}")
    if tuple(sample_state.shape) != (4, spec["proprio_dim"]):
        raise SystemExit(f"[STOP] state sample shape={tuple(sample_state.shape)}")

    planner_cfg = pi["planner_identified"]
    planner = PlannerLandscapeH5Dataset(
        source_h5=args.source,
        sidecar_h5=args.sidecar,
        transform=None,
        action_mean=metadata.action_mean,
        action_std=metadata.action_std,
        proprio_mean=metadata.proprio_mean,
        proprio_std=metadata.proprio_std,
        source_metadata=metadata,
        expected_protocol=planner_cfg["expected_protocol"],
        expected_groups=planner_cfg["expected_groups"],
        expected_branches_per_group=planner_cfg["expected_branches_per_group"],
        expected_config_sha256=planner_cfg["expected_config_sha256"],
        frameskip=custom["frameskip"],
        goal_offset_steps=planner_cfg["goal_offset_steps"],
    )
    planner_train, planner_valid = split_planner_groups_by_physical_episode(
        planner, train_episode_ids
    )
    planner_sample = planner[0]
    if tuple(planner_sample["action"].shape) != (4, 1, expected_action_width):
        raise SystemExit(f"[STOP] planner action shape={tuple(planner_sample['action'].shape)}")
    if tuple(planner_sample["context_proprio"].shape) != (1, spec["proprio_dim"]):
        raise SystemExit(
            f"[STOP] planner proprio shape={tuple(planner_sample['context_proprio'].shape)}"
        )

    result = {
        "status": "PASS",
        "task": args.task,
        "repository_commit": commit,
        "training_configs": {
            "pi": {
                "path": str(args.pi_config.resolve()),
                "sha256": sha256_file(args.pi_config),
            },
            "vanilla": {
                "path": str(args.vanilla_config.resolve()),
                "sha256": sha256_file(args.vanilla_config),
            },
        },
        "source": {
            "path": str(args.source.resolve()),
            "bytes": args.source.stat().st_size,
            "mtime_ns": args.source.stat().st_mtime_ns,
            "sha256": args.source_sha256.lower(),
            "metadata_layout": metadata.metadata_layout,
            "rows": metadata.row_count,
            "episodes": len(metadata),
            "action_dim": metadata.action_dim,
            "proprio_dim": metadata.proprio_dim,
            "proprio_keys": list(metadata.proprio_keys),
        },
        "split": {
            "train_episodes": len(train_episode_ids),
            "train_clips": train_clips,
            "valid_clips": valid_clips,
            "loader_microbatches": loader_microbatches,
            "used_microbatches": used_microbatches,
            "dropped_microbatches": loader_microbatches - used_microbatches,
            "optimizer_steps_per_pass": steps_per_pass,
            "optimizer_steps_five_passes": steps_per_pass * 5,
        },
        "planner": {
            "sidecar_path": str(args.sidecar.resolve()),
            "sidecar_bytes": args.sidecar.stat().st_size,
            "sidecar_mtime_ns": args.sidecar.stat().st_mtime_ns,
            "groups": len(planner),
            "train_groups": len(planner_train),
            "valid_groups": len(planner_valid),
            "branches_per_group": planner.branches_per_group,
            "sidecar_sha256": args.sidecar_sha256.lower(),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
