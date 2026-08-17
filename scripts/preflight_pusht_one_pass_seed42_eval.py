#!/usr/bin/env python3
"""Build a no-dataset-scan audit for matched PushT pass-boundary checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from scripts.stablewm_checkpoint_selection import expected_checkpoint_schedule


OPTIMIZER_STEPS_PER_PASS = 13923
MICROBATCHES_PER_PASS = 111384
GRADIENT_ACCUMULATION = 8
CONFIGURED_OPTIMIZER_HORIZON = 111464


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checkpoint_audit(
    path: Path,
    owner: str,
    expected_sha256: str,
    source_h5: Path,
    sidecar_h5: Path,
    expected_completed_passes: int,
) -> dict:
    actual_sha256 = sha256(path)
    require(actual_sha256 == expected_sha256, f"{owner} checkpoint SHA-256 differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    scale_keys = [key for key in (checkpoint.get("predictor") or {}) if key.endswith("planner_input_scale.log_scale")]
    require(len(scale_keys) == (1 if owner == "pi" else 0), f"{owner} scale ownership differs")
    expected_schedule = expected_checkpoint_schedule(
        optimizer_steps_per_pass=OPTIMIZER_STEPS_PER_PASS,
        configured_optimizer_step_budget=CONFIGURED_OPTIMIZER_HORIZON,
        expected_completed_passes=expected_completed_passes,
    )
    expected_fields = {
        "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_PASS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "optimizer_step_budget": CONFIGURED_OPTIMIZER_HORIZON,
        "epoch": expected_schedule["epoch"],
        "total_optimizer_steps": expected_schedule["total_optimizer_steps"],
        "optimizer_step_in_epoch": expected_schedule["optimizer_step_in_epoch"],
        "training_complete": expected_schedule["training_complete"],
    }
    for key, expected in expected_fields.items():
        require(
            checkpoint.get(key) == expected,
            f"{owner} checkpoint {key}={checkpoint.get(key)!r}, expected {expected!r}",
        )
    provenance = checkpoint.get("training_provenance") or {}
    source = provenance.get("source_h5") or {}
    sidecar = provenance.get("sidecar_h5") or {}
    source_stat = source_h5.stat()
    sidecar_stat = sidecar_h5.stat()
    require(source.get("path") == str(source_h5), f"{owner} source path differs")
    require(source.get("bytes") == source_stat.st_size, f"{owner} source byte size differs")
    require(source.get("mtime_ns") == source_stat.st_mtime_ns, f"{owner} source mtime differs")
    require(isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64, f"{owner} source marker is invalid")
    require(
        isinstance(sidecar.get("sha256"), str) and len(sidecar["sha256"]) == 64,
        f"{owner} sidecar marker is invalid",
    )
    if owner == "pi":
        require(sidecar.get("path") == str(sidecar_h5), "PI sidecar path differs")
        require(sidecar.get("bytes") == sidecar_stat.st_size, "PI sidecar byte size differs")
        require(sidecar.get("mtime_ns") == sidecar_stat.st_mtime_ns, "PI sidecar mtime differs")
    else:
        require(sidecar.get("path") is None, "vanilla unexpectedly owns a planner sidecar path")
    initialization = checkpoint.get("common_trainable_initialization_sha256")
    require(
        isinstance(initialization, str) and len(initialization) == 64, f"{owner} initialization fingerprint is invalid"
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "scale_keys": scale_keys,
        "repository_commit": provenance.get("repository_commit"),
        "content_hash_mode": provenance.get("content_hash_mode"),
        "source_sha256": source["sha256"],
        "sidecar_sha256": sidecar["sha256"],
        "common_trainable_initialization_sha256": initialization,
        "selection_policy": expected_schedule["selection_policy"],
        **expected_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-summary", type=Path, required=True)
    parser.add_argument("--pi-config", type=Path, required=True)
    parser.add_argument("--vanilla-config", type=Path, required=True)
    parser.add_argument("--pi-checkpoint", type=Path, required=True)
    parser.add_argument("--vanilla-checkpoint", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--sidecar-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-completed-passes", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()

    completed_passes = int(args.expected_completed_passes)
    expected_relay_status = {
        1: "PUSHT_MATCHED_ONE_PASS_COMPLETE",
        2: "PUSHT_MATCHED_TWO_PASS_COMPLETE",
    }[completed_passes]
    expected_optimizer_steps = completed_passes * OPTIMIZER_STEPS_PER_PASS
    expected_microbatches = completed_passes * MICROBATCHES_PER_PASS
    pass_label = {1: "one-pass", 2: "two-pass"}[completed_passes]
    protocol = {
        1: "pusht_jepa_wm_matched_one_pass_seed42_clean_preflight_v1",
        2: "pusht_jepa_wm_matched_two_pass_seed42_clean_preflight_v1",
    }[completed_passes]

    paths = {name: value.resolve() for name, value in vars(args).items() if isinstance(value, Path)}
    for name, path in paths.items():
        if name != "output" and not path.is_file():
            raise SystemExit(f"[STOP] missing {name}: {path}")
    relay = json.loads(paths["relay_summary"].read_text(encoding="utf-8"))
    require(relay.get("status") == expected_relay_status, "relay summary is incomplete")
    require(relay.get("optimizer_steps_per_arm") == expected_optimizer_steps, "relay optimizer steps differ")
    require(relay.get("microbatches_per_arm") == expected_microbatches, "relay microbatches differ")
    require(relay.get("gradient_accumulation_steps") == GRADIENT_ACCUMULATION, "relay accumulation differs")
    require(
        relay.get("configured_scheduler_horizon_optimizer_steps") == CONFIGURED_OPTIMIZER_HORIZON,
        "relay scheduler horizon differs",
    )
    require(relay.get("intentional_boundary_stop") is True, "relay boundary-stop marker is missing")
    if completed_passes == 2:
        require(relay.get("selected_completed_passes") == 2, "relay selected pass boundary differs")

    pi_config = yaml.safe_load(paths["pi_config"].read_text(encoding="utf-8"))
    vanilla_config = yaml.safe_load(paths["vanilla_config"].read_text(encoding="utf-8"))
    require(pi_config["model"]["predictor"].get("planner_identified_input_scale") is True, "PI config has no scale")
    require(pi_config.get("planner_identified", {}).get("enabled") is True, "PI config has no planner objective")
    require(
        vanilla_config["model"]["predictor"].get("planner_identified_input_scale") is False,
        "vanilla config has a scale",
    )
    require(vanilla_config.get("planner_identified") == {"enabled": False}, "vanilla planner objective is enabled")
    restored = deepcopy(vanilla_config)
    restored["folder"] = pi_config["folder"]
    restored["model"]["predictor"]["planner_identified_input_scale"] = True
    restored["planner_identified"] = deepcopy(pi_config["planner_identified"])
    require(restored == pi_config, "PI/vanilla configs differ outside intended PI fields")
    for label, config in (("PI", pi_config), ("vanilla", vanilla_config)):
        optimization = config["optimization"]["transition_model"]
        require(config["data"]["loader"]["batch_size"] == 16, f"{label} microbatch differs")
        require(optimization["gradient_accumulation_steps"] == GRADIENT_ACCUMULATION, f"{label} accumulation differs")
        require(
            optimization["expected_optimizer_steps_per_epoch"] == OPTIMIZER_STEPS_PER_PASS,
            f"{label} steps/pass differs",
        )
        require(
            optimization["total_optimizer_steps"] == CONFIGURED_OPTIMIZER_HORIZON,
            f"{label} configured horizon differs",
        )

    pi = checkpoint_audit(
        paths["pi_checkpoint"],
        "pi",
        relay["pi"]["sha256"],
        paths["source_h5"],
        paths["sidecar_h5"],
        completed_passes,
    )
    vanilla = checkpoint_audit(
        paths["vanilla_checkpoint"],
        "vanilla",
        relay["vanilla"]["sha256"],
        paths["source_h5"],
        paths["sidecar_h5"],
        completed_passes,
    )
    require(relay["pi"]["checkpoint"] == str(paths["pi_checkpoint"]), "relay PI path differs")
    require(relay["vanilla"]["checkpoint"] == str(paths["vanilla_checkpoint"]), "relay vanilla path differs")
    for key in (
        "repository_commit",
        "content_hash_mode",
        "source_sha256",
        "sidecar_sha256",
        "common_trainable_initialization_sha256",
    ):
        require(pi[key] == vanilla[key], f"PI/vanilla {key} differs")
    require(pi["repository_commit"] == relay["repository_commit"], "relay/checkpoint commit differs")
    require(
        isinstance(pi["repository_commit"], str) and len(pi["repository_commit"]) == 40,
        "training repository commit is invalid",
    )
    require(
        pi["content_hash_mode"] in {"sha256", "not_scanned_user_confirmed"},
        "training content-hash mode is invalid",
    )

    source_stat = paths["source_h5"].stat()
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "task": "pusht",
        "protocol": protocol,
        "repository_commit": pi["repository_commit"],
        "content_hash_mode": pi["content_hash_mode"],
        "source": {
            "path": str(paths["source_h5"]),
            "bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "sha256": pi["source_sha256"],
        },
        "planner": {
            "sidecar_path": str(paths["sidecar_h5"]),
            "sidecar_sha256": pi["sidecar_sha256"],
        },
        "training_configs": {
            "pi": {"path": str(paths["pi_config"]), "sha256": sha256(paths["pi_config"])},
            "vanilla": {"path": str(paths["vanilla_config"]), "sha256": sha256(paths["vanilla_config"])},
        },
        "split": {"optimizer_steps_per_pass": OPTIMIZER_STEPS_PER_PASS},
        "checkpoint_selection": {
            "policy": "native_complete_pass_boundary",
            "completed_passes": completed_passes,
            "optimizer_steps": expected_optimizer_steps,
            "microbatches": expected_microbatches,
            "configured_scheduler_horizon_optimizer_steps": CONFIGURED_OPTIMIZER_HORIZON,
        },
        "checkpoints": {"pi": pi, "vanilla": vanilla},
        "relay_summary": {"path": str(paths["relay_summary"]), "sha256": sha256(paths["relay_summary"])},
    }
    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if output.is_file():
        require(
            output.read_text(encoding="utf-8") == payload,
            "existing preflight audit differs; preserve and inspect it",
        )
        print(f"[reuse {pass_label} eval preflight PASS] {output}")
        return 0
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output)
    print(f"[{pass_label} eval preflight PASS] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
