#!/usr/bin/env python3
"""Summarize one clean, pass-matched PI-versus-vanilla evaluation seed."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from scripts.summarize_stablewm_task_multiseed import EXACT_PROTOCOL


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mcnemar_exact(pi: np.ndarray, vanilla: np.ndarray) -> dict:
    pi_only = int(np.logical_and(pi, ~vanilla).sum())
    vanilla_only = int(np.logical_and(~pi, vanilla).sum())
    discordant = pi_only + vanilla_only
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(pi_only, vanilla_only) + 1))
        pvalue = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "pi_only_successes": pi_only,
        "vanilla_only_successes": vanilla_only,
        "discordant": discordant,
        "exact_two_sided_p": pvalue,
    }


def summarize(
    input_root: Path,
    task: str,
    seed: int,
    *,
    expected_completed_passes: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict:
    require(expected_completed_passes > 0, "expected completed passes must be positive")
    audits = {}
    for arm, owner in (("learned", "pi"), ("vanilla_stepmatched", "vanilla")):
        path = input_root / task / f"seed{seed}" / arm / "arm_audit.json"
        require(path.is_file(), f"missing arm audit: {path}")
        audit = json.loads(path.read_text(encoding="utf-8"))
        require(audit.get("status") == "complete", f"incomplete arm audit: {path}")
        require(audit.get("task") == task, f"wrong task: {path}")
        require(audit.get("eval_seed") == seed, f"wrong seed: {path}")
        require(audit.get("arm") == arm and audit.get("owner") == owner, f"wrong arm/owner: {path}")
        for key, expected in EXACT_PROTOCOL.items():
            require(audit.get("evaluation", {}).get(key) == expected, f"{path}: evaluation.{key} differs")
        outcomes = audit.get("results", {}).get("episode_successes", [])
        require(len(outcomes) == 50, f"{path}: expected 50 outcomes")
        checkpoint = audit.get("checkpoint", {})
        require(
            checkpoint.get("selection_policy") == "native_complete_pass_boundary", f"{path}: wrong checkpoint policy"
        )
        require(
            checkpoint.get("selected_completed_passes") == expected_completed_passes,
            f"{path}: not pass {expected_completed_passes}",
        )
        optimizer_steps_per_pass = checkpoint.get("optimizer_steps_per_epoch")
        require(
            isinstance(optimizer_steps_per_pass, int) and optimizer_steps_per_pass > 0,
            f"{path}: invalid optimizer steps/pass",
        )
        require(checkpoint.get("optimizer_step_in_epoch") == 0, f"{path}: not a complete pass boundary")
        require(
            checkpoint.get("total_optimizer_steps")
            == expected_completed_passes * optimizer_steps_per_pass,
            f"{path}: optimizer steps differ from pass boundary",
        )
        audits[arm] = audit

    pi = audits["learned"]
    vanilla = audits["vanilla_stepmatched"]
    for key in (
        "repository_commit",
        "training_repository_commit",
        "lewm_repository_commit",
    ):
        require(pi.get(key) == vanilla.get(key), f"{key} differs")
    require(
        pi.get("source_h5", {}).get("sha256") == vanilla.get("source_h5", {}).get("sha256"),
        "source marker differs",
    )
    for key in (
        "paired_start_sha256",
        "environment_state_sha256",
        "seeds",
    ):
        require(
            pi.get("results", {}).get(key) == vanilla.get("results", {}).get(key),
            f"paired {key} differs",
        )
    require(
        pi.get("candidate_rng_audit", {}).get("generator_state_sequence_sha256")
        == vanilla.get("candidate_rng_audit", {}).get("generator_state_sequence_sha256"),
        "CEM RNG sequence differs",
    )
    pi_checkpoint = pi["checkpoint"]
    vanilla_checkpoint = vanilla["checkpoint"]
    require(
        pi_checkpoint["optimizer_steps_per_epoch"]
        == vanilla_checkpoint["optimizer_steps_per_epoch"],
        "PI/vanilla optimizer steps/pass differ",
    )
    require(
        pi_checkpoint["total_optimizer_steps"]
        == vanilla_checkpoint["total_optimizer_steps"],
        "PI/vanilla total optimizer steps differ",
    )
    require(pi_checkpoint.get("sha256") != vanilla_checkpoint.get("sha256"), "PI/vanilla checkpoints are identical")
    require(
        pi_checkpoint.get("common_trainable_initialization_sha256")
        == vanilla_checkpoint.get("common_trainable_initialization_sha256"),
        "PI/vanilla initialization differs",
    )

    pi_outcomes = np.asarray(pi["results"]["episode_successes"], dtype=bool)
    vanilla_outcomes = np.asarray(vanilla["results"]["episode_successes"], dtype=bool)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(pi_outcomes), size=(bootstrap_draws, len(pi_outcomes)))
    bootstrap = (pi_outcomes[indices].mean(axis=1) - vanilla_outcomes[indices].mean(axis=1)) * 100.0
    result = {
        "schema_version": 1,
        "status": "complete",
        "estimand": (
            f"cross-checkpoint matched {expected_completed_passes}-pass "
            "PI-LTC versus vanilla"
        ),
        "task": task,
        "eval_seed": seed,
        "episodes": 50,
        "action_noise_std": 0.0,
        "completed_training_passes_per_arm": expected_completed_passes,
        "optimizer_steps_per_pass": pi_checkpoint["optimizer_steps_per_epoch"],
        "optimizer_steps_per_arm": pi_checkpoint["total_optimizer_steps"],
        "pi_success_rate": float(pi_outcomes.mean() * 100.0),
        "vanilla_success_rate": float(vanilla_outcomes.mean() * 100.0),
        "difference_pp": float((pi_outcomes.mean() - vanilla_outcomes.mean()) * 100.0),
        "paired_bootstrap": {
            "draws": bootstrap_draws,
            "rng_seed": bootstrap_seed,
            "ci95_pp": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        },
        "mcnemar": mcnemar_exact(pi_outcomes, vanilla_outcomes),
        "paired_start_sha256": pi["results"]["paired_start_sha256"],
        "environment_state_sha256": pi["results"]["environment_state_sha256"],
        "candidate_rng_sequence_sha256": pi["candidate_rng_audit"]["generator_state_sequence_sha256"],
        "repository_commit": pi["repository_commit"],
        "training_repository_commit": pi["training_repository_commit"],
        "lewm_repository_commit": pi["lewm_repository_commit"],
        "checkpoints": {
            "pi": {
                "path": pi_checkpoint["path"],
                "sha256": pi_checkpoint["sha256"],
            },
            "vanilla": {
                "path": vanilla_checkpoint["path"],
                "sha256": vanilla_checkpoint["sha256"],
            },
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--task", choices=("pusht", "cube"), required=True)
    parser.add_argument("--eval-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--expected-completed-passes", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    args = parser.parse_args()
    result = summarize(
        args.input_root.resolve(),
        args.task,
        args.eval_seed,
        expected_completed_passes=args.expected_completed_passes,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
