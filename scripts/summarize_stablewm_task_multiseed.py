#!/usr/bin/env python3
"""Summarize same-checkpoint and cross-checkpoint estimands without mixing them."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44)
REQUIRED_ARMS = {
    "learned": "pi",
    "identity": "pi",
    "vanilla_stepmatched": "vanilla",
}
EXACT_PROTOCOL = {
    "episodes": 50,
    "action_noise_std": 0.0,
    "eval_budget": 50,
    "goal_offset_steps": 25,
    "horizon": 5,
    "receding_horizon": 5,
    "action_block": 5,
    "cem_steps": 16,
    "cem_samples": 256,
    "cem_topk": 64,
    "lewm_standardized_columns": ["action"],
}


def _load_arm(root: Path, task: str, seed: int, arm: str) -> dict:
    path = root / task / f"seed{seed}" / arm / "arm_audit.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("task") != task:
        raise ValueError(f"incomplete or wrong-task audit: {path}")
    if audit.get("protocol") != f"{task}_stablewm_exact_jepa_v1":
        raise ValueError(f"wrong protocol audit: {path}")
    if int(audit.get("eval_seed", -1)) != seed or audit.get("arm") != arm:
        raise ValueError(f"seed/arm mismatch: {path}")
    expected_owner = REQUIRED_ARMS.get(arm, "pi")
    if audit.get("owner") != expected_owner:
        raise ValueError(f"owner mismatch: {path}")
    evaluation = audit.get("evaluation", {})
    for key, expected in EXACT_PROTOCOL.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"{path}: evaluation.{key}={evaluation.get(key)!r}, expected {expected!r}")
    outcomes = audit.get("results", {}).get("episode_successes", [])
    if len(outcomes) != 50:
        raise ValueError(f"{path}: expected 50 episode outcomes")
    return audit


def _mcnemar_exact(left: np.ndarray, right: np.ndarray) -> dict:
    left_only = int(np.logical_and(left, ~right).sum())
    right_only = int(np.logical_and(~left, right).sum())
    discordant = left_only + right_only
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        pvalue = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "discordant": discordant,
        "exact_two_sided_p": pvalue,
    }


def _bootstrap(seed_pairs: dict[int, tuple[np.ndarray, np.ndarray]], draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ordered_seeds = sorted(seed_pairs)
    pooled_left = np.concatenate([seed_pairs[value][0] for value in ordered_seeds])
    pooled_right = np.concatenate([seed_pairs[value][1] for value in ordered_seeds])
    count = len(pooled_left)
    pooled_differences = np.empty(draws, dtype=np.float64)
    hierarchical_differences = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indices = rng.integers(0, count, size=count)
        pooled_differences[draw] = (
            pooled_left[indices].mean() - pooled_right[indices].mean()
        ) * 100.0
        sampled_seeds = rng.choice(ordered_seeds, size=len(ordered_seeds), replace=True)
        differences = []
        for sampled_seed in sampled_seeds:
            left, right = seed_pairs[int(sampled_seed)]
            episode_indices = rng.integers(0, len(left), size=len(left))
            differences.extend((left[episode_indices] - right[episode_indices]).tolist())
        hierarchical_differences[draw] = np.mean(differences) * 100.0
    return {
        "draws": draws,
        "rng_seed": seed,
        "pooled_episode_ci95_pp": np.percentile(pooled_differences, [2.5, 97.5]).tolist(),
        "hierarchical_seed_episode_ci95_pp": np.percentile(
            hierarchical_differences, [2.5, 97.5]
        ).tolist(),
    }


def _contrast(
    audits: dict[int, dict[str, dict]],
    *,
    left_arm: str,
    right_arm: str,
    estimand: str,
    draws: int,
    bootstrap_seed: int,
) -> dict:
    seed_pairs = {}
    per_seed = []
    for seed in sorted(audits):
        left_audit = audits[seed][left_arm]
        right_audit = audits[seed][right_arm]
        for audit_key in (
            "paired_start_sha256",
            "environment_state_sha256",
            "seeds",
        ):
            left_value = left_audit["results"].get(audit_key)
            right_value = right_audit["results"].get(audit_key)
            if left_value != right_value:
                raise ValueError(f"seed {seed}: {audit_key} differs between {left_arm}/{right_arm}")
        left_rng = left_audit["candidate_rng_audit"]["generator_state_sequence_sha256"]
        right_rng = right_audit["candidate_rng_audit"]["generator_state_sequence_sha256"]
        if left_rng != right_rng:
            raise ValueError(f"seed {seed}: CEM generator-state sequence differs")
        left = np.asarray(left_audit["results"]["episode_successes"], dtype=np.int8)
        right = np.asarray(right_audit["results"]["episode_successes"], dtype=np.int8)
        seed_pairs[seed] = (left, right)
        per_seed.append(
            {
                "seed": seed,
                "left_success_rate": float(left.mean() * 100.0),
                "right_success_rate": float(right.mean() * 100.0),
                "difference_pp": float((left.mean() - right.mean()) * 100.0),
                "mcnemar": _mcnemar_exact(left.astype(bool), right.astype(bool)),
            }
        )

    pooled_left = np.concatenate([pair[0] for pair in seed_pairs.values()])
    pooled_right = np.concatenate([pair[1] for pair in seed_pairs.values()])
    return {
        "estimand": estimand,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "per_seed": per_seed,
        "pooled": {
            "episodes": len(pooled_left),
            "left_success_rate": float(pooled_left.mean() * 100.0),
            "right_success_rate": float(pooled_right.mean() * 100.0),
            "difference_pp": float((pooled_left.mean() - pooled_right.mean()) * 100.0),
            "mcnemar": _mcnemar_exact(pooled_left.astype(bool), pooled_right.astype(bool)),
        },
        "bootstrap": _bootstrap(seed_pairs, draws, bootstrap_seed),
    }


def summarize(root: Path, task: str, draws: int, bootstrap_seed: int) -> dict:
    audits = {
        seed: {arm: _load_arm(root, task, seed, arm) for arm in REQUIRED_ARMS}
        for seed in SEEDS
    }
    for arm in REQUIRED_ARMS:
        hashes = {
            audits[seed][arm]["checkpoint"]["sha256"]
            for seed in SEEDS
        }
        if len(hashes) != 1:
            raise ValueError(f"{arm}: checkpoint differs across evaluation seeds")
    for field in ("repository_commit", "lewm_repository_commit"):
        values = {
            audits[seed][arm].get(field)
            for seed in SEEDS
            for arm in REQUIRED_ARMS
        }
        if len(values) != 1 or None in values:
            raise ValueError(f"{field} differs across evaluation arms")
    source_hashes = {
        audits[seed][arm].get("source_h5", {}).get("sha256")
        for seed in SEEDS
        for arm in REQUIRED_ARMS
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise ValueError("source HDF5 differs across evaluation arms")
    for seed, arms in audits.items():
        learned_hash = arms["learned"]["checkpoint"]["sha256"]
        identity_hash = arms["identity"]["checkpoint"]["sha256"]
        vanilla_hash = arms["vanilla_stepmatched"]["checkpoint"]["sha256"]
        if learned_hash != identity_hash:
            raise ValueError(f"seed {seed}: learned and identity are not the same checkpoint")
        if learned_hash == vanilla_hash:
            raise ValueError(f"seed {seed}: PI and vanilla checkpoints are unexpectedly identical")
        learned_initialization = arms["learned"]["checkpoint"].get(
            "common_trainable_initialization_sha256"
        )
        vanilla_initialization = arms["vanilla_stepmatched"]["checkpoint"].get(
            "common_trainable_initialization_sha256"
        )
        if (
            not isinstance(learned_initialization, str)
            or len(learned_initialization) != 64
            or learned_initialization != vanilla_initialization
        ):
            raise ValueError(f"seed {seed}: PI and vanilla initializations are not matched")

    same_checkpoint = _contrast(
        audits,
        left_arm="learned",
        right_arm="identity",
        estimand="A: same PI checkpoint, learned scale versus identity scale",
        draws=draws,
        bootstrap_seed=bootstrap_seed,
    )
    cross_checkpoint = _contrast(
        audits,
        left_arm="learned",
        right_arm="vanilla_stepmatched",
        estimand=(
            "B: task-specific LeWM-step-matched PI checkpoint versus matched "
            "vanilla checkpoint"
        ),
        draws=draws,
        bootstrap_seed=bootstrap_seed + 1,
    )
    optional_seed42 = {}
    for arm in ("fixed06", "fixed04"):
        path = root / task / "seed42" / arm / "arm_audit.json"
        if path.is_file():
            optional_audit = _load_arm(root, task, 42, arm)
            if (
                optional_audit["checkpoint"]["sha256"]
                != audits[42]["learned"]["checkpoint"]["sha256"]
            ):
                raise ValueError(f"seed 42: learned and {arm} are not the same checkpoint")
            optional = {42: {"learned": audits[42]["learned"], arm: optional_audit}}
            optional_seed42[arm] = _contrast(
                optional,
                left_arm="learned",
                right_arm=arm,
                estimand=f"optional seed42 same-checkpoint learned versus {arm}",
                draws=draws,
                bootstrap_seed=bootstrap_seed + (6 if arm == "fixed06" else 4),
            )
    return {
        "schema_version": 1,
        "status": "complete",
        "task": task,
        "seeds": list(SEEDS),
        "episodes_per_seed": 50,
        "same_checkpoint_estimand_A": same_checkpoint,
        "cross_checkpoint_estimand_B": cross_checkpoint,
        "optional_same_checkpoint_seed42": optional_seed42,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--task", choices=("pusht", "cube"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    args = parser.parse_args()
    result = summarize(
        args.input_root.resolve(),
        args.task,
        args.bootstrap_draws,
        args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
