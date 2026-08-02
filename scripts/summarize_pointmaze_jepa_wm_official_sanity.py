#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


TAG = "native_cem30_s300_k10_h6_nas6_ctxt2"


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_outcomes(root, arm):
    path = root / "arms" / arm / "simu_env_planning" / TAG / "episode_outcomes.csv"
    rows = read_csv(path)
    keys = [(row["task"], int(row["episode_index"]), int(row["episode_seed"])) for row in rows]
    values = np.asarray([int(row["success"]) for row in rows], dtype=np.int8)
    return keys, values


def paired_ci(left, right, seed, samples=100000):
    difference = left.astype(np.float64) - right.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 10000):
        end = min(start + 10000, samples)
        indices = rng.integers(0, len(difference), size=(end - start, len(difference)))
        means[start:end] = difference[indices].mean(axis=1) * 100.0
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def mcnemar_p(left_only, right_only):
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--pi-root", type=Path, required=True)
    parser.add_argument("--vanilla-root", type=Path, required=True)
    args = parser.parse_args()

    manifests = {
        "official": json.loads((args.official_root / "run_manifest.json").read_text()),
        "pi_learned": json.loads((args.pi_root / "run_manifest.json").read_text()),
        "vanilla_5pass": json.loads((args.vanilla_root / "run_manifest.json").read_text()),
    }
    protocols = [manifest["evaluation"] for manifest in manifests.values()]
    if protocols[1:] != protocols[:-1]:
        raise SystemExit("[STOP] official, PI, and vanilla evaluation protocols differ")
    protocol = protocols[0]
    episodes = int(protocol["episodes"])

    roots_and_arms = {
        "official": (args.official_root, "official"),
        "pi_learned": (args.pi_root, "learned"),
        "vanilla_5pass": (args.vanilla_root, "vanilla"),
    }
    outcomes = {}
    canonical_keys = None
    for name, (root, arm) in roots_and_arms.items():
        keys, values = load_outcomes(root, arm)
        if len(values) != episodes or len(set(keys)) != episodes:
            raise SystemExit(f"[STOP] incomplete or duplicate {name} outcomes: {len(values)}/{episodes}")
        if canonical_keys is None:
            canonical_keys = keys
        elif keys != canonical_keys:
            raise SystemExit(f"[STOP] {name} episode keys are not exactly paired")
        outcomes[name] = values

    arm_rows = [
        {
            "arm": name,
            "success_count": int(values.sum()),
            "episode_count": episodes,
            "success_rate": float(values.mean() * 100.0),
        }
        for name, values in outcomes.items()
    ]
    comparisons = []
    for index, comparator in enumerate(("pi_learned", "vanilla_5pass")):
        left = outcomes["official"]
        right = outcomes[comparator]
        left_only = int(np.logical_and(left == 1, right == 0).sum())
        right_only = int(np.logical_and(left == 0, right == 1).sum())
        ci_low, ci_high = paired_ci(left, right, seed=20260802 + index)
        comparisons.append(
            {
                "comparison": f"official_minus_{comparator}",
                "delta_percentage_points": float((left.mean() - right.mean()) * 100.0),
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "official_only_successes": left_only,
                "comparator_only_successes": right_only,
                "discordant_total": left_only + right_only,
                "exact_mcnemar_two_sided_p": mcnemar_p(left_only, right_only),
            }
        )

    paired_rows = []
    for row_index, key in enumerate(canonical_keys):
        paired_rows.append(
            {
                "task": key[0],
                "episode_index": key[1],
                "episode_seed": key[2],
                **{f"{name}_success": int(values[row_index]) for name, values in outcomes.items()},
            }
        )

    write_csv(args.official_root / "official_sanity_arm_summary.csv", arm_rows)
    write_csv(args.official_root / "official_sanity_paired_outcomes.csv", paired_rows)
    write_csv(args.official_root / "official_sanity_comparisons.csv", comparisons)
    summary = {
        "interpretation_gate": {
            "official_high": "The current native eval pipeline is functional; five-pass H5 training is insufficient or mismatched.",
            "official_low": "The common bottleneck is evaluation/data/action-normalization alignment, not the PI objective alone.",
        },
        "protocol": protocol,
        "manifests": {
            name: str((root / "run_manifest.json").resolve())
            for name, (root, _) in roots_and_arms.items()
        },
        "arms": arm_rows,
        "comparisons": comparisons,
    }
    (args.official_root / "official_sanity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print("[JEPA-WM official-checkpoint sanity comparison]")
    for row in arm_rows:
        print(f"  {row['arm']}: SR={row['success_rate']:.1f}% ({row['success_count']}/{episodes})")
    for row in comparisons:
        print(
            f"  {row['comparison']}: delta={row['delta_percentage_points']:+.1f}pp "
            f"CI95=[{row['paired_bootstrap_ci_low']:+.1f},{row['paired_bootstrap_ci_high']:+.1f}] "
            f"discordant={row['official_only_successes']}:{row['comparator_only_successes']} "
            f"p={row['exact_mcnemar_two_sided_p']:.6g}"
        )
    print(f"[saved] {args.official_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
