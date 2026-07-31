#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ARMS = ("learned", "identity", "fixed06", "fixed04")


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_ci(left, right, seed=20260731, samples=100000):
    differences = left.astype(np.float64) - right.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 10000):
        end = min(start + 10000, samples)
        indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
        means[start:end] = differences[indices].mean(axis=1) * 100.0
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def mcnemar_p(left_only, right_only):
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output_root / "run_manifest.json").read_text())
    protocol = manifest["evaluation"]
    episodes = int(protocol["episodes"])
    eval_seed = int(protocol["eval_seed"])
    tag = "native_cem30_s300_k10_h6_nas6_ctxt2"
    expected_keys = None
    outcomes = {}
    arm_rows = []
    pooled_rows = []

    for arm in ARMS:
        work_dir = args.output_root / "arms" / arm / "simu_env_planning" / tag
        rows = read_csv(work_dir / "episode_outcomes.csv")
        if len(rows) != episodes:
            raise SystemExit(f"[STOP] {arm}: rows={len(rows)}, expected={episodes}")
        keys = [(row["task"], int(row["episode_index"]), int(row["episode_seed"])) for row in rows]
        if len(set(keys)) != episodes:
            raise SystemExit(f"[STOP] {arm}: duplicate episode keys")
        expected_seeds = [int((eval_seed * eval_seed + ep * eval_seed) % (2**32 - 2)) for ep in range(episodes)]
        if [key[2] for key in keys] != expected_seeds:
            raise SystemExit(f"[STOP] {arm}: episode seeds do not match the declared protocol")
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise SystemExit(f"[STOP] {arm}: paired episode keys differ")

        audit = json.loads((work_dir / "planner_scale_audit.json").read_text())
        declared = manifest["arms"][arm]
        expected_effective = (
            manifest["checkpoint"]["learned_scale"] if declared["mode"] == "learned" else declared["value"]
        )
        if audit["mode"] != declared["mode"] or not math.isclose(
            audit["effective_scale"], expected_effective, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise SystemExit(f"[STOP] {arm}: runtime scale audit mismatch: {audit}")
        if not audit["parameter_unchanged"]:
            raise SystemExit(f"[STOP] {arm}: checkpoint parameter was mutated")

        values = np.asarray([int(row["success"]) for row in rows], dtype=np.int8)
        outcomes[arm] = values
        arm_rows.append(
            {
                "arm": arm,
                "effective_scale": audit["effective_scale"],
                "success_count": int(values.sum()),
                "episode_count": episodes,
                "success_rate": float(values.mean() * 100.0),
            }
        )
        for row, value in zip(rows, values):
            pooled_rows.append(
                {
                    "task": row["task"],
                    "episode_index": int(row["episode_index"]),
                    "episode_seed": int(row["episode_seed"]),
                    "arm": arm,
                    "success": int(value),
                }
            )

    comparisons = []
    learned = outcomes["learned"]
    for variant in ARMS[1:]:
        other = outcomes[variant]
        learned_only = int(np.logical_and(learned == 1, other == 0).sum())
        variant_only = int(np.logical_and(learned == 0, other == 1).sum())
        ci_low, ci_high = paired_ci(learned, other)
        comparisons.append(
            {
                "comparison": f"learned_minus_{variant}",
                "delta_percentage_points": float((learned.mean() - other.mean()) * 100.0),
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "learned_only_successes": learned_only,
                "variant_only_successes": variant_only,
                "discordant_total": learned_only + variant_only,
                "exact_mcnemar_two_sided_p": mcnemar_p(learned_only, variant_only),
            }
        )

    write_csv(args.output_root / "arm_summary.csv", arm_rows)
    write_csv(args.output_root / "paired_episode_outcomes.csv", pooled_rows)
    write_csv(args.output_root / "paired_comparisons.csv", comparisons)
    summary = {"manifest": manifest, "arms": arm_rows, "comparisons": comparisons}
    (args.output_root / "paired_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("[JEPA-WM PointMaze same-checkpoint scale intervention]")
    for row in arm_rows:
        print(f"  {row['arm']}: scale={row['effective_scale']:.6f} SR={row['success_rate']:.1f}%")
    print("[paired learned-minus-variant]")
    for row in comparisons:
        print(
            f"  {row['comparison']}: delta={row['delta_percentage_points']:+.1f}pp "
            f"CI95=[{row['paired_bootstrap_ci_low']:+.1f},{row['paired_bootstrap_ci_high']:+.1f}] "
            f"discordant={row['learned_only_successes']}:{row['variant_only_successes']} "
            f"p={row['exact_mcnemar_two_sided_p']:.6g}"
        )
    print(f"[saved] {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
