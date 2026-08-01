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


def paired_ci(left, right, seed=20260801, samples=100000):
    differences = left.astype(np.float64) - right.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 10000):
        end = min(start + 10000, samples)
        indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
        means[start:end] = differences[indices].mean(axis=1) * 100.0
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def mcnemar_p(left_only, right_only):
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def load_outcomes(root, arm):
    path = root / "arms" / arm / "simu_env_planning" / TAG / "episode_outcomes.csv"
    rows = read_csv(path)
    keys = [(row["task"], int(row["episode_index"]), int(row["episode_seed"])) for row in rows]
    values = np.asarray([int(row["success"]) for row in rows], dtype=np.int8)
    return rows, keys, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-root", type=Path, required=True)
    parser.add_argument("--pi-root", type=Path, required=True)
    args = parser.parse_args()

    vanilla_manifest = json.loads((args.vanilla_root / "run_manifest.json").read_text())
    pi_manifest = json.loads((args.pi_root / "run_manifest.json").read_text())
    if vanilla_manifest["evaluation"] != pi_manifest["evaluation"]:
        raise SystemExit("[STOP] vanilla and PI evaluation protocols differ")
    protocol = vanilla_manifest["evaluation"]
    episodes = int(protocol["episodes"])

    _, vanilla_keys, vanilla = load_outcomes(args.vanilla_root, "vanilla")
    _, pi_keys, pi = load_outcomes(args.pi_root, "learned")
    if len(vanilla) != episodes or len(pi) != episodes:
        raise SystemExit(f"[STOP] incomplete outcomes: vanilla={len(vanilla)} PI={len(pi)} expected={episodes}")
    if vanilla_keys != pi_keys:
        raise SystemExit("[STOP] vanilla and PI episode keys differ; comparison is not paired")
    if len(set(vanilla_keys)) != episodes:
        raise SystemExit("[STOP] duplicate paired episode keys")

    pi_only = int(np.logical_and(pi == 1, vanilla == 0).sum())
    vanilla_only = int(np.logical_and(pi == 0, vanilla == 1).sum())
    ci_low, ci_high = paired_ci(pi, vanilla)
    arm_rows = [
        {
            "arm": "pi_learned",
            "success_count": int(pi.sum()),
            "episode_count": episodes,
            "success_rate": float(pi.mean() * 100.0),
        },
        {
            "arm": "vanilla_5pass",
            "success_count": int(vanilla.sum()),
            "episode_count": episodes,
            "success_rate": float(vanilla.mean() * 100.0),
        },
    ]
    comparison = {
        "comparison": "pi_learned_minus_vanilla_5pass",
        "delta_percentage_points": float((pi.mean() - vanilla.mean()) * 100.0),
        "paired_bootstrap_ci_low": ci_low,
        "paired_bootstrap_ci_high": ci_high,
        "pi_only_successes": pi_only,
        "vanilla_only_successes": vanilla_only,
        "discordant_total": pi_only + vanilla_only,
        "exact_mcnemar_two_sided_p": mcnemar_p(pi_only, vanilla_only),
    }
    paired_rows = []
    for key, pi_value, vanilla_value in zip(pi_keys, pi, vanilla):
        paired_rows.append(
            {
                "task": key[0],
                "episode_index": key[1],
                "episode_seed": key[2],
                "pi_learned_success": int(pi_value),
                "vanilla_5pass_success": int(vanilla_value),
            }
        )

    write_csv(args.vanilla_root / "pi_vs_vanilla_arm_summary.csv", arm_rows)
    write_csv(args.vanilla_root / "pi_vs_vanilla_paired_outcomes.csv", paired_rows)
    write_csv(args.vanilla_root / "pi_vs_vanilla_comparison.csv", [comparison])
    summary = {
        "protocol": protocol,
        "pi_manifest": str((args.pi_root / "run_manifest.json").resolve()),
        "vanilla_manifest": str((args.vanilla_root / "run_manifest.json").resolve()),
        "arms": arm_rows,
        "comparison": comparison,
    }
    (args.vanilla_root / "pi_vs_vanilla_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print("[JEPA-WM PointMaze matched 5-pass comparison]")
    for row in arm_rows:
        print(f"  {row['arm']}: SR={row['success_rate']:.1f}% ({row['success_count']}/{episodes})")
    print(
        f"  {comparison['comparison']}: delta={comparison['delta_percentage_points']:+.1f}pp "
        f"CI95=[{ci_low:+.1f},{ci_high:+.1f}] "
        f"discordant={pi_only}:{vanilla_only} p={comparison['exact_mcnemar_two_sided_p']:.6g}"
    )
    print(f"[saved] {args.vanilla_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
