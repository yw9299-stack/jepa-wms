#!/usr/bin/env python3
"""Strict paired summary for JEPA-WM on OGBench PointMaze-Medium."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ARMS = ("learned", "identity", "fixed06", "fixed04", "vanilla_5pass")
EXPECTED_SCALES = {
    "identity": 1.0,
    "fixed06": 0.6,
    "fixed04": 0.4,
    "vanilla_5pass": None,
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _mcnemar_exact(learned_only: int, comparator_only: int) -> float:
    discordant = learned_only + comparator_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(learned_only, comparator_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_ci(delta: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(samples, dtype=np.float64)
    chunk = 10_000
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = delta[indices].mean(axis=1) * 100.0
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _load(root / "run_manifest.json")
    if manifest.get("protocol") != "ogbench_temporal_pointmaze_medium_stablewm_exact_v1":
        raise SystemExit("[STOP] run manifest protocol mismatch")
    if manifest["evaluation"]["eval_seed"] != args.eval_seed:
        raise SystemExit("[STOP] run manifest eval seed mismatch")
    if manifest["evaluation"]["episodes"] != args.episodes:
        raise SystemExit("[STOP] run manifest episode count mismatch")

    audits = {}
    reference_protocol = None
    reference_pairs = None
    pi_checkpoint = None
    for arm in ARMS:
        path = root / arm / "arm_audit.json"
        if not path.is_file():
            raise SystemExit(f"[STOP] missing arm audit: {path}")
        audit = _load(path)
        if audit.get("status") != "complete" or audit.get("training") is not False:
            raise SystemExit(f"[STOP] incomplete or training arm audit: {arm}")
        if audit.get("arm") != arm:
            raise SystemExit(f"[STOP] arm identity mismatch: {arm}")
        expected_owner = "vanilla" if arm == "vanilla_5pass" else "pi"
        if audit.get("owner") != expected_owner:
            raise SystemExit(f"[STOP] checkpoint owner mismatch: {arm}")
        if audit["evaluation"]["eval_seed"] != args.eval_seed:
            raise SystemExit(f"[STOP] eval seed mismatch: {arm}")
        outcomes = audit["results"]["episode_successes"]
        pairs = list(zip(audit["results"]["episodes_idx"], audit["results"]["start_steps"]))
        if len(outcomes) != args.episodes or len(pairs) != args.episodes:
            raise SystemExit(f"[STOP] episode cardinality mismatch: {arm}")
        expected_rate = 100.0 * sum(bool(value) for value in outcomes) / args.episodes
        if abs(expected_rate - float(audit["results"]["success_rate"])) > 1.0e-9:
            raise SystemExit(f"[STOP] success-rate arithmetic mismatch: {arm}")
        if reference_protocol is None:
            reference_protocol = audit["evaluation"]
            reference_pairs = pairs
        elif audit["evaluation"] != reference_protocol or pairs != reference_pairs:
            raise SystemExit(f"[STOP] pairing/protocol mismatch: {arm}")
        if arm in EXPECTED_SCALES:
            expected_scale = EXPECTED_SCALES[arm]
            actual_scale = audit.get("effective_scale")
            if expected_scale is None:
                if actual_scale is not None:
                    raise SystemExit("[STOP] vanilla arm unexpectedly exposes a PI scale")
            elif abs(float(actual_scale) - expected_scale) > 1.0e-6:
                raise SystemExit(f"[STOP] effective scale mismatch: {arm}")
        if arm != "vanilla_5pass":
            checkpoint_sha = audit["checkpoint"]["sha256"]
            if pi_checkpoint is None:
                pi_checkpoint = checkpoint_sha
            elif checkpoint_sha != pi_checkpoint:
                raise SystemExit("[STOP] PI arms do not use the same checkpoint")
        audits[arm] = audit

    arm_rows = []
    for arm in ARMS:
        audit = audits[arm]
        successes = sum(audit["results"]["episode_successes"])
        arm_rows.append(
            {
                "eval_seed": args.eval_seed,
                "arm": arm,
                "checkpoint_owner": audit["owner"],
                "effective_scale": audit.get("effective_scale"),
                "success_count": successes,
                "episode_count": args.episodes,
                "success_rate": 100.0 * successes / args.episodes,
            }
        )

    learned = np.asarray(audits["learned"]["results"]["episode_successes"], dtype=np.int8)
    outcome_rows = []
    comparison_rows = []
    for offset, comparator in enumerate(ARMS[1:]):
        other = np.asarray(audits[comparator]["results"]["episode_successes"], dtype=np.int8)
        delta = learned - other
        learned_only = int(np.sum(delta == 1))
        comparator_only = int(np.sum(delta == -1))
        low, high = _paired_ci(
            delta,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + offset,
        )
        comparison = f"learned_minus_{comparator}"
        comparison_rows.append(
            {
                "comparison": comparison,
                "eval_seed": args.eval_seed,
                "paired_episodes": args.episodes,
                "delta_percentage_points": float(delta.mean() * 100.0),
                "paired_bootstrap_ci_low": low,
                "paired_bootstrap_ci_high": high,
                "learned_only_successes": learned_only,
                "comparator_only_successes": comparator_only,
                "discordant_total": learned_only + comparator_only,
                "exact_mcnemar_two_sided_p": _mcnemar_exact(learned_only, comparator_only),
            }
        )
        for index, ((episode_id, start_step), learned_value, comparator_value) in enumerate(
            zip(reference_pairs, learned.tolist(), other.tolist())
        ):
            outcome_rows.append(
                {
                    "comparison": comparison,
                    "episode_index": index,
                    "episode_id": episode_id,
                    "start_step": start_step,
                    "learned_success": bool(learned_value),
                    "comparator_success": bool(comparator_value),
                }
            )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "protocol": manifest["protocol"],
        "training": False,
        "evaluation": manifest["evaluation"],
        "arms": arm_rows,
        "comparisons": comparison_rows,
    }
    summary_root = root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_root / "arm_summary.csv", arm_rows)
    _write_csv(summary_root / "paired_episode_outcomes.csv", outcome_rows)
    _write_csv(summary_root / "paired_comparisons.csv", comparison_rows)
    _write_json_atomic(summary_root / "paired_summary.json", summary)

    print("[JEPA-WM OGBench PointMaze-Medium exact-protocol summary]")
    for row in arm_rows:
        print(f"  {row['arm']}: SR={row['success_rate']:.1f}% " f"({row['success_count']}/{row['episode_count']})")
    print("[paired learned-minus-comparator]")
    for row in comparison_rows:
        print(
            f"  {row['comparison']}: delta={row['delta_percentage_points']:+.1f}pp "
            f"CI95=[{row['paired_bootstrap_ci_low']:+.1f},{row['paired_bootstrap_ci_high']:+.1f}] "
            f"discordant={row['learned_only_successes']}:{row['comparator_only_successes']} "
            f"p={row['exact_mcnemar_two_sided_p']:.6g}"
        )
    print(f"[saved] {summary_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
