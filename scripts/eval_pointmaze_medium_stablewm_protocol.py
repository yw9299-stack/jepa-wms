#!/usr/bin/env python3
"""Evaluate JEPA-WM checkpoints with LEWM's exact OGBench Medium protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir

from app.vjepa_wm.modelcustom.stablewm_pointmaze_cost import (
    build_stablewm_pointmaze_cost,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class RawCHWTensor:
    """Keep the evaluator's HWC->CHW conversion but defer JEPA normalization."""

    def __call__(self, value):
        return torch.as_tensor(value)


def _preflight_cost(model, source_h5: Path) -> dict:
    with h5py.File(source_h5, "r", swmr=True) as source:
        pixels = np.asarray(source["pixels"][[0, 25]], dtype=np.uint8)
        proprio = np.asarray(source["observation"][0], dtype=np.float32)
        action = np.asarray(source["action"][:], dtype=np.float32)

    samples = 4
    context = torch.from_numpy(pixels[0]).permute(2, 0, 1)[None, None]
    goal = torch.from_numpy(pixels[1]).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(proprio)[None, None]
    info = {
        "pixels": context.unsqueeze(1).expand(1, samples, *context.shape[1:]),
        "goal": goal.unsqueeze(1).expand(1, samples, *goal.shape[1:]),
        "proprio": proprio.unsqueeze(1).expand(1, samples, *proprio.shape[1:]),
    }
    candidates = torch.zeros(1, samples, 6, model.model.action_dim, device=model.model.device)
    candidates[:, 1:] = torch.randn_like(candidates[:, 1:])
    costs = model.get_cost(info, candidates)
    if costs.shape != (1, samples) or not torch.isfinite(costs).all():
        raise RuntimeError(f"planner cost preflight failed: shape={costs.shape}, costs={costs}")

    action_mean = action.mean(axis=0)
    action_std = action.std(axis=0)
    mean_error = float(np.max(np.abs(action_mean - model.action_mean.numpy())))
    std_error = float(np.max(np.abs(action_std - model.action_std.numpy())))
    if mean_error > 1.0e-6 or std_error > 1.0e-6:
        raise RuntimeError(
            "action normalization mismatch between checkpoint adapter and source HDF5: "
            f"mean_error={mean_error}, std_error={std_error}"
        )
    return {
        "passed": True,
        "candidate_cost_shape": list(costs.shape),
        "candidate_costs": [float(value) for value in costs[0].cpu()],
        "action_mean_max_abs_error": mean_error,
        "action_std_max_abs_error": std_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lewm-repo", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="jepa-latest.pth.tar")
    parser.add_argument("--owner", choices=("pi", "vanilla"), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    lewm_repo = args.lewm_repo.resolve()
    source_h5 = args.source_h5.resolve()
    training_config_path = args.training_config.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_path = checkpoint_dir / args.checkpoint
    output_root = args.output_root.resolve()
    arm_root = output_root / args.arm
    for required in (
        source_h5,
        training_config_path,
        checkpoint_path,
        lewm_repo / "eval_pointmaze_topdown.py",
        lewm_repo / "config/eval/pointmaze_topdown_medium.yaml",
    ):
        if not required.is_file():
            raise SystemExit(f"[STOP] missing required input: {required}")
    if args.episodes <= 0:
        raise SystemExit("[STOP] episodes must be positive")

    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    model = build_stablewm_pointmaze_cost(
        checkpoint_dir=checkpoint_dir,
        checkpoint=args.checkpoint,
        training_config=training_config,
        source_h5=source_h5,
        arm=args.arm,
        owner=args.owner,
        device=args.device,
    ).to(args.device)
    preflight = _preflight_cost(model, source_h5)
    print(f"[adapter preflight] {preflight}")

    # Import LEWM only after JEPA-WM is initialized.  Both repositories expose
    # top-level modules named ``src``; this ordering prevents namespace capture.
    os.environ["STABLEWM_HOME"] = str(source_h5.parents[1])
    sys.path.insert(0, str(lewm_repo))
    import eval_pointmaze_topdown as lewm_eval  # noqa: PLC0415

    captured: dict = {}
    original_evaluate = lewm_eval.evaluate_from_dataset_with_goal_pixels

    def evaluate_and_audit(world, dataset, **kwargs):
        results = original_evaluate(world, dataset, **kwargs)
        captured.update(
            {
                "episodes_idx": [int(value) for value in kwargs["episodes_idx"]],
                "start_steps": [int(value) for value in kwargs["start_steps"]],
                "episode_successes": [bool(value) for value in results["episode_successes"]],
                "success_rate": float(results["success_rate"]),
                "seeds": None if results.get("seeds") is None else np.asarray(results["seeds"]).tolist(),
            }
        )
        return results

    lewm_eval.evaluate_from_dataset_with_goal_pixels = evaluate_and_audit
    lewm_eval.swm.policy.AutoCostModel = lambda _policy: model
    lewm_eval.img_transform = lambda _cfg: RawCHWTensor()

    arm_root.mkdir(parents=True, exist_ok=True)
    result_path = arm_root / "result.txt"
    # This path belongs exclusively to this arm.  Truncation makes interrupted
    # reruns idempotent instead of appending a second, ambiguous result block.
    result_path.write_text("", encoding="utf-8")
    policy_placeholder = arm_root / "jepa_wm_medium_adapter"
    overrides = [
        f"policy={policy_placeholder}",
        "eval.dataset_name=ogbench/temporal_pointmaze_medium_topdown",
        f"seed={args.eval_seed}",
        f"eval.num_eval={args.episodes}",
        "eval.action_noise_std=0.0",
        "eval.eval_budget=50",
        "eval.goal_offset_steps=25",
        "eval.sample_remaining_min=25",
        "eval.sample_remaining_max=50",
        "eval.save_video=false",
        "eval.context_cost_weight=1.0",
        "eval.cost_mode=latent",
        "eval.pi_ltc_scale_intervention_mode=none",
        "world.render_transform=flip_ud",
        "world.show_target=false",
        "world.coordinate_offset=[1.2,1.2]",
        "solver.num_samples=256",
        "solver.n_steps=16",
        "solver.topk=64",
        "plan_config.horizon=6",
        "plan_config.receding_horizon=6",
        "plan_config.action_block=5",
        "output.filename=result.txt",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(lewm_repo / "config/eval")):
        cfg = compose(config_name="pointmaze_topdown_medium", overrides=overrides)

    print(
        "[Medium protocol] "
        f"owner={args.owner} arm={args.arm} seed={args.eval_seed} episodes={args.episodes} "
        "H6/R6/action_block5 CEM16/256/64 budget50 offset25"
    )
    lewm_eval.run.__wrapped__(cfg)
    if len(captured.get("episode_successes", [])) != args.episodes:
        raise RuntimeError("evaluator did not return the requested number of episode outcomes")

    scale_audit = model.planner_scale_audit
    effective_scale = None if scale_audit is None else float(scale_audit["effective_scale"])
    audit = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "ogbench_temporal_pointmaze_medium_stablewm_exact_v1",
        "training": False,
        "owner": args.owner,
        "arm": args.arm,
        "effective_scale": effective_scale,
        "scale_audit": scale_audit,
        "checkpoint": {
            "path": str(checkpoint_path),
            "size": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
        },
        "source_h5": {
            "path": str(source_h5),
            "size": source_h5.stat().st_size,
            "mtime_ns": source_h5.stat().st_mtime_ns,
        },
        "repository_commit": _git_commit(repository),
        "lewm_repository_commit": _git_commit(lewm_repo),
        "evaluation": {
            "task": "Temporal PointMaze-Medium",
            "eval_seed": args.eval_seed,
            "episodes": args.episodes,
            "action_noise_std": 0.0,
            "eval_budget": 50,
            "goal_offset_steps": 25,
            "sample_remaining_min": 25,
            "sample_remaining_max": 50,
            "history_size": 1,
            "horizon": 6,
            "receding_horizon": 6,
            "action_block": 5,
            "cem_steps": 16,
            "cem_samples": 256,
            "cem_topk": 64,
            "cost": "visual latent squared L2 endpoint",
            "success": "simulator distance <= 0.5",
            "maze_spec": "medium",
            "coordinate_offset": [1.2, 1.2],
            "render_transform": "flip_ud",
        },
        "adapter_preflight": preflight,
        "encoding_cache_audit": model.encoding_cache_audit,
        "results": captured,
    }
    _write_json_atomic(arm_root / "arm_audit.json", audit)
    print(
        f"[arm complete] {args.arm}: SR={captured['success_rate']:.1f}% "
        f"({sum(captured['episode_successes'])}/{args.episodes})"
    )
    print(f"[saved] {arm_root / 'arm_audit.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    raise SystemExit(main())
