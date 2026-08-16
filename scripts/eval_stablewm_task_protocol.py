#!/usr/bin/env python3
"""Evaluate PushT/Cube JEPA-WM checkpoints with the exact LEWM protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - register PushT Blosc decoding
import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from app.vjepa_wm.modelcustom.stablewm_pointmaze_cost import (  # noqa: E402
    build_stablewm_pointmaze_cost,
)
from scripts.generate_stablewm_task_training_configs import TASKS  # noqa: E402
from scripts.stablewm_checkpoint_selection import (  # noqa: E402
    expected_checkpoint_schedule,
)


TASK_PROTOCOLS = {
    "pusht": {
        "config": "pusht",
        "dataset": "pusht_expert_train",
        "environment": "swm/PushT-v1",
        "success": "translation state error <20 and wrapped angular error <pi/9",
        "state_columns": ("state", "proprio"),
    },
    "cube": {
        "config": "cube",
        "dataset": "ogbench/cube_single_expert",
        "environment": "swm/OGBCube-v0 single",
        "success": "object-to-goal distance <=0.04 m",
        "state_columns": (
            "qpos",
            "qvel",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _tracked_dirty(path: Path) -> bool:
    return bool(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ],
            text=True,
        ).strip()
    )


def _update_hash(digest, name: str, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(name.encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class RawCHWTensor:
    def __call__(self, value):
        return torch.as_tensor(value)


def _planner_preflight(model, source_h5: Path, eval_seed: int) -> dict:
    metadata = model.source_metadata
    context_row = int(metadata.episode_offsets[0])
    goal_row = context_row + 25
    if goal_row >= metadata.episode_offsets[0] + metadata.episode_lengths[0]:
        raise RuntimeError("first source episode cannot supply the 25-step preflight goal")
    with h5py.File(source_h5, "r", swmr=True) as source:
        pixels = np.asarray(source["pixels"][[context_row, goal_row]], dtype=np.uint8)
        proprio = metadata.load_proprio(
            source, np.asarray([context_row], dtype=np.int64)
        )

    context = torch.from_numpy(pixels[0]).permute(2, 0, 1)[None, None]
    goal = torch.from_numpy(pixels[1]).permute(2, 0, 1)[None, None]
    proprio = torch.from_numpy(proprio)[None]

    def costs(candidates, chunk_size):
        samples = candidates.shape[1]
        info = {
            "pixels": context.unsqueeze(1).expand(1, samples, *context.shape[1:]),
            "goal": goal.unsqueeze(1).expand(1, samples, *goal.shape[1:]),
            "proprio": proprio.unsqueeze(1).expand(1, samples, *proprio.shape[1:]),
        }
        requested = model.candidate_chunk_size
        try:
            model.candidate_chunk_size = chunk_size
            return model.get_cost(info, candidates)
        finally:
            model.candidate_chunk_size = requested

    generator = torch.Generator(device=model.model.device).manual_seed(eval_seed)
    candidates = torch.randn(
        1,
        256,
        5,
        model.model.action_dim,
        generator=generator,
        device=model.model.device,
    )
    candidates[:, 0] = 0
    candidate_hash = hashlib.sha256(candidates.cpu().numpy().tobytes()).hexdigest()

    small = candidates[:, :8]
    small_unchunked = costs(small, 8)
    small_chunked = costs(small, 2)
    full_reference = costs(candidates, 64)
    full_chunked = costs(candidates, model.candidate_chunk_size)
    small_error = float((small_unchunked - small_chunked).abs().max().cpu())
    full_error = float((full_reference - full_chunked).abs().max().cpu())
    for label, left, right, error in (
        ("small unchunked/chunked", small_unchunked, small_chunked, small_error),
        ("full 256 reference/chunked", full_reference, full_chunked, full_error),
    ):
        if not torch.allclose(left, right, rtol=1.0e-5, atol=1.0e-3):
            raise RuntimeError(f"{label} planner costs differ: max_abs={error}")
    if full_chunked.shape != (1, 256) or not torch.isfinite(full_chunked).all():
        raise RuntimeError("full 256-candidate planner preflight is non-finite or malformed")
    return {
        "passed": True,
        "candidate_count": 256,
        "horizon": 5,
        "action_block_dim": model.model.action_dim,
        "candidate_sha256": candidate_hash,
        "small_unchunked_vs_chunked_max_abs": small_error,
        "full_reference64_vs_requested_max_abs": full_error,
        "requested_chunk_size": model.candidate_chunk_size,
    }


def _checkpoint_audit(
    path: Path,
    *,
    owner: str,
    expected_commit: str,
    source_sha256: str,
    sidecar_sha256: str,
    expected_optimizer_steps_per_epoch: int,
    expected_optimizer_step_budget: int,
    expected_completed_passes: int | None = None,
    smoke: bool = False,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    predictor = checkpoint.get("predictor") or {}
    scale_keys = [key for key in predictor if key.endswith("planner_input_scale.log_scale")]
    if owner == "pi" and len(scale_keys) != 1:
        raise RuntimeError(f"PI checkpoint scale keys={scale_keys}")
    if owner == "vanilla" and scale_keys:
        raise RuntimeError(f"vanilla checkpoint unexpectedly has scale keys={scale_keys}")
    provenance = checkpoint.get("training_provenance")
    initialization_sha256 = checkpoint.get("common_trainable_initialization_sha256")
    if not smoke:
        if not isinstance(provenance, dict):
            raise RuntimeError("checkpoint lacks strict training provenance")
        if provenance.get("repository_commit") != expected_commit:
            raise RuntimeError("checkpoint repository commit differs from expected training commit")
        if provenance.get("source_h5", {}).get("sha256") != source_sha256:
            raise RuntimeError("checkpoint source SHA-256 differs")
        if provenance.get("sidecar_h5", {}).get("sha256") != sidecar_sha256:
            raise RuntimeError("checkpoint sidecar SHA-256 differs")
        if checkpoint.get("optimizer_steps_per_epoch") != expected_optimizer_steps_per_epoch:
            raise RuntimeError("checkpoint optimizer-step schedule differs")
        if checkpoint.get("optimizer_step_budget") != expected_optimizer_step_budget:
            raise RuntimeError("checkpoint optimizer-step budget differs")
        try:
            expected_schedule = expected_checkpoint_schedule(
                optimizer_steps_per_pass=expected_optimizer_steps_per_epoch,
                configured_optimizer_step_budget=expected_optimizer_step_budget,
                expected_completed_passes=expected_completed_passes,
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        completed_passes = expected_schedule["epoch"]
        partial_pass_steps = expected_schedule["optimizer_step_in_epoch"]
        expected_total_optimizer_steps = expected_schedule["total_optimizer_steps"]
        expected_training_complete = expected_schedule["training_complete"]
        if checkpoint.get("total_optimizer_steps") != expected_total_optimizer_steps:
            raise RuntimeError("checkpoint total optimizer-step count differs")
        if (checkpoint.get("training_complete") is True) != expected_training_complete:
            raise RuntimeError("checkpoint training-complete state differs")
        if int(checkpoint.get("epoch", -1)) != completed_passes:
            raise RuntimeError("checkpoint completed-pass count differs")
        if int(checkpoint.get("optimizer_step_in_epoch", -1)) != partial_pass_steps:
            raise RuntimeError("checkpoint partial-pass optimizer-step count differs")
        if (
            not isinstance(initialization_sha256, str)
            or len(initialization_sha256) != 64
            or any(character not in "0123456789abcdef" for character in initialization_sha256)
        ):
            raise RuntimeError("checkpoint lacks a valid common initialization fingerprint")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "epoch": int(checkpoint["epoch"]),
        "optimizer_steps_per_epoch": int(checkpoint.get("optimizer_steps_per_epoch", 0)),
        "total_optimizer_steps": int(checkpoint.get("total_optimizer_steps", 0)),
        "optimizer_step_budget": int(checkpoint.get("optimizer_step_budget", 0)),
        "optimizer_step_in_epoch": int(checkpoint.get("optimizer_step_in_epoch", 0)),
        "training_complete": checkpoint.get("training_complete") is True,
        "selection_policy": ("configured_final_horizon" if smoke else expected_schedule["selection_policy"]),
        "selected_completed_passes": int(checkpoint["epoch"]),
        "scale_keys": scale_keys,
        "training_provenance": provenance,
        "common_trainable_initialization_sha256": initialization_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--lewm-repo", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="jepa-latest.pth.tar")
    parser.add_argument("--owner", choices=("pi", "vanilla"), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--candidate-chunk-size", type=int, default=32)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-training-commit")
    parser.add_argument("--expected-lewm-commit", required=True)
    parser.add_argument("--expected-completed-passes", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-complete", action="store_true")
    args = parser.parse_args()
    expected_training_commit = args.expected_training_commit or args.expected_commit

    if args.owner == "pi" and args.arm not in {"learned", "identity", "fixed06", "fixed04"}:
        raise SystemExit("[STOP] invalid PI arm")
    if args.owner == "vanilla" and args.arm != "vanilla_stepmatched":
        raise SystemExit("[STOP] vanilla owner requires arm=vanilla_stepmatched")
    if args.episodes != (1 if args.smoke else 50):
        raise SystemExit("[STOP] smoke requires 1 episode; exact evaluation requires 50")
    if args.candidate_chunk_size <= 0:
        raise SystemExit("[STOP] candidate chunk size must be positive")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("[STOP] exact StableWM evaluation requires an available CUDA device")

    repository = REPOSITORY
    lewm_repo = args.lewm_repo.resolve()
    source_h5 = args.source_h5.resolve()
    training_config_path = args.training_config.resolve()
    checkpoint_path = args.checkpoint_dir.resolve() / args.checkpoint
    if _git_commit(repository) != args.expected_commit or _tracked_dirty(repository):
        raise SystemExit("[STOP] JEPA evaluator repository commit/dirty state differs")
    if _git_commit(lewm_repo) != args.expected_lewm_commit or _tracked_dirty(lewm_repo):
        raise SystemExit("[STOP] LEWM evaluator repository commit/dirty state differs")
    for required in (
        source_h5,
        training_config_path,
        checkpoint_path,
        lewm_repo / "eval.py",
        lewm_repo / f"config/eval/{TASK_PROTOCOLS[args.task]['config']}.yaml",
    ):
        if not required.is_file():
            raise SystemExit(f"[STOP] missing required input: {required}")
    preflight_audit = json.loads(args.preflight_audit.read_text(encoding="utf-8"))
    source_audit = preflight_audit.get("source", {})
    source_stat = source_h5.stat()
    config_audit = preflight_audit.get("training_configs", {}).get(args.owner, {})
    if (
        preflight_audit.get("status") != "PASS"
        or preflight_audit.get("task") != args.task
        or preflight_audit.get("repository_commit") != expected_training_commit
        or source_audit.get("path") != str(source_h5)
        or source_audit.get("bytes") != source_stat.st_size
        or source_audit.get("mtime_ns") != source_stat.st_mtime_ns
        or source_audit.get("sha256") != args.source_sha256
        or config_audit.get("path") != str(training_config_path)
        or config_audit.get("sha256") != _sha256(training_config_path)
        or preflight_audit.get("planner", {}).get("sidecar_sha256")
        != args.sidecar_sha256
    ):
        raise SystemExit(
            "[STOP] data/training preflight audit or source-file state does not match evaluation"
        )

    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    transition_optimization = training_config["optimization"]["transition_model"]
    expected_optimizer_steps_per_epoch = int(
        preflight_audit["split"]["optimizer_steps_per_pass"]
    )
    if (
        int(transition_optimization["expected_optimizer_steps_per_epoch"])
        != expected_optimizer_steps_per_epoch
    ):
        raise SystemExit("[STOP] training config optimizer steps/pass differs from preflight")
    expected_optimizer_step_budget = int(
        transition_optimization["total_optimizer_steps"]
    )
    checkpoint_audit = _checkpoint_audit(
        checkpoint_path,
        owner=args.owner,
        expected_commit=expected_training_commit,
        source_sha256=args.source_sha256,
        sidecar_sha256=args.sidecar_sha256,
        expected_optimizer_steps_per_epoch=expected_optimizer_steps_per_epoch,
        expected_optimizer_step_budget=expected_optimizer_step_budget,
        expected_completed_passes=args.expected_completed_passes,
        smoke=args.smoke,
    )
    arm_root = (
        args.output_root.resolve()
        / args.task
        / ("smoke" if args.smoke else f"seed{args.eval_seed}")
        / args.arm
    )
    existing_audit_path = arm_root / "arm_audit.json"
    if args.reuse_complete and existing_audit_path.is_file():
        existing = json.loads(existing_audit_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("task") == args.task
            and existing.get("owner") == args.owner
            and existing.get("arm") == args.arm
            and existing.get("eval_seed") == args.eval_seed
            and existing.get("checkpoint", {}).get("sha256")
            == checkpoint_audit["sha256"]
            and existing.get("source_h5", {}).get("sha256") == args.source_sha256
            and existing.get("repository_commit") == args.expected_commit
            and existing.get("training_repository_commit")
            == expected_training_commit
            and existing.get("lewm_repository_commit") == args.expected_lewm_commit
            and len(existing.get("results", {}).get("episode_successes", []))
            == args.episodes
        ):
            print(f"[reuse complete] {existing_audit_path}")
            return 0
        raise SystemExit("[STOP] existing arm audit is not reusable; preserve and inspect it")
    model = build_stablewm_pointmaze_cost(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint=args.checkpoint,
        training_config=training_config,
        source_h5=source_h5,
        arm=args.arm,
        owner=args.owner,
        candidate_chunk_size=args.candidate_chunk_size,
        device=args.device,
        strict_checkpoint=not args.smoke,
    ).to(args.device)
    torch.cuda.reset_peak_memory_stats()
    adapter_preflight = _planner_preflight(model, source_h5, args.eval_seed)

    os.environ["STABLEWM_HOME"] = str(
        source_h5.parent if args.task == "pusht" else source_h5.parents[1]
    )
    sys.path.insert(0, str(lewm_repo))
    import eval as lewm_eval  # noqa: PLC0415

    captured: dict = {}
    original_evaluate = lewm_eval.evaluate_from_dataset_with_options

    def evaluate_and_audit(world, dataset, **kwargs):
        episodes = np.asarray(kwargs["episodes_idx"], dtype=np.int64)
        starts = np.asarray(kwargs["start_steps"], dtype=np.int64)
        metadata = model.source_metadata
        position_by_id = {
            int(physical_id): position
            for position, physical_id in enumerate(metadata.episode_ids)
        }
        positions = np.asarray([position_by_id[int(value)] for value in episodes])
        context_rows = metadata.episode_offsets[positions] + starts
        goal_rows = context_rows + 25
        state_digest = hashlib.sha256()
        _update_hash(state_digest, "episodes", episodes)
        _update_hash(state_digest, "starts", starts)
        _update_hash(state_digest, "context_rows", context_rows)
        _update_hash(state_digest, "goal_rows", goal_rows)
        with h5py.File(source_h5, "r", swmr=True) as source:
            for column in TASK_PROTOCOLS[args.task]["state_columns"]:
                _update_hash(state_digest, f"context/{column}", source[column][context_rows])
                _update_hash(state_digest, f"goal/{column}", source[column][goal_rows])
        results = original_evaluate(world, dataset, **kwargs)
        captured.update(
            {
                "episodes_idx": episodes.tolist(),
                "start_steps": starts.tolist(),
                "context_rows": context_rows.tolist(),
                "goal_rows": goal_rows.tolist(),
                "paired_start_sha256": hashlib.sha256(
                    np.stack((episodes, starts, context_rows, goal_rows), axis=1).tobytes()
                ).hexdigest(),
                "environment_state_sha256": state_digest.hexdigest(),
                "episode_successes": [bool(value) for value in results["episode_successes"]],
                "success_rate": float(results["success_rate"]),
                "seeds": None
                if results.get("seeds") is None
                else np.asarray(results["seeds"]).tolist(),
            }
        )
        return results

    lewm_eval.evaluate_from_dataset_with_options = evaluate_and_audit
    lewm_eval.swm.policy.AutoCostModel = lambda _policy: model
    lewm_eval.img_transform = lambda _cfg: RawCHWTensor()

    original_cem = lewm_eval.swm.solver.CEMSolver
    solver_instances = []

    class AuditedCEMSolver(original_cem):
        def __init__(self, *solver_args, **solver_kwargs):
            super().__init__(*solver_args, **solver_kwargs)
            self.generator_state_digest = hashlib.sha256()
            self.generator_state_calls = 0
            solver_instances.append(self)

        def solve(self, *solver_args, **solver_kwargs):
            state = self.torch_gen.get_state().cpu().numpy()
            _update_hash(
                self.generator_state_digest,
                f"call/{self.generator_state_calls}",
                state,
            )
            self.generator_state_calls += 1
            return super().solve(*solver_args, **solver_kwargs)

    lewm_eval.swm.solver.CEMSolver = AuditedCEMSolver

    arm_root.mkdir(parents=True, exist_ok=True)
    (arm_root / "result.txt").write_text("", encoding="utf-8")
    policy_placeholder = arm_root / "jepa_wm_adapter"
    overrides = [
        f"policy={policy_placeholder}",
        f"eval.dataset_name={TASK_PROTOCOLS[args.task]['dataset']}",
        f"seed={args.eval_seed}",
        f"eval.num_eval={args.episodes}",
        "eval.action_noise_std=0.0",
        "eval.eval_budget=50",
        "eval.goal_offset_steps=25",
        "eval.context_cost_weight=1.0",
        "dataset.keys_to_cache=[action]",
        "solver.num_samples=256",
        "solver.n_steps=16",
        "solver.topk=64",
        "plan_config.horizon=5",
        "plan_config.receding_horizon=5",
        "plan_config.action_block=5",
        "output.filename=result.txt",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(lewm_repo / "config/eval")):
        cfg = compose(config_name=TASK_PROTOCOLS[args.task]["config"], overrides=overrides)
    if list(cfg.dataset.keys_to_cache) != ["action"]:
        raise RuntimeError("evaluator must leave image/proprio normalization to JEPA-WM")
    lewm_eval.run.__wrapped__(cfg)
    if len(captured.get("episode_successes", [])) != args.episodes:
        raise RuntimeError("evaluator returned the wrong episode count")
    if len(solver_instances) != 1:
        raise RuntimeError(f"expected one audited CEM solver, got {len(solver_instances)}")
    solver = solver_instances[0]
    candidate_rng_audit = {
        "generator_state_sequence_sha256": solver.generator_state_digest.hexdigest(),
        "solve_calls": solver.generator_state_calls,
        "seed": args.eval_seed,
        "cem_steps": 16,
        "cem_samples": 256,
        "cem_topk": 64,
    }

    audit = {
        "schema_version": 1,
        "status": "complete",
        "protocol": f"{args.task}_stablewm_exact_jepa_v1",
        "training": False,
        "task": args.task,
        "owner": args.owner,
        "arm": args.arm,
        "eval_seed": args.eval_seed,
        "effective_scale": None
        if model.planner_scale_audit is None
        else float(model.planner_scale_audit["effective_scale"]),
        "scale_audit": model.planner_scale_audit,
        "checkpoint": checkpoint_audit,
        "source_h5": {
            "path": str(source_h5),
            "bytes": source_h5.stat().st_size,
            "sha256": args.source_sha256,
        },
        "repository_commit": _git_commit(repository),
        "training_repository_commit": expected_training_commit,
        "lewm_repository_commit": _git_commit(lewm_repo),
        "evaluation": {
            "environment": TASK_PROTOCOLS[args.task]["environment"],
            "episodes": args.episodes,
            "action_noise_std": 0.0,
            "eval_budget": 50,
            "goal_offset_steps": 25,
            "history_size": 1,
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
            "cem_steps": 16,
            "cem_samples": 256,
            "cem_topk": 64,
            "cost": "visual latent squared L2 endpoint",
            "lewm_standardized_columns": ["action"],
            "success": TASK_PROTOCOLS[args.task]["success"],
        },
        "adapter_preflight": adapter_preflight,
        "encoding_cache_audit": model.encoding_cache_audit,
        "candidate_chunk_audit": model.candidate_chunk_audit,
        "candidate_rng_audit": candidate_rng_audit,
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "results": captured,
    }
    _write_json_atomic(arm_root / "arm_audit.json", audit)
    print(
        f"[arm complete] task={args.task} arm={args.arm} seed={args.eval_seed} "
        f"SR={captured['success_rate']:.1f}%"
    )
    print(f"[saved] {arm_root / 'arm_audit.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    raise SystemExit(main())
