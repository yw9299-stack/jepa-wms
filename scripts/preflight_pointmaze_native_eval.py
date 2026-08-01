#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 10):
        raise SystemExit(f"[STOP] native PointMaze requires Python 3.10, found {sys.version}")

    import gym
    import d4rl
    import mujoco_py

    from evals.simu_env_planning.envs.init import make_env

    config = OmegaConf.create(yaml.safe_load(args.config.read_text()))
    if config.task_specification.task != "maze-base":
        raise SystemExit(f"[STOP] unexpected task: {config.task_specification.task}")
    model_frameskip = int(config.model_kwargs.data.custom.frameskip)
    if int(config.frameskip) != model_frameskip:
        raise SystemExit(
            f"[STOP] evaluation frameskip={config.frameskip} differs from model data frameskip={model_frameskip}"
        )

    env = None
    try:
        env = make_env(config)
        initial, goal = env.sample_random_init_goal_states(int(config.meta.seed))
        goal_obs, goal_info = env.prepare(int(config.meta.seed), goal)
        initial_obs, initial_info = env.prepare(int(config.meta.seed), initial)
        action = torch.zeros(env.action_space.shape, dtype=torch.float32)
        next_obs, _, _, _, next_info = env.step(action)

        for name, observation in (
            ("goal", goal_obs),
            ("initial", initial_obs),
            ("next", next_obs),
        ):
            if not isinstance(observation, torch.Tensor):
                raise RuntimeError(f"{name} observation is not a tensor: {type(observation)!r}")
            if tuple(observation.shape[-3:]) != (3, 224, 224):
                raise RuntimeError(f"{name} observation shape={tuple(observation.shape)}")
            if not torch.isfinite(observation.float()).all():
                raise RuntimeError(f"{name} observation contains non-finite values")

        audit = {
            "status": "passed",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gym": gym.__version__,
            "mujoco_py": mujoco_py.__version__,
            "task": str(config.task_specification.task),
            "seed": int(config.meta.seed),
            "frameskip": int(config.frameskip),
            "initial_state": np.asarray(initial).tolist(),
            "goal_state": np.asarray(goal).tolist(),
            "initial_observation_shape": list(initial_obs.shape),
            "next_observation_shape": list(next_obs.shape),
            "initial_position": np.asarray(initial_info["state"][:2]).tolist(),
            "next_position": np.asarray(next_info["state"][:2]).tolist(),
        }
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.audit.with_suffix(args.audit.suffix + ".tmp")
        temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.audit)
        print(
            "[native environment preflight] PASS "
            f"task={audit['task']} image_shape={audit['initial_observation_shape']} "
            f"mujoco_py={audit['mujoco_py']}"
        )
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
