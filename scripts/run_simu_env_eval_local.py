#!/usr/bin/env python3

import argparse

import torch.distributed as dist
import yaml

from evals.simu_env_planning.eval import main as run_eval
from src.utils.distributed import init_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    if not dist.is_initialized():
        init_distributed(rank_and_world_size=(0, 1))
    try:
        run_eval(config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
