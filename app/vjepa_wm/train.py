# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass
import time
from collections import defaultdict

import imageio
import lpips as lpips_lib
import matplotlib
import numpy as np
import submitit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from app.plan_common.datasets.planner_landscape_h5 import (
    PlannerLandscapeH5Dataset,
    split_planner_groups_by_physical_episode,
)
from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.datasets.utils import init_data
from app.plan_common.models.wm_heads import (
    WorldModelPoseReadoutHead,
    WorldModelRewardReadoutHead,
    WorldModelViTImageHead,
)
from app.vjepa_wm.planner_landscape import (
    install_planner_scale_gradient,
    planner_input_scale,
    planner_landscape_result,
)
from app.vjepa_wm.utils import (
    build_plan_eval_args,
    build_unroll_decode_eval_args,
    clean_state_dict,
    init_opt,
    init_video_model,
    load_checkpoint,
)
from app.vjepa_wm.video_wm import VideoWM
from evals.main_distributed import launch_evals_with_parsed_args as launch_evals
from src.datasets.utils.utils import get_dataset_paths
from src.utils.cluster import slurm_account_partition_and_qos
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.yaml_utils import convert_to_dict_recursive, dump_yaml, expand_env_vars

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
DEFAULT_EVAL_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def _common_trainable_initialization_sha256(
    predictor,
    action_encoder,
    proprio_encoder,
):
    """Fingerprint matched trainable state while excluding PI's extra scalar."""

    digest = hashlib.sha256()
    for module_name, module in (
        ("predictor", predictor),
        ("action_encoder", action_encoder),
        ("proprio_encoder", proprio_encoder),
    ):
        if module is None:
            continue
        for key, value in sorted(module.state_dict().items()):
            if key.endswith("planner_input_scale.log_scale"):
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update(module_name.encode("utf-8"))
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    args = expand_env_vars(args)
    folder = args.get("folder")
    checkpoint_folder = args.get("checkpoint_folder", folder)
    os.makedirs(checkpoint_folder, exist_ok=True)
    # -- META
    cfgs_meta = args.get("meta")
    strict_provenance = bool(cfgs_meta.get("strict_provenance", False))
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    load_opt_scale_epoch = cfgs_meta.get("load_opt_scale_epoch", False)
    freeze_encoder = cfgs_meta.get("freeze_encoder", True)
    r_file = cfgs_meta.get("read_checkpoint", None)
    pretrained_path = cfgs_meta.get("pretrained_path", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)

    eval_freq = cfgs_meta.get("eval_freq", DEFAULT_EVAL_FREQ)
    plan_only_eval_mode = cfgs_meta.get("plan_only_eval_mode", False)
    unroll_decode_eval_only_mode = cfgs_meta.get("unroll_decode_eval_only_mode", False)
    light_eval_only_mode = cfgs_meta.get("light_eval_only_mode", False)
    skip_planning_eval = cfgs_meta.get("skip_planning_eval", False)

    # -- LIGHT EVALS (keep as subconfigs, extract only frequently-checked flags)
    cfgs_data_traj_rollout_eval = cfgs_meta.get("data_traj_rollout_eval", {})
    cfgs_energy_landscape_eval = cfgs_meta.get("energy_landscape_eval", {})
    do_data_traj_rollout_eval = cfgs_data_traj_rollout_eval.get("do_data_traj_rollout_eval", False)
    do_energy_landscape_eval = cfgs_energy_landscape_eval.get("do_energy_landscape_eval", False)
    data_traj_decode_gt = cfgs_data_traj_rollout_eval.get("data_traj_decode_gt", False)

    light_eval_freq = cfgs_meta.get("light_eval_freq", 100)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    quick_debug = cfgs_meta.get("quick_debug", False)
    canary_optimizer_steps = int(cfgs_meta.get("canary_optimizer_steps", 0) or 0)
    if canary_optimizer_steps < 0:
        raise ValueError("meta.canary_optimizer_steps must be non-negative")
    stop_after_complete_passes = int(
        os.environ.get("PI_LTC_STOP_AFTER_COMPLETE_PASSES", "0") or 0
    )
    if stop_after_complete_passes < 0:
        raise ValueError(
            "PI_LTC_STOP_AFTER_COMPLETE_PASSES must be non-negative"
        )
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"⚙️  Using dtype: {which_dtype}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- EVALS
    cfgs_plan_evals = args.get("evals", None)
    cfgs_unroll_decode_evals = args.get("unroll_decode_evals", None)

    # -- MODEL (extract only fields needed outside init_video_model)
    cfgs_model = args.get("model")

    # Extract heads config
    cfgs_heads = cfgs_model.get("heads_cfg", {})
    heads_architectures = cfgs_heads.get("architectures", {})
    pretrain_dec_path = cfgs_heads.get("pretrain_dec_path", None)
    new_path_heads = cfgs_heads.get("new_path_heads", {})

    # Fields needed for training loop
    rollout_cfg = cfgs_model.get("rollout_cfg", {})
    rollout_steps = rollout_cfg.get("rollout_steps", 0)
    train_rollout_prefixes = rollout_cfg.get("train_rollout_prefixes", "random")
    rollout_stop_gradient = rollout_cfg.get("rollout_stop_gradient", True)
    ctxt_window_train_rollout = rollout_cfg.get("ctxt_window_train_rollout", 8)
    do_parallel_rollout = rollout_cfg.get("do_parallel_rollout", False)
    do_sequential_rollout = rollout_cfg.get("do_sequential_rollout", True)
    sampling_rollout = rollout_cfg.get("sampling_rollout", False)
    prepend_gt_rollout_parallel = rollout_cfg.get("prepend_gt", False)

    sampling_scheduler_cfg = rollout_cfg.get("sampling_scheduler", {})
    sampling_scheduler_type = sampling_scheduler_cfg.get("type", "linear")
    sampling_scheduler_start = sampling_scheduler_cfg.get("start", 0.0)
    sampling_scheduler_end = sampling_scheduler_cfg.get("end", 0.0)

    # Derived values needed for dimensions (computed values used across the code)
    action_tokens = cfgs_model.get("action_encoder", {}).get("action_tokens", 1)
    proprio_tokens = cfgs_model.get("proprio_encoder", {}).get("proprio_tokens", 1)
    action_emb_dim = cfgs_model.get("action_encoder", {}).get("action_emb_dim", 0)
    proprio_emb_dim = cfgs_model.get("proprio_encoder", {}).get("proprio_emb_dim", 0)
    use_proprio = proprio_tokens > 0 or proprio_emb_dim > 0
    use_action = action_tokens > 0 or action_emb_dim > 0
    tubelet_size_enc = cfgs_model.get("tubelet_size_enc", 2)

    cfgs_wm_encoding = cfgs_model.get("wm_encoding", {})

    if cfgs_wm_encoding.get("dup_image", False):
        assert tubelet_size_enc == 1, "Batchify video only works with tubelet_size_enc=1"

    # -- DATA (extract only fields needed outside init_data)
    cfgs_data = args.get("data")
    cfgs_validation = cfgs_data.get("validation", {})
    cfgs_loader = cfgs_data.get("loader", {})
    cfgs_custom = cfgs_data.get("custom", {})
    cfgs_droid = cfgs_data.get("droid", {})

    # Compute dataset paths. ``paths`` is the explicit local-file path used by
    # reproducible StableWorldModel HDF5 training and bypasses registry lookup.
    datasets = cfgs_data.get("datasets", [])
    explicit_dataset_paths = cfgs_data.get("paths", None)
    datasets_weights = cfgs_data.get("datasets_weights", None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(datasets), "Must have one sampling weight specified for each dataset"

    dataset_type = cfgs_data.get("dataset_type", "custom")
    if explicit_dataset_paths is not None:
        dataset_paths = list(explicit_dataset_paths)
    elif dataset_type.lower() == "mixed_dataset":
        dataset_paths = datasets
    else:
        dataset_paths = get_dataset_paths(datasets)

    val_datasets = cfgs_validation.get("val_datasets", [])
    val_dataset_paths = get_dataset_paths(val_datasets) if val_datasets else None

    # val_datasets_1 subconfig (for second validation set)
    val_datasets_1 = cfgs_validation.get("val_datasets_1", None)
    val_datasets_1_paths = None
    if val_datasets_1 is not None:
        val_datasets_1_paths = get_dataset_paths(val_datasets_1.get("names"))

    # Fields used outside init_data
    frameskip = cfgs_custom.get("frameskip", True)
    action_skip = cfgs_custom.get("action_skip", 1)
    state_skip = cfgs_custom.get("state_skip", 1)
    img_size = cfgs_data.get("img_size", 256)
    num_workers = cfgs_loader.get("num_workers", 1)
    filter_first_episodes = cfgs_custom.get("filter_first_episodes", None)
    val_viz_rank0_loader = cfgs_validation.get("val_viz_rank0_loader", False)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")

    # -- LOSS
    cfgs_loss = args.get("loss")

    # -- OPTIMIZATION (simplify main_optimizer logic)
    cfgs_opt = args.get("optimization")
    train_heads = cfgs_opt["train_heads"]
    cfgs_planner_identified = args.get("planner_identified", {})
    planner_identified_enabled = bool(cfgs_planner_identified.get("enabled", False))

    training_provenance = None
    if strict_provenance:
        required_environment = (
            "PI_LTC_REPOSITORY_COMMIT",
            "PI_LTC_SOURCE_SHA256",
            "PI_LTC_SIDECAR_SHA256",
        )
        missing_environment = [name for name in required_environment if not os.environ.get(name)]
        if missing_environment:
            raise ValueError(
                "strict provenance requires environment variables: "
                + ", ".join(missing_environment)
            )
        content_hash_mode = os.environ.get("PI_LTC_CONTENT_HASH_MODE", "sha256")
        if content_hash_mode not in {"sha256", "not_scanned_user_confirmed"}:
            raise ValueError(f"unsupported strict provenance content-hash mode: {content_hash_mode}")
        repository_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        repository_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain=v1", "--untracked-files=no"],
                text=True,
            ).strip()
        )
        if repository_commit != os.environ["PI_LTC_REPOSITORY_COMMIT"]:
            raise ValueError("strict provenance repository commit differs from the launcher")
        if repository_dirty:
            raise ValueError("strict provenance requires a clean tracked worktree")
        if dataset_type != "stablewm_h5" or len(dataset_paths) != 1:
            raise ValueError("strict provenance requires exactly one StableWM HDF5 source")
        source_path = Path(dataset_paths[0]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        sidecar_value = cfgs_planner_identified.get("sidecar_h5")
        sidecar_path = Path(sidecar_value).resolve() if sidecar_value else None
        training_provenance = {
            "repository_commit": repository_commit,
            "repository_tracked_dirty": repository_dirty,
            "content_hash_mode": content_hash_mode,
            "source_h5": {
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "mtime_ns": source_path.stat().st_mtime_ns,
                "sha256": os.environ["PI_LTC_SOURCE_SHA256"],
            },
            "sidecar_h5": {
                "path": str(sidecar_path) if sidecar_path is not None else None,
                "bytes": sidecar_path.stat().st_size if sidecar_path is not None else None,
                "mtime_ns": (
                    sidecar_path.stat().st_mtime_ns if sidecar_path is not None else None
                ),
                "sha256": os.environ["PI_LTC_SIDECAR_SHA256"],
            },
            "resolved_config": convert_to_dict_recursive(args),
        }

    main_optimizer = cfgs_opt["main_optimizer"]
    if main_optimizer == "transition_model":
        num_epochs = cfgs_opt["transition_model"]["num_epochs"]
        ipe = cfgs_opt["transition_model"]["iterations_per_epoch"]
        configured_optimizer_step_budget = cfgs_opt["transition_model"].get(
            "total_optimizer_steps"
        )
        gradient_accumulation_steps = int(cfgs_opt["transition_model"].get("gradient_accumulation_steps", 1))
        train_predictor = True
        train_heads_on_predictor = False
    else:  # image_head | state_head | reward_head
        num_epochs = cfgs_opt["heads"][main_optimizer]["num_epochs"]
        ipe = cfgs_opt["heads"][main_optimizer]["iterations_per_epoch"]
        train_predictor = cfgs_opt["heads"]["train_predictor"]
        train_heads_on_predictor = cfgs_opt["heads"]["train_heads_on_predictor"]
        gradient_accumulation_steps = 1
        configured_optimizer_step_budget = None
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if planner_identified_enabled and (main_optimizer != "transition_model" or train_heads):
        raise ValueError("planner-identified training requires transition_model as the sole optimizer")

    # -- LOGGING
    cfgs_logging = args.get("logging")
    tag = cfgs_logging.get("write_tag", "jepa")
    latest_format = cfgs_logging.get("latest_format", "pth.tar")
    cfgs_wandb = cfgs_logging.get("wandb")

    if light_eval_only_mode:
        light_eval_freq = 1
    if plan_only_eval_mode or light_eval_only_mode or quick_debug or unroll_decode_eval_only_mode:
        filter_first_episodes, num_workers = 10, 0

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"🚀 Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    train_log_file = os.path.join(folder, f"log_r{rank}.csv")
    pref_tag = f"{tag}-" if tag else ""
    latest_file = pref_tag + f"latest.{latest_format}"
    latest_path = os.path.join(checkpoint_folder, latest_file)
    finetune = pretrained_path is not None
    logger.info(f"{'🔧 Finetuning mode' if finetune else '🆕 Training from scratch'}")
    # finetune covers the case of training heads on top of frozen transition_model
    # if cfgs_model["pretrained_path"] is None, i.e. training head just on encoder
    head_training_mode = main_optimizer in ["image_head", "state_head"]
    resume = os.path.exists(latest_path)
    resume_finetune = os.path.exists(latest_path) and finetune
    resume_latest = os.path.exists(latest_path) and not finetune
    if resume_finetune:
        logger.info("♻️  Resuming from checkpoint")
    load_path = None
    if load_model:
        if resume_finetune:
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
            load_opt_scale_epoch = not head_training_mode
        elif resume_latest:
            load_path = latest_path
            load_opt_scale_epoch = not head_training_mode
        else:  # not resuming, i.e. not os.path.exists(latest_path)
            load_path = pretrained_path
        if load_path is None or not os.path.exists(load_path):
            load_path = None
            load_model = False

    train_csv_logger, eval_csv_logger = None, None

    def create_csv_logger(losses, total_stats, train=False):
        if train:
            csv_log_file = train_log_file
        else:
            if light_eval_only_mode:
                csv_log_file = os.path.join(folder, f"light_eval_only_eval_r{rank}.csv")
            else:
                csv_log_file = os.path.join(folder, f"eval_r{rank}.csv")
        # Get all unique keys from eval_losses and eval_total_stats
        excluded_keys = [
            "eval_data/image_rollouts",
            "eval_data/image_rollouts_noisy_actions",
            "eval_data/image_animated_rollout",
        ]
        all_keys = {key for key in list(losses.keys()) + list(total_stats.keys()) if key not in excluded_keys}
        # Sort keys lexicographically for consistent order
        sorted_keys = sorted(all_keys)
        # Create a format tuple for each key
        new_columns = [("%.5f", key) for key in sorted_keys]
        # Initialize the logger with new columns
        if train:
            global train_csv_logger_columns
            train_csv_logger_columns = ["epoch", "itr", "loss", "gpu-time(ms)", "iter-time(ms)"] + sorted_keys
        else:
            global eval_csv_logger_columns
            eval_csv_logger_columns = ["epoch", "itr"] + sorted_keys
        return CSVLogger(csv_log_file, ("%d", "epoch"), ("%d", "itr"), *new_columns)

    # -- init data-loaders/samplers
    transform = make_transforms(
        img_size=img_size,
        **cfgs_data_aug,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, **cfgs_data_aug)

    # Prepare data kwargs from config, flattening nested structures and filtering out non-init_data fields
    excluded_keys = [
        "datasets",
        "paths",
        "val_datasets",
        "img_size",
        # subfields of cfgs_data
        "validation",
        "loader",
        "custom",
        "droid",
    ]
    data_kwargs = {k: v for k, v in cfgs_data.items() if k not in excluded_keys}
    # Add fields from nested structures
    data_kwargs.update(cfgs_validation)
    data_kwargs.update(cfgs_loader)
    data_kwargs.update(cfgs_custom)
    data_kwargs.update(cfgs_droid)
    # Add computed/override parameters
    data_kwargs.update(
        {
            "data_paths": dataset_paths,
            "val_data_paths": val_dataset_paths,
            "transform": transform,
            "world_size": world_size,
            "rank": rank,
            # overridden by quick_debug
            "filter_first_episodes": filter_first_episodes,  # Potentially overridden by mode
            "num_workers": num_workers,
        }
    )

    val_data_iters = []
    (
        dataset,
        val_dataset,
        traj_dataset,
        val_traj_dataset,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
        viz_val_data_loader,
    ) = init_data(**data_kwargs)
    val_data_iters.append((val_dataset, val_traj_dataset, val_unsupervised_loader))

    if val_datasets_1:
        # Reuse data_kwargs and override val_datasets_1-specific fields
        data_kwargs_1 = data_kwargs.copy()
        data_kwargs_1.update(
            {
                "dset_fraction": 1,
                "val_dset_fraction": 1,
                "data_paths": val_datasets_1_paths,
                "val_data_paths": val_datasets_1_paths,
                "batch_size": val_datasets_1.get("batch_size", 4),
                "drop_last": val_datasets_1.get("drop_last", True),
                "fps": val_datasets_1.get("fps", 4),
                "dataset_fpcs": val_datasets_1.get("fpcs", [8]),
                "val_dataset_fpcs": val_datasets_1.get("fpcs", [8]),
                "camera_views": val_datasets_1.get("camera_views", ["exterior_image_2_left"]),
                "droid_to_rcasa_action_format": val_datasets_1.get("droid_to_rcasa_action_format", 1),
            }
        )
        _, val_dataset_1, _, val_traj_dataset_1, _, val_unsupervised_loader_1, _, viz_val_data_loader = init_data(
            **data_kwargs_1
        )
        val_data_iters.append((val_dataset_1, val_traj_dataset_1, val_unsupervised_loader_1))
    if dataset_type in {"custom", "stablewm_h5"} and traj_dataset is not None:
        preprocessor = Preprocessor(
            action_mean=traj_dataset.action_mean,
            action_std=traj_dataset.action_std,
            state_mean=traj_dataset.state_mean,
            state_std=traj_dataset.state_std,
            proprio_mean=traj_dataset.proprio_mean,
            proprio_std=traj_dataset.proprio_std,
            transform=transform,
            inverse_transform=inverse_transform,
        )
    else:
        preprocessor = None

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    if planner_identified_enabled and ipe != _dlen:
        raise ValueError(
            "planner-identified canonical passes require iterations_per_epoch=null "
            f"(loader={_dlen}, configured={ipe})"
        )
    optimizer_steps_per_epoch = int(ipe) // gradient_accumulation_steps
    if optimizer_steps_per_epoch < 1:
        raise ValueError("not enough microbatches for one optimizer step")
    expected_optimizer_steps_per_epoch = (
        cfgs_opt.get("transition_model", {}).get(
            "expected_optimizer_steps_per_epoch"
        )
        if main_optimizer == "transition_model"
        else None
    )
    if (
        expected_optimizer_steps_per_epoch is not None
        and optimizer_steps_per_epoch
        != int(expected_optimizer_steps_per_epoch)
    ):
        raise ValueError(
            "optimizer steps per physical pass differ from the pinned task "
            f"schedule: actual={optimizer_steps_per_epoch}, "
            f"expected={expected_optimizer_steps_per_epoch}"
        )
    microbatches_per_epoch = optimizer_steps_per_epoch * gradient_accumulation_steps
    dropped_microbatches = int(ipe) - microbatches_per_epoch
    ipe = microbatches_per_epoch
    natural_optimizer_steps = int(num_epochs) * optimizer_steps_per_epoch
    exact_optimizer_step_budget = configured_optimizer_step_budget is not None
    optimizer_step_budget = (
        int(configured_optimizer_step_budget)
        if exact_optimizer_step_budget
        else natural_optimizer_steps
    )
    if optimizer_step_budget < 1:
        raise ValueError("total_optimizer_steps must be positive")
    if exact_optimizer_step_budget and not (
        (int(num_epochs) - 1) * optimizer_steps_per_epoch
        < optimizer_step_budget
        <= natural_optimizer_steps
    ):
        raise ValueError(
            "num_epochs must be the minimal driver horizon containing the exact "
            "optimizer-step budget: "
            f"epochs={num_epochs}, steps_per_epoch={optimizer_steps_per_epoch}, "
            f"budget={optimizer_step_budget}"
        )
    maximum_complete_passes = optimizer_step_budget // optimizer_steps_per_epoch
    if (
        stop_after_complete_passes > 0
        and stop_after_complete_passes > maximum_complete_passes
    ):
        raise ValueError(
            "requested pass-boundary stop exceeds the configured optimizer "
            "horizon: "
            f"requested={stop_after_complete_passes}, "
            f"maximum_complete_passes={maximum_complete_passes}"
        )
    training_microbatch_budget = optimizer_step_budget * gradient_accumulation_steps
    logger.info(
        "Training schedule: "
        f"loader_microbatches={_dlen}, used_microbatches={ipe}, "
        f"gradient_accumulation={gradient_accumulation_steps}, "
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch}, "
        f"dropped_tail_microbatches={dropped_microbatches}, "
        f"optimizer_step_budget={optimizer_step_budget}"
    )
    logger.info(f"📊 Iterations per epoch: {ipe} (dataset size: {_dlen})")
    if canary_optimizer_steps:
        if not planner_identified_enabled:
            raise ValueError("canary_optimizer_steps requires planner-identified training")
        if canary_optimizer_steps >= optimizer_steps_per_epoch:
            raise ValueError(
                "canary_optimizer_steps must be shorter than one physical pass: "
                f"requested={canary_optimizer_steps}, pass={optimizer_steps_per_epoch}"
            )
        logger.info(
            "PI canary enabled: "
            f"optimizer_steps={canary_optimizer_steps}, "
            f"microbatches={canary_optimizer_steps * gradient_accumulation_steps}"
        )
    if main_optimizer == "transition_model":
        cfgs_opt["transition_model"]["iterations_per_epoch"] = optimizer_steps_per_epoch
    elif main_optimizer == "image_head":
        cfgs_opt["heads"]["image_head"]["iterations_per_epoch"] = ipe
    elif main_optimizer == "state_head":
        cfgs_opt["heads"]["state_head"]["iterations_per_epoch"] = ipe

    planner_loader = None
    planner_sampler = None
    planner_validation_groups = None
    planner_validation_loader = None
    planner_validation_sampler = None
    if planner_identified_enabled:
        if dataset_type != "stablewm_h5":
            raise ValueError("planner-identified cross-model training requires dataset_type=stablewm_h5")
        if len(dataset_paths) != 1:
            raise ValueError("planner-identified training requires one source HDF5")
        sidecar_h5 = cfgs_planner_identified.get("sidecar_h5")
        if not sidecar_h5:
            raise ValueError("planner_identified.sidecar_h5 is required")
        planner_history_size = int(
            cfgs_planner_identified.get(
                "history_size",
                cfgs_custom.get("num_hist", 1),
            )
        )
        if planner_history_size != int(cfgs_custom.get("num_hist", 1)):
            raise ValueError(
                "planner history_size must equal data.custom.num_hist"
            )
        planner_dataset = PlannerLandscapeH5Dataset(
            source_h5=dataset_paths[0],
            sidecar_h5=sidecar_h5,
            transform=transform,
            action_mean=traj_dataset.action_mean,
            action_std=traj_dataset.action_std,
            proprio_mean=traj_dataset.proprio_mean,
            proprio_std=traj_dataset.proprio_std,
            source_metadata=traj_dataset,
            expected_protocol=cfgs_planner_identified.get("expected_protocol", "pointmaze_planner_counterfactual_v1"),
            expected_groups=cfgs_planner_identified.get("expected_groups"),
            expected_branches_per_group=cfgs_planner_identified.get(
                "expected_branches_per_group"
            ),
            expected_config_sha256=cfgs_planner_identified.get(
                "expected_config_sha256"
            ),
            frameskip=frameskip,
            action_skip=action_skip,
            history_size=planner_history_size,
            goal_offset_steps=cfgs_planner_identified.get("goal_offset_steps", 25),
        )
        planner_training_groups, planner_validation_groups = split_planner_groups_by_physical_episode(
            planner_dataset,
            traj_dataset.train_episode_ids,
        )
        planner_sampler = DistributedSampler(
            planner_training_groups,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        planner_num_workers = int(cfgs_planner_identified.get("num_workers", 4))
        planner_loader_kwargs = {
            "dataset": planner_training_groups,
            "batch_size": int(cfgs_planner_identified.get("groups_per_batch", 16)),
            "sampler": planner_sampler,
            "drop_last": True,
            "pin_memory": bool(cfgs_planner_identified.get("pin_memory", True)),
            "num_workers": planner_num_workers,
            "persistent_workers": planner_num_workers > 0
            and bool(cfgs_planner_identified.get("persistent_workers", False)),
        }
        if planner_num_workers > 0:
            planner_loader_kwargs["prefetch_factor"] = int(cfgs_planner_identified.get("prefetch_factor", 2))
        planner_loader = DataLoader(**planner_loader_kwargs)
        if len(planner_loader) < 1:
            raise ValueError("planner landscape loader has no complete group batch")
        planner_validation_sampler = DistributedSampler(
            planner_validation_groups,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if len(planner_validation_groups) % world_size:
            raise ValueError(
                "planner held-out groups must divide world_size exactly so "
                "checkpoint selection never duplicates validation groups"
            )
        planner_validation_loader_kwargs = dict(planner_loader_kwargs)
        planner_validation_loader_kwargs.update(
            {
                "dataset": planner_validation_groups,
                "batch_size": int(
                    cfgs_planner_identified.get(
                        "validation_groups_per_batch",
                        cfgs_planner_identified.get("groups_per_batch", 16),
                    )
                ),
                "sampler": planner_validation_sampler,
                "drop_last": False,
            }
        )
        planner_validation_loader = DataLoader(**planner_validation_loader_kwargs)
        if len(planner_validation_loader) < 1:
            raise ValueError("planner held-out loader has no group batch")
        logger.info(
            "Planner landscape split: "
            f"train_groups={len(planner_training_groups)}, "
            f"validation_groups={len(planner_validation_groups)}, "
            f"train_batches={len(planner_loader)}, "
            f"validation_batches={len(planner_validation_loader)}"
        )

    # Logger
    class Trainer:
        _visible_wandb_metrics = (
            "epoch",
            "loss",
            "planner_landscape_loss",
            "planner_log_scale",
            "planner_scale",
            "planner_valid_group_fraction",
            "planner_valid_pairwise_sign_accuracy",
            "planner_predicted_to_real_cost_std_ratio",
            "info/planner_scale/lr",
            "optim/planner_scale/grad_norm_before_clip",
            "validation/planner_landscape_loss",
            "validation/planner_valid_pairwise_sign_accuracy",
            "validation/planner_predicted_to_real_cost_std_ratio",
        )

        def __init__(self, config):
            if quick_debug:
                config["debug"] = quick_debug
            self.config = config
            self.ipe = ipe
            self.use_wandb = config.get("use_wandb", False)
            self.disable_wandb_media = config.get("disable_wandb_media", False)
            self.log_media_locally = config.get("log_media_locally", False)
            self.local_log_dir = None
            if self.log_media_locally and rank == 0:
                self.local_log_dir = os.path.join(folder, "local_logs")
                os.makedirs(self.local_log_dir, exist_ok=True)
            if self.use_wandb and rank == 0:
                project_name = config.get("project", "vjepa_wm") if not config["debug"] else "vjepa_wm_debug"
                wandb_run_id_file = os.path.join(folder, "wandb_run_id.txt")
                if os.path.exists(wandb_run_id_file):
                    with open(wandb_run_id_file, "r") as f:
                        wandb_run_id = f.read().strip()
                    wandb.init(project=project_name, id=wandb_run_id, resume="allow", dir=folder)
                    logger.info(f"Resuming Wandb run {wandb_run_id}")
                else:
                    wandb.init(project=project_name, dir=folder)
                    with open(wandb_run_id_file, "w") as f:
                        f.write(wandb.run.id)
                wandb.run.name = os.path.basename(folder)
                # Retain every scalar for audit/export while preventing W&B's
                # automatic workspace from creating dozens of low-value panels.
                wandb.define_metric("*", hidden=True)
                wandb.define_metric("total_optimizer_step", hidden=True)
                wandb.define_metric(
                    "validation/total_optimizer_steps",
                    hidden=True,
                )
                planner_step_metrics = {
                    "planner_landscape_loss",
                    "planner_log_scale",
                    "planner_scale",
                    "planner_valid_group_fraction",
                    "planner_valid_pairwise_sign_accuracy",
                    "planner_predicted_to_real_cost_std_ratio",
                    "info/planner_scale/lr",
                    "optim/planner_scale/grad_norm_before_clip",
                }
                for metric_name in self._visible_wandb_metrics:
                    step_metric = (
                        "validation/total_optimizer_steps"
                        if metric_name.startswith("validation/")
                        else (
                            "total_optimizer_step"
                            if metric_name in planner_step_metrics
                            else None
                        )
                    )
                    if step_metric is None:
                        wandb.define_metric(metric_name, hidden=False)
                    else:
                        wandb.define_metric(
                            metric_name,
                            step_metric=step_metric,
                            hidden=False,
                        )
                self.job_set = set()

        def log(self, epoch, itr, losses, total_stats, eval_losses=None, eval_total_stats=None, image_stats=None):
            microbatch_in_epoch = itr + 1
            optimizer_step_in_epoch = microbatch_in_epoch // gradient_accumulation_steps
            log_dict = {
                "epoch": epoch + 1,
                "itr": itr,
                "microbatch_in_epoch": microbatch_in_epoch,
                "microbatches_per_epoch": ipe,
                "optimizer_step_in_epoch": optimizer_step_in_epoch,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "optimizer_step_budget": optimizer_step_budget,
                "total_microbatch_step": epoch * ipe + microbatch_in_epoch,
                "total_optimizer_step": (
                    epoch * optimizer_steps_per_epoch + optimizer_step_in_epoch
                ),
                "epoch_progress": microbatch_in_epoch / ipe,
            }
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                log_dict[key] = value
            for key, value in total_stats.items():
                log_dict[key] = value
            if eval_losses is not None:
                for key, value in eval_losses.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().item()
                    log_dict[key] = value
            if eval_total_stats is not None:
                for key, value in eval_total_stats.items():
                    log_dict[key] = value
            if image_stats:  # not None or nonempty
                if self.log_media_locally and rank == 0:
                    self.log_media_local(image_stats, epoch, itr)
                if not self.disable_wandb_media:
                    log_dict.update(image_stats)
            if "loss" in log_dict.keys() and itr % log_freq == 0:
                logger.info(
                    "[epoch %d/%d | global opt %d/%d | pass opt %d/%d | "
                    "micro %d/%d | %.1f%%] "
                    "loss: %.3f | "
                    % (
                        epoch + 1,
                        num_epochs,
                        epoch * optimizer_steps_per_epoch
                        + optimizer_step_in_epoch,
                        optimizer_step_budget,
                        optimizer_step_in_epoch,
                        optimizer_steps_per_epoch,
                        microbatch_in_epoch,
                        ipe,
                        100.0 * microbatch_in_epoch / ipe,
                        log_dict["loss"],
                    )
                )
            if self.use_wandb and rank == 0:
                wandb.log(log_dict)

        def log_media_local(self, image_stats, epoch=None, itr=None):
            step = epoch * ipe + itr
            for key, value in image_stats.items():
                subfolder = os.path.join(self.local_log_dir, "/".join(key.split("/")))
                os.makedirs(subfolder, exist_ok=True)
                filename = f"{step}.gif" if isinstance(value, wandb.Video) else f"{step}.pdf"
                if isinstance(value, wandb.Video):
                    frames = value._prepare_video(value.data)
                    duration = 1.0 / 10  # assuming 10 FPS
                    imageio.mimsave(os.path.join(subfolder, filename), frames, duration=duration, loop=0)
                elif isinstance(value, wandb.Image):
                    value.image.save(os.path.join(subfolder, filename))
                elif isinstance(value, matplotlib.figure.Figure):
                    value.savefig(os.path.join(subfolder, filename), bbox_inches=None)

    trainer = Trainer(cfgs_wandb)

    # -- init model
    if use_action:
        actions_per_vid_feat = tubelet_size_enc * frameskip // action_skip
        model_action_dim = traj_dataset.action_dim * tubelet_size_enc * frameskip // action_skip
    else:
        actions_per_vid_feat, model_action_dim = None, None
    if use_proprio:
        proprio_multiplier = tubelet_size_enc * frameskip // state_skip
        model_proprio_dim = traj_dataset.proprio_dim * tubelet_size_enc // state_skip
    else:
        proprio_multiplier, model_proprio_dim = None, None

    # Prepare model kwargs by flattening nested configs and filtering out non-init_video_model fields
    excluded_keys = [
        "rollout_cfg",
        "heads_cfg",
        "pretrained_path",
        "visual_encoder",
        "action_encoder",
        "proprio_encoder",
        "predictor",
        "wm_encoding",
        "attn",
    ]
    model_kwargs = {k: v for k, v in cfgs_model.items() if k not in excluded_keys}
    if "visual_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["visual_encoder"])
    if "action_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["action_encoder"])
    if "proprio_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["proprio_encoder"])
    if "predictor" in cfgs_model:
        model_kwargs.update(cfgs_model["predictor"])
    model_kwargs.update(
        {
            "device": device,
            "img_size": img_size,
            "action_dim": model_action_dim,  # Computed from dataset
            "proprio_dim": model_proprio_dim,  # Computed from dataset
            "cfgs_attn_pattern": cfgs_model.get("attn", None),  # Pass attn subconfig
            "use_proprio": use_proprio,  # Computed derived value
            "use_action": use_action,  # Computed derived value
        }
    )
    # Dataset/planner construction differs between PI-LTC and vanilla.  Reset the
    # model-initialization RNG boundary so those setup paths cannot perturb the
    # common predictor/encoder initialization used by the matched comparison.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    predictor, encoder, action_encoder, proprio_encoder = init_video_model(**model_kwargs)
    common_trainable_initialization_sha256 = _common_trainable_initialization_sha256(
        predictor,
        action_encoder,
        proprio_encoder,
    )
    logger.info(
        "Common trainable initialization SHA-256: "
        f"{common_trainable_initialization_sha256}"
    )

    heads = {}
    if train_heads or pretrain_dec_path is not None:
        if "image_head" in heads_architectures:
            image_head_type = heads_architectures["image_head"]["kind"]
            if image_head_type is not None and image_head_type.lower() != "none":
                if image_head_type == "vit":
                    decoder = WorldModelViTImageHead(
                        head_config=dict(heads_architectures["image_head"]["config"]),
                        inverse_transform=inverse_transform,
                        device=device,
                    )
                heads["image_head"] = decoder
        if "state_head" in heads_architectures:
            state_decoder = WorldModelPoseReadoutHead(
                head_config=dict(heads_architectures["state_head"]["config"]), device=device
            )
            heads["state_head"] = state_decoder
        if "reward_head" in heads_architectures:
            reward_decoder = WorldModelRewardReadoutHead(
                head_config=dict(heads_architectures["reward_head"]["config"]), device=device
            )
            heads["reward_head"] = reward_decoder

    # -- init optimizer and scheduler
    if train_predictor and predictor is not None:
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            encoder=encoder,
            freeze_encoder=freeze_encoder,
            planner_scale_lr_multiplier=(
                float(
                    cfgs_planner_identified.get(
                        "scale_lr_multiplier",
                        1.0,
                    )
                )
                if planner_identified_enabled
                else None
            ),
            **cfgs_opt["transition_model"],
        )
        clip_grad = cfgs_opt["transition_model"]["clip_grad"]
        use_radamw = cfgs_opt["transition_model"]["use_radamw"]
        if sampling_scheduler_type == "linear":
            # linear decay from sampling_scheduler_start to sampling_scheduler_end
            rollout_sampling_scheduler = (
                sampling_scheduler_start
                - i
                * (sampling_scheduler_start - sampling_scheduler_end)
                / training_microbatch_budget
                for i in range(training_microbatch_budget + 1)
            )
        elif sampling_scheduler_type == "exponential":
            # exponential decay from sampling_scheduler_start to sampling_scheduler_end
            rollout_sampling_scheduler = (
                sampling_scheduler_start
                * (sampling_scheduler_end / sampling_scheduler_start)
                ** (i / training_microbatch_budget)
                for i in range(training_microbatch_budget + 1)
            )
        elif sampling_scheduler_type == "sigmoid":
            rollout_sampling_scheduler = (
                sampling_scheduler_start
                + (sampling_scheduler_end - sampling_scheduler_start)
                * (
                    1
                    / (
                        1
                        + np.exp(
                            -10 * (i / training_microbatch_budget - 0.5)
                        )
                    )
                )
                for i in range(training_microbatch_budget + 1)
            )
    else:
        optimizer, scaler, scheduler, wd_scheduler, clip_grad, use_radamw = None, None, None, None, None, None
    if train_heads:
        for name, head in heads.items():
            head.init_opt(**dict(cfgs_opt["heads"][name]))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        # to resume predictor or head training
        load_heads = heads and not pretrain_dec_path and resume
        logger.info(f"Load heads: {load_heads}")
        (
            predictor,
            action_encoder,
            proprio_encoder,
            heads,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            heads=heads,
            opt=optimizer,
            scaler=scaler,
            load_opt_scale_epoch=load_opt_scale_epoch,
            load_heads=load_heads,
            train_heads=train_heads,
            train_predictor=train_predictor,
            strict_resume=bool(strict_provenance and resume_latest),
            expected_optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            expected_optimizer_step_budget=optimizer_step_budget,
            expected_gradient_accumulation_steps=gradient_accumulation_steps,
            expected_training_provenance=training_provenance,
            expected_common_trainable_initialization_sha256=(
                common_trainable_initialization_sha256
            ),
        )
        # Only resume the schedulers if we resume a pretraining or a finetuning
        # Not if we start a finetuning: we reset them
        if load_opt_scale_epoch and scheduler is not None and wd_scheduler is not None:
            for _ in range(start_epoch * optimizer_steps_per_epoch):
                scheduler.step()
                wd_scheduler.step()
        if light_eval_only_mode:
            start_epoch -= 1
    if canary_optimizer_steps and start_epoch != 0:
        raise RuntimeError("PI canary must start from the canonical epoch-0 initialization")

    # Load pretrained heads from pretrain_dec_path
    if pretrain_dec_path is not None:
        for name, head in heads.items():
            if new_path_heads.get(name, True):
                head_path = pretrain_dec_path[name].removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                head.load_checkpoint(head_path)
                logger.info(f"loaded pretrained head named {name}")
            else:
                checkpoint = torch.load(pretrain_dec_path[name], map_location=torch.device("cpu"))
                epoch = checkpoint["epoch"]
                pretrained_dict = clean_state_dict(checkpoint[name])
                msg = head.model.load_state_dict(pretrained_dict, strict=False)
                logger.info(f"loaded pretrained head named {name} from epoch {epoch} with msg: {msg}")
                del checkpoint

    # DDP wrapping after loading state_dicts
    if not freeze_encoder:
        encoder = DDP(encoder, static_graph=False, find_unused_parameters=False)
    if train_predictor:
        if action_encoder is not None:
            action_encoder = DDP(action_encoder, static_graph=False, find_unused_parameters=False)
        if proprio_encoder is not None:
            proprio_encoder = DDP(proprio_encoder, static_graph=False, find_unused_parameters=False)
        predictor = DDP(
            predictor,
            static_graph=False,
            find_unused_parameters=planner_identified_enabled,
        )
    for name in heads.keys():
        heads[name].model = DDP(heads[name].model, static_graph=False, find_unused_parameters=False)

    # Prepare VideoWM kwargs from config
    wm_kwargs = {
        "device": device,
        # Model components
        "encoder": encoder,
        "predictor": predictor,
        "action_encoder": action_encoder,
        "proprio_encoder": proprio_encoder,
        # Computed dimensions
        "action_dim": model_action_dim,
        "proprio_dim": model_proprio_dim,
        "use_proprio": use_proprio,
        "use_action": use_action,
        # From cfgs_model (pass directly from config)
        "action_tokens": action_tokens,
        "proprio_tokens": proprio_tokens,
        "grid_size": cfgs_model.get("grid_size", 14),
        "tubelet_size_enc": cfgs_model.get("tubelet_size_enc", 2),
        "action_conditioning": cfgs_model.get("action_conditioning", "token"),
        "proprio_encoding": cfgs_model.get("proprio_encoding", "feature"),
        "enc_type": cfgs_model["visual_encoder"].get("enc_type", "vjepa"),
        "pred_type": cfgs_model["predictor"].get("pred_type", "dino_wm"),
        "action_encoder_inpred": cfgs_model["action_encoder"].get("action_encoder_inpred", False),
        "proprio_encoder_inpred": cfgs_model["proprio_encoder"].get("proprio_encoder_inpred", False),
        **cfgs_wm_encoding,
        # From cfgs_data
        "action_skip": action_skip,
        "frameskip": frameskip,
        "img_size": cfgs_data.get("img_size", 256),
        # Heads
        "heads": heads,
        # Optimization
        "scaler": scaler,
        "optimizer": optimizer,
        "clip_grad": clip_grad,
        "mixed_precision": mixed_precision,
        "use_radamw": use_radamw,
        # Loss config (pass subconfig directly)
        "cfgs_loss": cfgs_loss,
    }
    world_model = VideoWM(**wm_kwargs)
    if planner_identified_enabled:
        scale_module = planner_input_scale(world_model)
        logger.info(
            "Planner-identified input scale enabled: "
            f"scale={scale_module.input_scale.detach().item():.6f}; "
            "ordinary transition loss owns encoder/action/proprio/predictor, "
            "planner landscape owns log_input_scale only"
        )
        zero = torch.zeros((), device=device)
        planner_last_stats = {
            "planner_landscape_loss": zero,
            "planner_pairwise_sign_accuracy": zero,
            "planner_valid_pairwise_sign_accuracy": zero,
            "planner_predicted_cost_std": zero,
            "planner_real_cost_std": zero,
            "planner_predicted_to_real_cost_std_ratio": zero,
            "planner_zero_difference_baseline": zero.new_tensor(1.0),
            "planner_improvement_over_zero_difference": zero,
            "planner_context_frames": zero.new_tensor(
                float(cfgs_custom.get("num_hist", 1))
            ),
            "planner_valid_group_count": zero,
            "planner_valid_group_fraction": zero,
            "planner_target_energy_min": zero,
            "planner_target_energy_median": zero,
            "planner_target_energy_max": zero,
            "planner_target_energy_eps": zero,
            "planner_target_energy_effective_floor": zero,
            "planner_target_energy_relative_floor": zero,
            "planner_target_energy_min_to_median_ratio": zero,
            "planner_scale": scale_module.input_scale.detach(),
            "planner_log_scale": scale_module.log_scale.detach(),
            "planner_scale_gradient": zero,
        }
    else:
        planner_last_stats = {}

    # -- Initialize LPIPS once for evaluation
    lpips = lpips_lib.LPIPS(net="vgg").eval().to(device)

    def save_checkpoint(
        epoch,
        path,
        *,
        total_optimizer_step_count=None,
        optimizer_step_in_epoch=0,
        training_complete=False,
        planner_validation_metrics=None,
        planner_selection=None,
    ):
        if rank != 0:
            return
        if total_optimizer_step_count is None:
            total_optimizer_step_count = int(epoch) * optimizer_steps_per_epoch
        save_dict = {
            "predictor": world_model.predictor.state_dict() if world_model.predictor is not None else None,
            "opt": optimizer.state_dict() if optimizer is not None else None,
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "planner_identified": (dict(cfgs_planner_identified) if planner_identified_enabled else None),
            "total_optimizer_steps": int(total_optimizer_step_count),
            "optimizer_step_in_epoch": int(optimizer_step_in_epoch),
            "optimizer_step_budget": optimizer_step_budget,
            "training_complete": bool(training_complete),
            "requested_stop_after_complete_passes": (
                stop_after_complete_passes
            ),
            "planner_validation_metrics": planner_validation_metrics,
            "planner_selection": planner_selection,
            "training_provenance": training_provenance,
            "common_trainable_initialization_sha256": (
                common_trainable_initialization_sha256
            ),
        }
        if world_model.action_encoder is not None and not cfgs_model["action_encoder"].get(
            "action_encoder_inpred", False
        ):
            save_dict.update({"action_encoder": world_model.action_encoder.state_dict()})
        if (
            world_model.proprio_encoder is not None
            and use_proprio
            and not cfgs_model["proprio_encoder"].get("proprio_encoder_inpred", False)
        ):
            save_dict.update({"proprio_encoder": world_model.proprio_encoder.state_dict()})
        if train_heads:
            for name, head in world_model.heads.items():
                head_path = path.removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                head.save_checkpoint(epoch, head_path)
        temporary_path = f"{path}.tmp"
        try:
            torch.save(save_dict, temporary_path)
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise

    logger.info("Initializing loader...")
    train_loader = iter(unsupervised_loader)
    planner_loader_iter = iter(planner_loader) if planner_loader is not None else None
    if val_data_iters != [(None, None, None)]:
        val_loader_iters = []
        for vl_dset, vl_traj_dset, vl_loader in val_data_iters:
            val_loader_iters.append(iter(vl_loader))

    if viz_val_data_loader is not None:
        viz_val_loader_iter = iter(viz_val_data_loader)

    def load_clips(sample):
        all_clips = []
        return sample[0][0].to(device, non_blocking=True), None, None

    def get_batch(train=True, idx=0):
        nonlocal train_loader, val_loader_iters
        if dataset_type in {"custom", "stablewm_h5"}:
            try:
                if train:
                    obs, action, state, reward = next(train_loader)
                else:
                    obs, action, state, reward = next(val_loader_iters[idx])
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info(f"Exhausted data loaders idx {idx} with {train=}. Refreshing...")
                if train:
                    train_loader = iter(unsupervised_loader)
                    obs, action, state, reward = next(train_loader)
                else:
                    val_loader_iters[idx] = iter(val_data_iters[idx][2])
                    obs, action, state, reward = next(val_loader_iters[idx])
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=dtype, non_blocking=True)
            action = action.to(device, dtype=dtype, non_blocking=True)
            state = state.to(device, dtype=dtype, non_blocking=True)
            reward = reward.to(device, dtype=dtype, non_blocking=True)
            return obs, action, state, reward, None, None
        else:
            try:
                if train:
                    sample = next(train_loader)
                else:
                    sample = next(val_loader_iters[idx])
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted data loaders. Refreshing...")
                if train:
                    train_loader = iter(unsupervised_loader)
                    sample = next(train_loader)
                else:
                    val_loader_iters[idx] = iter(val_data_iters[idx][2])
                    sample = next(val_loader_iters[idx])
            all_clips, all_masks_enc, all_masks_pred = load_clips(sample)
            all_clips = {"visual": all_clips}
            return all_clips, None, None, None, all_masks_enc, all_masks_pred

    def get_planner_batch():
        nonlocal planner_loader_iter
        if planner_loader is None:
            raise RuntimeError("planner loader requested while PI-LTC is disabled")
        try:
            return next(planner_loader_iter)
        except StopIteration:
            planner_loader_iter = iter(planner_loader)
            return next(planner_loader_iter)

    planner_validation_history_path = Path(folder) / "planner_validation_history.json"
    planner_best_checkpoint_path = os.path.join(
        checkpoint_folder,
        pref_tag + f"planner-best.{latest_format}",
    )
    planner_validation_document = {
        "schema_version": 1,
        "selection_criterion": (
            "minimum held-out normalized pairwise landscape loss at a complete "
            "physical data-pass boundary, restricted to loss<1 (better than the "
            "zero-difference response) and valid sign accuracy>0.5; sign "
            "accuracy breaks exact ties"
        ),
        "records": [],
        "best": None,
    }
    if rank == 0 and planner_identified_enabled and planner_validation_history_path.exists():
        loaded_validation_document = json.loads(
            planner_validation_history_path.read_text(encoding="utf-8")
        )
        if loaded_validation_document.get("schema_version") != 1:
            raise ValueError("unsupported planner validation history schema")
        planner_validation_document = loaded_validation_document

    def evaluate_planner_validation():
        if planner_validation_loader is None:
            raise RuntimeError("planner validation requested while PI-LTC is disabled")
        # loss*valid, sign*valid, predicted_std*groups, real_std*groups,
        # valid groups, all groups
        totals = torch.zeros(6, dtype=torch.float64, device=device)
        for validation_batch in planner_validation_loader:
            validation_result = planner_landscape_result(
                world_model,
                validation_batch,
                device=device,
                dtype=dtype,
                mixed_precision=mixed_precision,
                target_energy_eps=float(
                    cfgs_planner_identified.get("target_energy_eps", 1.0e-8)
                ),
                target_energy_relative_floor=float(
                    cfgs_planner_identified.get(
                        "target_energy_relative_floor",
                        0.0,
                    )
                ),
                compute_scale_gradient=False,
            )
            validation_stats = validation_result.stats
            group_count = int(validation_batch["next_visual"].shape[0])
            valid_count = float(validation_stats["planner_valid_group_count"])
            totals[0] += float(validation_result.loss) * valid_count
            totals[1] += (
                float(validation_stats["planner_valid_pairwise_sign_accuracy"])
                * valid_count
            )
            totals[2] += float(validation_stats["planner_predicted_cost_std"]) * group_count
            totals[3] += float(validation_stats["planner_real_cost_std"]) * group_count
            totals[4] += valid_count
            totals[5] += group_count
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        valid_count = float(totals[4].item())
        group_count = float(totals[5].item())
        if valid_count <= 0.0 or group_count != float(len(planner_validation_groups)):
            raise RuntimeError(
                "planner held-out evaluation has no identifiable groups or did not "
                "cover every held-out group exactly once"
            )
        predicted_std = float(totals[2].item() / group_count)
        real_std = float(totals[3].item() / group_count)
        validation_loss = float(totals[0].item() / valid_count)
        metrics = {
            "planner_landscape_loss": validation_loss,
            "planner_valid_pairwise_sign_accuracy": float(
                totals[1].item() / valid_count
            ),
            "planner_predicted_cost_std": predicted_std,
            "planner_real_cost_std": real_std,
            "planner_predicted_to_real_cost_std_ratio": predicted_std
            / max(real_std, float(cfgs_planner_identified.get("target_energy_eps", 1.0e-8)) ** 0.5),
            "planner_valid_group_count": int(valid_count),
            "planner_group_count": int(group_count),
            "planner_valid_group_fraction": valid_count / group_count,
            "planner_zero_difference_baseline": 1.0,
            "planner_improvement_over_zero_difference": 1.0 - validation_loss,
            "planner_scale": float(
                planner_input_scale(world_model).input_scale.detach().cpu()
            ),
            "planner_log_scale": float(
                planner_input_scale(world_model).log_scale.detach().cpu()
            ),
            "planner_context_frames": int(cfgs_custom.get("num_hist", 1)),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError("planner held-out evaluation produced a non-finite metric")
        return metrics

    def persist_planner_validation(
        metrics,
        *,
        boundary,
        total_optimizer_steps,
        checkpoint_epoch,
        optimizer_step_in_epoch,
        eligible_for_selection,
        training_complete,
    ):
        if rank != 0:
            return False
        beats_zero_difference = metrics["planner_landscape_loss"] < 1.0
        beats_chance_ordering = (
            metrics["planner_valid_pairwise_sign_accuracy"] > 0.5
        )
        scientifically_eligible = (
            bool(eligible_for_selection)
            and beats_zero_difference
            and beats_chance_ordering
        )
        record = {
            "boundary": str(boundary),
            "total_optimizer_steps": int(total_optimizer_steps),
            "completed_physical_passes": int(total_optimizer_steps)
            // optimizer_steps_per_epoch,
            "optimizer_step_in_pass": int(optimizer_step_in_epoch),
            "complete_pass_boundary": bool(eligible_for_selection),
            "beats_zero_difference_baseline": beats_zero_difference,
            "beats_chance_pairwise_ordering": beats_chance_ordering,
            "eligible_for_selection": scientifically_eligible,
            "metrics": dict(metrics),
            "checkpoint": None,
        }
        best = planner_validation_document.get("best")
        is_best = False
        if scientifically_eligible:
            if best is None:
                is_best = True
            else:
                candidate_loss = metrics["planner_landscape_loss"]
                best_loss = best["metrics"]["planner_landscape_loss"]
                candidate_sign = metrics[
                    "planner_valid_pairwise_sign_accuracy"
                ]
                best_sign = best["metrics"][
                    "planner_valid_pairwise_sign_accuracy"
                ]
                is_best = candidate_loss < best_loss or (
                    candidate_loss == best_loss and candidate_sign > best_sign
                )
        if is_best:
            selection = {
                "criterion": planner_validation_document["selection_criterion"],
                "total_optimizer_steps": int(total_optimizer_steps),
                "completed_physical_passes": int(total_optimizer_steps)
                // optimizer_steps_per_epoch,
            }
            save_checkpoint(
                checkpoint_epoch,
                planner_best_checkpoint_path,
                total_optimizer_step_count=total_optimizer_steps,
                optimizer_step_in_epoch=optimizer_step_in_epoch,
                training_complete=training_complete,
                planner_validation_metrics=dict(metrics),
                planner_selection=selection,
            )
            record["checkpoint"] = planner_best_checkpoint_path
            planner_validation_document["best"] = record
        planner_validation_document["records"].append(record)
        temporary_path = planner_validation_history_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(planner_validation_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(planner_validation_history_path)
        if trainer.use_wandb:
            wandb.log(
                {
                    "validation/planner_landscape_loss": metrics[
                        "planner_landscape_loss"
                    ],
                    "validation/planner_valid_pairwise_sign_accuracy": metrics[
                        "planner_valid_pairwise_sign_accuracy"
                    ],
                    "validation/planner_predicted_to_real_cost_std_ratio": metrics[
                        "planner_predicted_to_real_cost_std_ratio"
                    ],
                    "validation/planner_scale": metrics["planner_scale"],
                    "validation/total_optimizer_steps": int(total_optimizer_steps),
                }
            )
        logger.info(
            "[PLANNER HELD-OUT] "
            f"boundary={boundary} steps={total_optimizer_steps} "
            f"loss={metrics['planner_landscape_loss']:.6f} "
            f"sign={metrics['planner_valid_pairwise_sign_accuracy']:.4f} "
            f"scale={metrics['planner_scale']:.6f} best={is_best}"
        )
        return is_best

    def get_viz_batch():
        """Get a visualization batch from the non-distributed validation loader (rank 0 only)"""
        nonlocal viz_val_loader_iter
        if rank != 0 or viz_val_loader_iter is None:
            return None, None, None, None, None, None
        if dataset_type in {"custom", "stablewm_h5"}:
            try:
                obs, action, state, reward = next(viz_val_loader_iter)
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted viz data loader. Refreshing...")
                viz_val_loader_iter = iter(viz_val_data_loader)
                obs, action, state, reward = next(viz_val_loader_iter)
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=torch.float32, non_blocking=True)
            action = action.to(device, dtype=torch.float32, non_blocking=True)
            state = state.to(device, dtype=torch.float32, non_blocking=True)
            reward = reward.to(device, dtype=torch.float32, non_blocking=True)
            return obs, action, state, reward, None, None
        else:
            try:
                sample = next(viz_val_loader_iter)
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted viz data loader. Refreshing...")
                viz_val_loader_iter = iter(viz_val_data_loader)
                sample = next(viz_val_loader_iter)
            all_clips, all_masks_enc, all_masks_pred = load_clips(sample)
            all_clips = {"visual": all_clips}
            return all_clips, None, None, None, all_masks_enc, all_masks_pred

    # -- TRAINING LOOP
    if not (plan_only_eval_mode or unroll_decode_eval_only_mode):
        canary_complete = False
        step_budget_complete = False
        planner_initial_validation_metrics = None
        if (
            planner_identified_enabled
            and not light_eval_only_mode
            and start_epoch == 0
            and bool(cfgs_planner_identified.get("validation_at_start", True))
        ):
            planner_initial_validation_metrics = evaluate_planner_validation()
            persist_planner_validation(
                planner_initial_validation_metrics,
                boundary="initialization",
                total_optimizer_steps=0,
                checkpoint_epoch=0,
                optimizer_step_in_epoch=0,
                eligible_for_selection=False,
                training_complete=False,
            )
        if canary_optimizer_steps and planner_initial_validation_metrics is None:
            raise ValueError(
                "PI canary requires planner_identified.validation_at_start=true"
            )
        canary_valid_fraction_meter = AverageMeter()
        canary_landscape_loss_meter = AverageMeter()
        canary_sign_accuracy_meter = AverageMeter()
        canary_scale_gradient_abs_meter = AverageMeter()
        canary_effective_energy_floor_meter = AverageMeter()
        canary_min_to_median_energy_ratio_meter = AverageMeter()
        boundary_stop_complete = False
        for epoch in range(start_epoch, num_epochs):
            logger.info("\n" + "─" * 50)
            logger.info(f"📈 Epoch {epoch + 1}/{num_epochs}")
            logger.info("─" * 50)

            # -- update distributed-data-loader epoch
            unsupervised_sampler.set_epoch(epoch)
            # Recreate the iterator so the deliberately dropped tail never
            # leaks into the next physical pass.
            train_loader = iter(unsupervised_loader)
            if planner_sampler is not None:
                planner_sampler.set_epoch(epoch)
                planner_loader_iter = iter(planner_loader)

            loss_meter = AverageMeter()
            gpu_time_meter = AverageMeter()
            wall_time_meter = AverageMeter()

            for itr in range(ipe):
                itr_start_time = time.time()
                if quick_debug or light_eval_only_mode:
                    if itr > 5:
                        break

                def step_model(obs, action, state, reward, train=True):
                    optimizer_boundary = train and (itr + 1) % gradient_accumulation_steps == 0
                    rates = defaultdict(float)
                    rates["info/transition_model/optimizer_step"] = (
                        epoch * optimizer_steps_per_epoch + (itr + 1) // gradient_accumulation_steps
                    )
                    rates["info/transition_model/microbatch_step"] = epoch * ipe + itr + 1
                    rates["info/transition_model/optimizer_step_in_epoch"] = (
                        (itr + 1) // gradient_accumulation_steps
                    )
                    rates["info/transition_model/microbatch_in_epoch"] = itr + 1
                    rates["info/transition_model/epoch_progress"] = (itr + 1) / ipe
                    if train:
                        if train_predictor and optimizer_boundary:
                            rates["info/transition_model/lr"] = scheduler.step()
                            rates["info/transition_model/wd"] = wd_scheduler.step()
                        elif train_predictor:
                            rates["info/transition_model/lr"] = optimizer.param_groups[0]["lr"]
                            rates["info/transition_model/wd"] = optimizer.param_groups[0]["weight_decay"]
                        if planner_identified_enabled:
                            planner_scale_groups = [
                                group
                                for group in optimizer.param_groups
                                if group.get("group_name") == "planner_scale"
                            ]
                            if len(planner_scale_groups) != 1:
                                raise RuntimeError(
                                    "planner scale optimizer group is missing or duplicated"
                                )
                            rates["info/planner_scale/lr"] = planner_scale_groups[0][
                                "lr"
                            ]
                        if train_heads:
                            for name, head in world_model.heads.items():
                                rates[f"info/{name}/lr"] = head.scheduler.step()
                                rates[f"info/{name}/wd"] = head.wd_scheduler.step()
                    else:
                        rates["info/transition_model/lr"] = 0.0
                        rates["info/transition_model/wd"] = 0.0
                        for name, head in world_model.heads.items():
                            rates[f"info/{name}/lr"] = 0.0
                            rates[f"info/{name}/wd"] = 0.0
                    # --

                    # Step 1. Forward
                    total_stats = {}
                    total_transition_loss = 0.0
                    total_head_loss = 0.0
                    if action is not None:
                        total_stats.update(
                            {
                                "act_mean": action.mean(),
                                "act_std": action.std(),
                                "act_min": action.min(),
                                "act_max": action.max(),
                            }
                        )
                    # 1. TRAIN PREDICTOR TO PREDICT ONE STEP IN THE FUTURE USING TEACHER FORCING
                    train_rollout_result = {}
                    parallel_rollout_result = {}
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        video_features, proprio_features, action_features = world_model.encode(obs, action)
                        if train_predictor and train and predictor is not None:
                            pred_video_features, pred_action_features, pred_proprio_features = (
                                world_model.forward_pred(
                                    video_features,
                                    action_features,
                                    proprio_features,
                                )
                            )
                            predictor_losses = world_model.compute_loss(
                                pred_video_features,
                                pred_proprio_features,
                                video_features,
                                proprio_features,
                                shift=1,
                            )
                        else:
                            pred_video_features, pred_proprio_features = None, None
                            predictor_losses = {}
                    predictor_loss = predictor_losses.get("loss", 0.0) / (rollout_steps + 1)
                    if train and train_predictor and predictor is not None:
                        total_transition_loss += predictor_loss
                    stats = defaultdict(list)
                    for k in predictor_losses:
                        val = predictor_losses[k]
                        if isinstance(val, torch.Tensor):
                            val = val.detach().clone()
                        else:
                            val = torch.tensor(val)
                        stats[k].append(val.unsqueeze(0))
                    # mean over 1 element in list
                    stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                    for k in stats:
                        for j in range(len(stats[k])):
                            train_rollout_result[f"train_rollout/{k}/{j+1}"] = stats[k][j].item()
                    # 2. TRAIN HEADS ON FEATURES FROM ENCODER
                    if "image_head" in world_model.heads and train_heads:
                        # Encoder produces features of videos normalized by mean and std
                        # So let's keep them for target of decoder loss
                        target_rgb_video = world_model.heads["image_head"].preprocess_rgb(
                            obs["visual"][:, ::tubelet_size_enc]
                        )
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_image_losses = world_model.heads["image_head"].compute_loss(
                                video_features,
                                target_rgb_video,
                                global_step=epoch * ipe + itr,
                            )
                        rates["info/image_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_image_losses = {k: v.mean() for k, v in encoder_image_losses.items()}
                        # the weight is like that because we train the head on encoder features and potentially on predictor too
                        loss = encoder_image_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["image_head"].backward(loss)
                        encoder_image_losses = {
                            "encoder_image_" + k: v.item() for k, v in encoder_image_losses.items()
                        }
                        total_stats.update(encoder_image_losses)
                    if "state_head" in world_model.heads and train_heads:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_state_losses = world_model.heads["state_head"].compute_loss(
                                video_features, None, state[:, ::tubelet_size_enc]
                            )
                        rates["info/state_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_state_losses = {k: v.mean() for k, v in encoder_state_losses.items()}
                        loss = encoder_state_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["state_head"].backward(loss)
                        encoder_state_losses = {
                            "encoder_state_" + k: v.item() for k, v in encoder_state_losses.items()
                        }
                        total_stats.update(encoder_state_losses)
                    if "reward_head" in world_model.heads and train_heads:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_reward_losses = world_model.heads["reward_head"].compute_loss(
                                video_features, reward[:, ::tubelet_size_enc]
                            )
                        rates["info/reward_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_reward_losses = {k: v.mean() for k, v in encoder_reward_losses.items()}
                        loss = encoder_reward_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["reward_head"].backward(loss)
                        encoder_reward_losses = {
                            "encoder_reward_" + k: v.item() for k, v in encoder_reward_losses.items()
                        }
                        total_stats.update(encoder_reward_losses)
                    # 3. TRAIN HEADS ON FEATURES FROM PREDICTOR
                    if train_heads_on_predictor and train_heads:
                        if "image_head" in world_model.heads:
                            target_rgb_video = world_model.heads["image_head"].preprocess_rgb(
                                obs["visual"][:, ::tubelet_size_enc]
                            )
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                predictor_image_losses = world_model.heads["image_head"].compute_loss(
                                    pred_video_features[:, :-1].detach(),
                                    target_rgb_video[:, 1:],
                                    global_step=epoch * ipe + itr,
                                )
                            rates["info/image_head/examples_seen"] += pred_video_features.shape[0] * (
                                pred_video_features.shape[1] - 1
                            )
                            predictor_image_losses = {k: v.mean() for k, v in predictor_image_losses.items()}
                            if train:
                                world_model.heads["image_head"].backward(predictor_image_losses["loss"])
                            predictor_image_losses = {
                                "predictor_image_" + k: v.item() for k, v in predictor_image_losses.items()
                            }
                            total_stats.update(predictor_image_losses)
                        if "state_head" in world_model.heads:
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                predictor_state_losses = world_model.heads["state_head"].compute_loss(
                                    video_features, None, state[:, ::tubelet_size_enc]
                                )
                            rates["info/state_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                            predictor_state_losses = {k: v.mean() for k, v in predictor_state_losses.items()}
                            loss = predictor_state_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                            if train:
                                world_model.heads["state_head"].backward(predictor_state_losses["loss"])
                            predictor_state_losses = {
                                "predictor_state_" + k: v.item() for k, v in predictor_state_losses.items()
                            }
                            total_stats.update(predictor_state_losses)
                    # 4. TRAIN PREDICTOR ON FURTHER AUTOREGRESSIVE ROLLOUT
                    if rollout_steps > 1 and train and train_predictor:
                        if do_sequential_rollout:
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                if train_rollout_prefixes == "random":
                                    prefixes = torch.randint(video_features.shape[1] - rollout_steps, size=(1,))
                                elif train_rollout_prefixes == "first":
                                    prefixes = [0]
                                elif train_rollout_prefixes == "all":
                                    prefixes = list(range(video_features.shape[1] - rollout_steps))
                                stats = defaultdict(list)
                                for t in prefixes:
                                    rollout_losses, total_rollout_loss, _, _ = world_model.rollout(
                                        video_features=video_features,
                                        pred_video_features=pred_video_features,
                                        proprio_features=proprio_features,
                                        pred_proprio_features=pred_proprio_features,
                                        action_features=action_features,
                                        action_noise=0.0,
                                        # we have `len(prefixes)` loss terms so weight them equally
                                        loss_weight=1.0 / len(prefixes),
                                        rollout_steps=rollout_steps - 1,
                                        rollout_stop_gradient=rollout_stop_gradient,
                                        ctxt_window=ctxt_window_train_rollout,
                                        mode="sequential",
                                        t=t,
                                    )
                                    total_transition_loss += total_rollout_loss
                                    for k in rollout_losses:
                                        stats[k].append(rollout_losses[k])
                                stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                for k in stats:
                                    for j in range(len(stats[k])):
                                        train_rollout_result[f"train_rollout/{k}/{j+2}"] = stats[k][j].item()
                        if do_parallel_rollout:
                            gt_prob = next(rollout_sampling_scheduler)
                            rates["info/transition_model/sampling_gt_prob"] = gt_prob
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                stats = defaultdict(list)
                                rollout_parallel_losses, total_rollout_parallel_loss, _, _ = world_model.rollout(
                                    video_features=video_features,
                                    proprio_features=proprio_features,
                                    action_features=action_features,
                                    pred_video_features=pred_video_features,
                                    pred_proprio_features=pred_proprio_features,
                                    action_noise=0.0,
                                    loss_weight=1.0,
                                    rollout_steps=rollout_steps - 1,
                                    rollout_stop_gradient=rollout_stop_gradient,
                                    ctxt_window=ctxt_window_train_rollout,
                                    mode="parallel",
                                    gt_prob=gt_prob,
                                    prepend_gt=prepend_gt_rollout_parallel,
                                )
                                total_transition_loss += total_rollout_parallel_loss
                                for k in rollout_parallel_losses:
                                    stats[k].append(rollout_parallel_losses[k])
                                stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                for k in stats:
                                    for j in range(len(stats[k])):
                                        parallel_rollout_result[f"train_rollout_parallel/{k}/{j+2}"] = stats[k][
                                            j
                                        ].item()
                    total_stats.update(train_rollout_result)
                    total_stats.update(parallel_rollout_result)
                    total_stats.update(planner_last_stats)

                    # Construct a clean losses dict that aggregates all relevant training losses
                    losses = {}
                    losses["predictor_loss"] = (
                        total_transition_loss.item()
                        if isinstance(total_transition_loss, torch.Tensor)
                        else total_transition_loss
                    )
                    losses["head_loss"] = (
                        total_head_loss.item() if isinstance(total_head_loss, torch.Tensor) else total_head_loss
                    )
                    losses["loss"] = losses["predictor_loss"] + losses["head_loss"]
                    if planner_identified_enabled:
                        losses["planner_landscape_loss"] = planner_last_stats["planner_landscape_loss"]

                    grad_stats, optim_stats = {}, {}
                    # 5. OPTIMIZATION STEP
                    # so far, we only computed losses and ran .backward() so accumulate gradients on several objectives. Below is the
                    # only place where we perform an actual optimization step
                    if train:
                        if train_heads:
                            for name in world_model.heads.keys():
                                # If not train_heads, world_model.heads[name].model.module.decoder_embed.weight.grad should be None
                                grad_stats[name], optim_stats[name] = world_model.heads[name].optimization_step()
                        if train_predictor:
                            world_model.backward(total_transition_loss / gradient_accumulation_steps)
                            if optimizer_boundary:
                                if planner_identified_enabled:
                                    planner_result = planner_landscape_result(
                                        world_model,
                                        get_planner_batch(),
                                        device=device,
                                        dtype=dtype,
                                        mixed_precision=mixed_precision,
                                        target_energy_eps=float(
                                            cfgs_planner_identified.get(
                                                "target_energy_eps",
                                                1.0e-8,
                                            )
                                        ),
                                        target_energy_relative_floor=float(
                                            cfgs_planner_identified.get(
                                                "target_energy_relative_floor",
                                                0.0,
                                            )
                                        ),
                                    )
                                    planner_last_stats.update(planner_result.stats)
                                    total_stats.update(planner_last_stats)
                                    losses["planner_landscape_loss"] = planner_result.loss
                                    install_planner_scale_gradient(
                                        world_model,
                                        planner_result.scale_gradient,
                                    )
                                (
                                    grad_stats["transition_model"],
                                    optim_stats["transition_model"],
                                ) = world_model.optimization_step(
                                    separately_clipped_parameters=(
                                        (planner_input_scale(world_model).log_scale,)
                                        if planner_identified_enabled
                                        else None
                                    )
                                )
                                transition_grad_stats = grad_stats["transition_model"]
                                if (
                                    planner_identified_enabled
                                    and transition_grad_stats is not None
                                ):
                                    total_stats[
                                        "optim/planner_scale/grad_norm_before_clip"
                                    ] = transition_grad_stats.separate_global_norm
                        for key in list(grad_stats.keys()):
                            grad_stats[f"optim/{key}/grad_norm"] = (
                                grad_stats[key].global_norm if grad_stats[key] is not None else 0.0
                            )
                            del grad_stats[key]
                        for key in list(optim_stats.keys()):
                            optim_stats[f"optim/{key}/first_moment"] = (
                                optim_stats[key].get("exp_avg").avg if optim_stats[key] is not None else 0.0
                            )
                            optim_stats[f"optim/{key}/second_moment"] = (
                                optim_stats[key].get("exp_avg_sq").avg if optim_stats[key] is not None else 0.0
                            )
                            del optim_stats[key]
                    total_stats.update(grad_stats)
                    total_stats.update(optim_stats)
                    total_stats.update(dict(rates))
                    # 6. ONCE IN A WHILE DO LONG-ROLLOUT EVALUATION WITH DATASET ACTIONS OR RANDOM ACTIONS
                    eval_rollout_result = {}
                    image_stats = {}
                    if not train:
                        world_model.eval()

                        @torch.no_grad
                        def val_rollout(
                            video_features,
                            action_features,
                            proprio_features,
                            pred_video_features,
                            pred_proprio_features,
                            gt_obs,
                            gt_state,
                            prefix_rollout_result=None,
                            val_rollout_steps=5,
                            ctxt_window=None,
                        ):
                            """
                            gt_obs:
                                visual: (B, T, C, H, W)
                            """
                            rollout_steps = min(val_rollout_steps, video_features.shape[1] - 1)
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                prefixes = range(video_features.shape[1] - rollout_steps)
                                val_rollout_result = {}

                                # Helper function to run rollout with different action noise levels and decode heads
                                def run_rollout_and_decode(action_noise, rollout_prefix):
                                    """Run rollout with given action noise and decode with image/state heads."""
                                    stats = defaultdict(list)
                                    for t in prefixes:
                                        rollout_losses, _, last_vid_feats, last_prop_feats = world_model.rollout(
                                            video_features=video_features,
                                            pred_video_features=pred_video_features,
                                            proprio_features=proprio_features,
                                            pred_proprio_features=pred_proprio_features,
                                            action_features=action_features,
                                            action_noise=action_noise,
                                            rollout_steps=rollout_steps,
                                            rollout_stop_gradient=True,
                                            debug=action_noise == 0.0,
                                            ctxt_window=ctxt_window,
                                            mode="sequential",
                                            t=t,
                                        )
                                        # last_vid_feats: [B T V H W D]
                                        for k in rollout_losses:
                                            stats[k].append(rollout_losses[k])

                                    image_samples = None
                                    if "image_head" in world_model.heads:
                                        image_samples = world_model.heads["image_head"].decode(
                                            last_vid_feats[:, -rollout_steps - 1 :]
                                        )
                                        for h in range(1, image_samples.shape[1]):
                                            key = f"{rollout_prefix}/lpips/{h}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            v = lpips(
                                                image_samples.squeeze(2)[:, h]
                                                .permute(0, 3, 1, 2)
                                                .to(world_model.device, dtype=torch.float32)
                                                / 255.0,
                                                gt_obs["visual"][:, h * tubelet_size_enc].to(
                                                    world_model.device, dtype=torch.float32
                                                ),
                                            ).mean()
                                            val_rollout_result[key] = v.detach().cpu().item()

                                    if "state_head" in world_model.heads:
                                        target_states = gt_state[:, ::tubelet_size_enc][:, -rollout_steps - 1 :]
                                        state_loss = world_model.heads["state_head"].compute_loss(
                                            last_vid_feats[:, -rollout_steps - 1 :],
                                            None,
                                            target_states,
                                            reduce_mean=False,
                                        )
                                        for k in state_loss:
                                            for j in range(state_loss[k].shape[1]):
                                                key = f"{rollout_prefix}/encoder_state_{k}/{j}"
                                                if prefix_rollout_result is not None:
                                                    key = f"{prefix_rollout_result}/{key}"
                                                val_rollout_result[key] = state_loss[k][:, j].mean().item()

                                    # Aggregate rollout losses
                                    stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                    for k in stats:
                                        for j in range(len(stats[k])):
                                            key = f"{rollout_prefix}/{k}/{j+1}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            val_rollout_result[key] = stats[k][j].item()

                                    return image_samples

                                eval_image_samples = run_rollout_and_decode(
                                    action_noise=0.0, rollout_prefix="val_rollout"
                                )

                                # Optionally decode position from ground truth visual features
                                if "state_head" in world_model.heads and data_traj_decode_gt:
                                    target_states = gt_state[:, ::tubelet_size_enc][:, -rollout_steps - 1 :]
                                    decode_gt_state_loss = world_model.heads["state_head"].compute_loss(
                                        video_features[:, -rollout_steps - 1 :], None, target_states, reduce_mean=False
                                    )
                                    for k in decode_gt_state_loss:
                                        for j in range(decode_gt_state_loss[k].shape[1]):
                                            key = f"val_rollout/decode_gt_state_{k}/{j}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            val_rollout_result[key] = decode_gt_state_loss[k][:, j].mean().item()

                                noisy_eval_image_samples = run_rollout_and_decode(
                                    action_noise=0.05, rollout_prefix="noisy_val_rollout"
                                )

                                if "image_head" in world_model.heads:
                                    t = eval_image_samples.shape[1]
                                    b = gt_obs["visual"].shape[0]
                                    # b = gt_obs["visual"][:, ::tubelet_size_enc].shape[0]
                                    b = min(4, b)
                                    rgb_v = inverse_transform(gt_obs["visual"][:, ::tubelet_size_enc].cpu())
                                    rgb_v = (255.0 * rgb_v).clip(0.0, 255.0).to(torch.uint8)
                                    rgb_v = rearrange(rgb_v, "b t (v c) h w -> b t v h w c", c=3)
                                    rgb = torch.stack([eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
                                    rgb = (
                                        rearrange(
                                            rgb,
                                            "b t e v h w c -> (b v e h) (t w) c",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    rgb_noised = torch.stack([noisy_eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
                                    rgb_noised = (
                                        rearrange(
                                            rgb_noised,
                                            "b t e v h w c -> (b v e h) (t w) c",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    animation = torch.stack(
                                        [rgb_v[:, -t:], eval_image_samples, noisy_eval_image_samples], dim=2
                                    )[:b]
                                    animation = (
                                        rearrange(
                                            animation,
                                            "b t e v h w c -> t c (b v h) (e w)",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                else:
                                    rgb, rgb_noised, animation = None, None, None
                            return rgb, rgb_noised, animation, val_rollout_result

                        if do_data_traj_rollout_eval:
                            rgb, rgb_noised, animation, eval_rollout_result = val_rollout(
                                video_features,
                                action_features,
                                proprio_features,
                                pred_video_features,
                                pred_proprio_features,
                                gt_obs=obs,
                                prefix_rollout_result="data_traj",
                                val_rollout_steps=cfgs_data_traj_rollout_eval.get("data_traj_eval_rollout_steps", 3),
                                ctxt_window=cfgs_data_traj_rollout_eval.get("data_traj_eval_ctxt_window", None),
                                gt_state=state,
                            )
                        if do_energy_landscape_eval:
                            gt_visual = inverse_transform(obs["visual"].cpu())
                            gt_visual = (255.0 * gt_visual).clip(0.0, 255.0).to(torch.uint8)
                            gt_proprio = preprocessor.denormalize_proprios(obs["proprio"].cpu())
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                energy_landscape = world_model.compute_energy_landscape(
                                    video_features,
                                    action_features,
                                    proprio_features,
                                    gt_visual,
                                    gt_proprio,
                                    rollout_steps=cfgs_energy_landscape_eval.get("energy_landscape_rollout_steps", 1),
                                    ctxt_window=cfgs_energy_landscape_eval.get("energy_landscape_ctxt_window", None),
                                    proprio_dim=traj_dataset.proprio_dim,
                                    action_dim=traj_dataset.action_dim,
                                    actions_per_vid_feat=actions_per_vid_feat,
                                    dataset_path=dataset_paths[0],
                                    preprocessor=preprocessor,
                                )
                            image_stats.update(
                                {
                                    "data_traj/energy_landscape": energy_landscape,
                                }
                            )
                        if "image_head" in world_model.heads:
                            if do_data_traj_rollout_eval:
                                image_stats.update(
                                    {
                                        "data_traj/image_rollouts": wandb.Image(rgb),
                                        "data_traj/image_rollouts_noisy_actions": wandb.Image(rgb_noised),
                                        "data_traj/image_animated_rollout": wandb.Video(animation, fps=6),
                                    }
                                )
                        world_model.train()
                    total_stats.update(eval_rollout_result)
                    return (
                        float(losses["loss"]),
                        losses,
                        optim_stats,
                        total_stats,
                        image_stats,
                    )

                # In train mode, image_stats is empty
                if not light_eval_only_mode:
                    obs, action, state, reward, masks_enc, masks_pred = get_batch()
                    (loss, losses, optim_stats, total_stats, image_stats), gpu_etime_ms = gpu_timer(
                        lambda: step_model(obs, action, state, reward, train=True)
                    )
                    iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
                    loss_meter.update(loss)
                    gpu_time_meter.update(gpu_etime_ms)
                    wall_time_meter.update(iter_elapsed_time_ms)
                    if (
                        canary_optimizer_steps
                        and planner_identified_enabled
                        and (itr + 1) % gradient_accumulation_steps == 0
                    ):
                        canary_valid_fraction_meter.update(
                            float(total_stats["planner_valid_group_fraction"])
                        )
                        canary_landscape_loss_meter.update(
                            float(losses["planner_landscape_loss"])
                        )
                        canary_sign_accuracy_meter.update(
                            float(total_stats["planner_valid_pairwise_sign_accuracy"])
                        )
                        canary_scale_gradient_abs_meter.update(
                            abs(float(total_stats["planner_scale_gradient"]))
                        )
                        canary_effective_energy_floor_meter.update(
                            float(
                                total_stats[
                                    "planner_target_energy_effective_floor"
                                ]
                            )
                        )
                        canary_min_to_median_energy_ratio_meter.update(
                            float(
                                total_stats[
                                    "planner_target_energy_min_to_median_ratio"
                                ]
                            )
                        )
                    if train_csv_logger is None:  # Initialize the logger once
                        train_csv_logger = create_csv_logger(losses, total_stats, train=True)
                else:
                    losses = {}
                    total_stats = {}

                if itr % light_eval_freq == light_eval_freq - 1 and val_loader_iters is not None:
                    # image_stats overrides the empty image_stats from the train step at same itr
                    image_stats, eval_losses, eval_total_stats = {}, {}, {}
                    # Then, use non-distributed validation data for visualization on rank 0
                    if val_viz_rank0_loader and rank == 0:
                        viz_obs, viz_action, viz_state, viz_reward, _, _ = get_viz_batch()
                        if viz_obs is not None:
                            with torch.no_grad():
                                (_, _, _, _, viz_img_stats), _ = gpu_timer(
                                    lambda: step_model(viz_obs, viz_action, viz_state, viz_reward, train=False)
                                )
                                if light_eval_only_mode:
                                    image_stats.update(
                                        {f"light_eval_only/epoch-{epoch+1}/{k}": v for k, v in viz_img_stats.items()}
                                    )
                                else:
                                    image_stats.update(viz_img_stats)
                    # First, use distributed validation data for metrics
                    for idx in range(len(val_loader_iters)):
                        obs, action, state, reward, masks_enc, masks_pred = get_batch(train=False, idx=idx)
                        with torch.no_grad():
                            (_, val_losses, _, val_total_stats, val_img_stats), gpu_etime_ms = gpu_timer(
                                lambda: step_model(obs, action, state, reward, train=False)
                            )
                            if not light_eval_only_mode:
                                eval_losses.update({f"eval_data/load-{idx}/{k}": v for k, v in val_losses.items()})
                                eval_total_stats.update(
                                    {f"eval_data/load-{idx}/{k}": v for k, v in val_total_stats.items()}
                                )
                            # Only use distributed batch images if we don't have visualization images
                            if not val_viz_rank0_loader:
                                prefixed_stats = {f"load-{idx}/{k}": v for k, v in val_img_stats.items()}
                                if light_eval_only_mode:
                                    image_stats.update(
                                        {f"light_eval_only/epoch-{epoch+1}/{k}": v for k, v in prefixed_stats.items()}
                                    )
                                else:
                                    image_stats.update(prefixed_stats)
                    if eval_csv_logger is None:  # Initialize the logger once
                        eval_csv_logger = create_csv_logger(eval_losses, eval_total_stats, train=False)
                else:
                    eval_losses = {}
                    eval_total_stats = {}

                # -- Logging
                def log_stats():
                    trainer.log(epoch, itr, losses, total_stats, eval_losses, eval_total_stats, image_stats)
                    if not light_eval_only_mode:
                        log_values = [epoch + 1, itr, loss, gpu_etime_ms, iter_elapsed_time_ms]
                        for key in train_csv_logger_columns[5:]:
                            if key in losses:
                                value = losses[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            elif key in total_stats:
                                value = total_stats[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            else:
                                log_values.append(0.0)
                        train_csv_logger.log(*log_values)
                        if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                            logger.info(
                                "[epoch %d/%d | global opt %d/%d | "
                                "pass opt %d/%d | micro %d/%d | %.1f%%] "
                                "[mem: %.2e] "
                                "[gpu: %.1f ms]"
                                "[wall: %.1f ms]"
                                % (
                                    epoch + 1,
                                    num_epochs,
                                    epoch * optimizer_steps_per_epoch
                                    + (itr + 1)
                                    // gradient_accumulation_steps,
                                    optimizer_step_budget,
                                    (itr + 1) // gradient_accumulation_steps,
                                    optimizer_steps_per_epoch,
                                    itr + 1,
                                    ipe,
                                    100.0 * (itr + 1) / ipe,
                                    torch.cuda.max_memory_allocated() / 1024.0**2,
                                    gpu_time_meter.avg,
                                    wall_time_meter.avg,
                                )
                            )
                    if itr % light_eval_freq == light_eval_freq - 1:
                        log_values = [epoch + 1, itr]
                        for key in eval_csv_logger_columns[2:]:
                            if key in eval_losses:
                                value = eval_losses[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            elif key in eval_total_stats:
                                log_values.append(eval_total_stats[key])
                            else:
                                log_values.append(0.0)
                        eval_csv_logger.log(*log_values)

                log_stats()
                if not light_eval_only_mode:
                    assert not np.isnan(loss), "loss is nan"
                if (
                    canary_optimizer_steps
                    and (itr + 1) % gradient_accumulation_steps == 0
                    and (itr + 1) // gradient_accumulation_steps
                    >= canary_optimizer_steps
                ):
                    canary_complete = True
                    break
                if (
                    exact_optimizer_step_budget
                    and (itr + 1) % gradient_accumulation_steps == 0
                    and epoch * optimizer_steps_per_epoch
                    + (itr + 1) // gradient_accumulation_steps
                    >= optimizer_step_budget
                ):
                    step_budget_complete = True
                    break
            logger.info("avg. loss %.3f" % loss_meter.avg)

            planner_validation_metrics = None
            planner_validation_total_steps = None
            planner_validation_step_in_pass = 0
            planner_validation_boundary = None
            if planner_identified_enabled and not light_eval_only_mode:
                if canary_complete:
                    planner_validation_total_steps = int(canary_optimizer_steps)
                    planner_validation_step_in_pass = int(canary_optimizer_steps)
                    planner_validation_boundary = "canary"
                elif step_budget_complete:
                    planner_validation_total_steps = int(optimizer_step_budget)
                    planner_validation_step_in_pass = (
                        planner_validation_total_steps % optimizer_steps_per_epoch
                    )
                    planner_validation_boundary = (
                        "pass"
                        if planner_validation_step_in_pass == 0
                        else "final_partial_pass"
                    )
                elif itr + 1 == ipe:
                    planner_validation_total_steps = (
                        epoch + 1
                    ) * optimizer_steps_per_epoch
                    planner_validation_boundary = "pass"
                if planner_validation_total_steps is not None:
                    planner_validation_metrics = evaluate_planner_validation()
                    completed_passes = (
                        planner_validation_total_steps // optimizer_steps_per_epoch
                    )
                    persist_planner_validation(
                        planner_validation_metrics,
                        boundary=planner_validation_boundary,
                        total_optimizer_steps=planner_validation_total_steps,
                        checkpoint_epoch=completed_passes,
                        optimizer_step_in_epoch=planner_validation_step_in_pass,
                        eligible_for_selection=(
                            planner_validation_boundary == "pass"
                            and planner_validation_total_steps > 0
                        ),
                        training_complete=(
                            step_budget_complete
                            or (
                                not exact_optimizer_step_budget
                                and epoch == num_epochs - 1
                                and planner_validation_boundary == "pass"
                            )
                        ),
                    )

            boundary_stop_complete = (
                stop_after_complete_passes > 0
                and not canary_complete
                and not step_budget_complete
                and itr + 1 == ipe
                and epoch + 1 >= stop_after_complete_passes
            )

            if canary_complete:
                if rank == 0:
                    scale = planner_input_scale(world_model)
                    log_scale_drift_per_1000_steps = (
                        (
                            planner_validation_metrics["planner_log_scale"]
                            - planner_initial_validation_metrics["planner_log_scale"]
                        )
                        * 1000.0
                        / canary_optimizer_steps
                    )
                    max_canary_drift = cfgs_planner_identified.get(
                        "canary_max_abs_log_scale_drift_per_1000_steps"
                    )
                    canary_drift_passed = (
                        max_canary_drift is None
                        or abs(log_scale_drift_per_1000_steps)
                        <= float(max_canary_drift)
                    )
                    require_non_degradation = bool(
                        cfgs_planner_identified.get(
                            "canary_require_heldout_non_degradation",
                            False,
                        )
                    )
                    canary_validation_passed = (
                        not require_non_degradation
                        or planner_validation_metrics["planner_landscape_loss"]
                        <= planner_initial_validation_metrics[
                            "planner_landscape_loss"
                        ]
                    )
                    canary_gate_passed = (
                        canary_drift_passed and canary_validation_passed
                    )
                    summary = {
                        "status": (
                            "CANARY_COMPLETE"
                            if canary_gate_passed
                            else "CANARY_REJECTED"
                        ),
                        "canary_gate_passed": canary_gate_passed,
                        "canary_log_scale_drift_passed": canary_drift_passed,
                        "canary_heldout_non_degradation_passed": (
                            canary_validation_passed
                        ),
                        "optimizer_steps": canary_optimizer_steps,
                        "microbatches": (
                            canary_optimizer_steps * gradient_accumulation_steps
                        ),
                        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                        "microbatches_per_epoch": ipe,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "planner_history_size": int(
                            cfgs_custom.get("num_hist", 1)
                        ),
                        "planner_scale_lr_multiplier": float(
                            cfgs_planner_identified.get(
                                "scale_lr_multiplier",
                                1.0,
                            )
                        ),
                        "target_energy_eps": float(
                            cfgs_planner_identified.get(
                                "target_energy_eps",
                                1.0e-8,
                            )
                        ),
                        "target_energy_relative_floor": float(
                            cfgs_planner_identified.get(
                                "target_energy_relative_floor",
                                0.0,
                            )
                        ),
                        "final_log_scale": float(scale.log_scale.detach().cpu()),
                        "final_scale": float(scale.input_scale.detach().cpu()),
                        "heldout_initial": planner_initial_validation_metrics,
                        "heldout_final": planner_validation_metrics,
                        "log_scale_drift_per_1000_optimizer_steps": (
                            log_scale_drift_per_1000_steps
                        ),
                        "canary_max_abs_log_scale_drift_per_1000_steps": (
                            None
                            if max_canary_drift is None
                            else float(max_canary_drift)
                        ),
                        "canary_require_heldout_non_degradation": (
                            require_non_degradation
                        ),
                        "valid_group_fraction": {
                            "mean": canary_valid_fraction_meter.avg,
                            "min": canary_valid_fraction_meter.min,
                            "max": canary_valid_fraction_meter.max,
                        },
                        "planner_landscape_loss": {
                            "mean": canary_landscape_loss_meter.avg,
                            "min": canary_landscape_loss_meter.min,
                            "max": canary_landscape_loss_meter.max,
                        },
                        "planner_valid_pairwise_sign_accuracy_mean": (
                            canary_sign_accuracy_meter.avg
                        ),
                        "planner_scale_gradient_abs": {
                            "mean": canary_scale_gradient_abs_meter.avg,
                            "max": canary_scale_gradient_abs_meter.max,
                        },
                        "target_energy_effective_floor": {
                            "mean": canary_effective_energy_floor_meter.avg,
                            "min": canary_effective_energy_floor_meter.min,
                            "max": canary_effective_energy_floor_meter.max,
                        },
                        "target_energy_min_to_median_ratio": {
                            "mean": canary_min_to_median_energy_ratio_meter.avg,
                            "min": canary_min_to_median_energy_ratio_meter.min,
                            "max": canary_min_to_median_energy_ratio_meter.max,
                        },
                        "repository_commit": (
                            training_provenance.get("repository_commit")
                            if training_provenance is not None
                            else None
                        ),
                    }
                    summary_path = Path(folder) / "canary_summary.json"
                    temporary_path = summary_path.with_suffix(".json.tmp")
                    temporary_path.write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    temporary_path.replace(summary_path)
                    logger.info(
                        f"[{summary['status']}] summary={summary_path}"
                    )
                break

            if step_budget_complete:
                completed_full_passes, optimizer_step_in_partial_pass = divmod(
                    optimizer_step_budget,
                    optimizer_steps_per_epoch,
                )
                if rank == 0:
                    save_checkpoint(
                        completed_full_passes,
                        latest_path,
                        total_optimizer_step_count=optimizer_step_budget,
                        optimizer_step_in_epoch=optimizer_step_in_partial_pass,
                        training_complete=True,
                        planner_validation_metrics=planner_validation_metrics,
                    )
                    step_file = pref_tag + f"step{optimizer_step_budget}.{latest_format}"
                    step_path = os.path.join(checkpoint_folder, step_file)
                    save_checkpoint(
                        completed_full_passes,
                        step_path,
                        total_optimizer_step_count=optimizer_step_budget,
                        optimizer_step_in_epoch=optimizer_step_in_partial_pass,
                        training_complete=True,
                        planner_validation_metrics=planner_validation_metrics,
                    )
                    if (
                        optimizer_step_in_partial_pass == 0
                        and completed_full_passes > 0
                        and save_every_freq > 0
                    ):
                        pass_file = pref_tag + (
                            f"e{completed_full_passes - 1}.{latest_format}"
                        )
                        pass_path = os.path.join(checkpoint_folder, pass_file)
                        save_checkpoint(
                            completed_full_passes,
                            pass_path,
                            total_optimizer_step_count=optimizer_step_budget,
                            optimizer_step_in_epoch=0,
                            training_complete=True,
                            planner_validation_metrics=planner_validation_metrics,
                        )
                    scale = (
                        planner_input_scale(world_model)
                        if planner_identified_enabled
                        else None
                    )
                    summary = {
                        "status": "TRAINING_COMPLETE",
                        "optimizer_steps": optimizer_step_budget,
                        "optimizer_steps_per_pass": optimizer_steps_per_epoch,
                        "completed_full_passes": completed_full_passes,
                        "optimizer_steps_in_partial_pass": (
                            optimizer_step_in_partial_pass
                        ),
                        "gradient_accumulation_steps": (
                            gradient_accumulation_steps
                        ),
                        "final_log_scale": (
                            float(scale.log_scale.detach().cpu())
                            if scale is not None
                            else None
                        ),
                        "final_scale": (
                            float(scale.input_scale.detach().cpu())
                            if scale is not None
                            else None
                        ),
                        "planner_heldout_final": planner_validation_metrics,
                        "planner_best_checkpoint": (
                            planner_validation_document.get("best", {}).get(
                                "checkpoint"
                            )
                            if planner_identified_enabled
                            and planner_validation_document.get("best")
                            else None
                        ),
                        "repository_commit": (
                            training_provenance.get("repository_commit")
                            if training_provenance is not None
                            else None
                        ),
                    }
                    summary_path = Path(folder) / "training_summary.json"
                    temporary_path = summary_path.with_suffix(".json.tmp")
                    temporary_path.write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    temporary_path.replace(summary_path)
                    logger.info(
                        "[STEP BUDGET COMPLETE] "
                        f"optimizer_steps={optimizer_step_budget} "
                        f"summary={summary_path}"
                    )
                break

            # -- Save Last
            if not light_eval_only_mode:
                if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
                    if rank == 0:
                        save_checkpoint(
                            epoch + 1,
                            latest_path,
                            training_complete=(
                                not exact_optimizer_step_budget
                                and epoch == num_epochs - 1
                            ),
                            planner_validation_metrics=planner_validation_metrics,
                        )
                        if save_every_freq > 0 and epoch % save_every_freq == 0:
                            save_every_file = pref_tag + f"e{epoch}.{latest_format}"
                            save_every_path = os.path.join(checkpoint_folder, save_every_file)
                            save_checkpoint(
                                epoch + 1,
                                save_every_path,
                                training_complete=(
                                    not exact_optimizer_step_budget
                                    and epoch == num_epochs - 1
                                ),
                                planner_validation_metrics=planner_validation_metrics,
                            )

            if boundary_stop_complete:
                if rank == 0:
                    scale = (
                        planner_input_scale(world_model)
                        if planner_identified_enabled
                        else None
                    )
                    summary = {
                        "status": "PASS_BOUNDARY_COMPLETE",
                        "requested_complete_passes": (
                            stop_after_complete_passes
                        ),
                        "completed_full_passes": epoch + 1,
                        "optimizer_steps": (
                            (epoch + 1) * optimizer_steps_per_epoch
                        ),
                        "optimizer_steps_per_pass": (
                            optimizer_steps_per_epoch
                        ),
                        "configured_optimizer_step_budget": (
                            optimizer_step_budget
                        ),
                        "gradient_accumulation_steps": (
                            gradient_accumulation_steps
                        ),
                        "final_log_scale": (
                            float(scale.log_scale.detach().cpu())
                            if scale is not None
                            else None
                        ),
                        "final_scale": (
                            float(scale.input_scale.detach().cpu())
                            if scale is not None
                            else None
                        ),
                        "planner_heldout_at_boundary": (
                            planner_validation_metrics
                        ),
                        "planner_best_checkpoint": (
                            planner_validation_document.get("best", {}).get(
                                "checkpoint"
                            )
                            if planner_identified_enabled
                            and planner_validation_document.get("best")
                            else None
                        ),
                        "repository_commit": (
                            training_provenance.get("repository_commit")
                            if training_provenance is not None
                            else None
                        ),
                    }
                    summary_path = Path(folder) / "pass_boundary_summary.json"
                    temporary_path = summary_path.with_suffix(".json.tmp")
                    temporary_path.write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    temporary_path.replace(summary_path)
                    logger.info(
                        "[PASS BOUNDARY COMPLETE] "
                        f"passes={epoch + 1} summary={summary_path}"
                    )
                break

            # -- Launch Planning Eval
            if not light_eval_only_mode and not skip_planning_eval:
                if (epoch % eval_freq == 0) or epoch == (num_epochs - 1):
                    if save_every_freq > 0:
                        checkpoint = (
                            pref_tag + f"latest.{latest_format}"
                            if epoch == (num_epochs - 1)
                            else pref_tag + f"e{epoch}.{latest_format}"
                        )
                    else:
                        checkpoint = pref_tag + f"latest.{latest_format}"
                    launch_planning_evals(
                        rank,
                        epoch + 1,
                        folder,
                        checkpoint,
                        cfgs_plan_evals,
                        cfgs_model,
                        cfgs_data,
                        cfgs_data_aug,
                        "",
                        world_model=world_model,
                        dset=val_traj_dataset,
                        preprocessor=preprocessor,
                        checkpoint_folder=checkpoint_folder,
                    )
                    world_model.train()
    elif unroll_decode_eval_only_mode:
        logger.info("Launching unroll-decode evals only mode")
        checkpoint = pref_tag + f"latest.{latest_format}"
        launch_unroll_decode_eval(
            rank,
            start_epoch,
            folder,
            checkpoint,
            cfgs_unroll_decode_evals,
            cfgs_model,
            cfgs_data,
            cfgs_data_aug,
        )
    else:
        logger.info("Skipping training loop due to plan_only_eval_mode being enabled.")
        checkpoint = pref_tag + f"latest.{latest_format}"
        launch_planning_evals(
            rank,
            start_epoch,
            folder,
            checkpoint,
            cfgs_plan_evals,
            cfgs_model,
            cfgs_data,
            cfgs_data_aug,
            "-plan-only",
            world_model=world_model,
            dset=val_traj_dataset,
            preprocessor=preprocessor,
            checkpoint_folder=checkpoint_folder,
        )


def launch_planning_evals(
    rank,
    epoch,
    folder,
    checkpoint,
    cfgs_plan_evals,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
    tag_suffix,
    world_model=None,
    dset=None,
    preprocessor=None,
    checkpoint_folder=None,
):
    """
    Launch planning evaluations for the current training checkpoint.

    This function generates complete eval configs by merging training model/data settings
    with eval config templates, then either submits distributed eval jobs via sbatch
    or runs them locally (if separate=False).

    The eval config generation flow:
    1. Load eval config templates from cfgs_plan_evals["eval_cfg_paths"]
       (typically located in configs/online_plan_evals/)
    2. Call build_plan_eval_args() to merge training configs (model, data, data_aug)
       with these templates
    3. Either dump configs for debugging or submit/run eval jobs

    Config options in cfgs_plan_evals:
    - dump_eval_configs (bool): If True, dump generated configs to disk and exit early
      without launching evals. The dump directory is automatically derived from
      eval_cfg_paths (e.g., "configs/online_plan_evals/mz/..." -> "configs/dump_online_evals/mz/").
      Output filenames are derived from the template basenames.
    - separate (bool): If True (default), submit eval jobs via sbatch. If False, run
      evals on rank 0 of the current training job.

    To generate eval configs without running training (e.g., for an already-trained model):
    1. Set meta.plan_only_eval_mode: true in your training config
    2. Set evals.dump_eval_configs: true in your training config
    3. Run: python -m app.main --fname <your_config.yaml> --debug
    4. Configs will be saved to configs/dump_online_evals/<env>/ (derived from eval_cfg_paths)

    Args:
        rank: Process rank in distributed training
        epoch: Current training epoch
        folder: Output folder path for the training run
        checkpoint: Checkpoint filename to evaluate
        cfgs_plan_evals: Evaluation configuration dict from training config
        cfgs_model: Model configuration from training config
        cfgs_data: Data configuration from training config
        cfgs_data_aug: Data augmentation configuration from training config
        tag_suffix: Suffix to append to evaluation tags
        world_model: Optional loaded world model (for non-separate eval mode)
        dset: Optional validation dataset (for non-separate eval mode)
        preprocessor: Optional data preprocessor (for non-separate eval mode)
    """
    eval_cfg_paths = cfgs_plan_evals.get("eval_cfg_paths", None)
    eval_nodes = cfgs_plan_evals.get("nodes", None)
    eval_episodes = cfgs_plan_evals.get("eval_episodes", None)
    eval_low_pri = cfgs_plan_evals.get("low_pri", True)
    separate = cfgs_plan_evals.get("separate", True)
    override_cfgs_data = cfgs_plan_evals.get("override_cfgs_data", True)
    override_datasets = cfgs_plan_evals.get("override_datasets", True)
    # task_specification
    evals_obs = cfgs_plan_evals.get("obs", None)
    # planner
    evals_alpha = cfgs_plan_evals.get("alpha", None)
    max_episode_steps = cfgs_plan_evals.get("max_episode_steps", None)
    num_act_stepped = cfgs_plan_evals.get("num_act_stepped", None)
    horizon = cfgs_plan_evals.get("horizon", None)
    evals_decode = cfgs_plan_evals.get("decode", None)
    sum_all_diffs = cfgs_plan_evals.get("sum_all_diffs", None)
    goal_H = cfgs_plan_evals.get("goal_H", None)
    num_elites = cfgs_plan_evals.get("num_elites", None)
    if eval_cfg_paths is not None:
        eval_nodes, eval_tasks_per_node, args_eval, eval_cpus_per_task = build_plan_eval_args(
            app_name="vjepa_wm",
            folder=folder,
            checkpoint=checkpoint,
            eval_cfg_paths=eval_cfg_paths,
            cfgs_model=cfgs_model,
            cfgs_data=cfgs_data,
            cfgs_data_aug=cfgs_data_aug,
            override_cfgs_data=override_cfgs_data,
            override_datasets=override_datasets,
            tag=f"epoch-{epoch}{tag_suffix}",
            evals_decode=evals_decode,
            sum_all_diffs=sum_all_diffs,
            evals_obs=evals_obs,
            evals_alpha=evals_alpha,
            eval_nodes=eval_nodes,
            eval_episodes=eval_episodes,
            max_episode_steps=max_episode_steps,
            num_act_stepped=num_act_stepped,
            horizon=horizon,
            goal_H=goal_H,
            num_elites=num_elites,
            wrapper_kwargs=cfgs_plan_evals.get("wrapper_kwargs", {}),
            checkpoint_folder=checkpoint_folder,
        )

        # Dump eval configs if in dump_eval_configs mode (useful for generating configs without training)
        dump_eval_configs = cfgs_plan_evals.get("dump_eval_configs", False)
        if dump_eval_configs:
            if rank == 0:
                # Deduce dump directory from eval_cfg_paths
                # e.g., "configs/online_plan_evals/mz/ng/..." -> "configs/dump_online_evals/mz/"
                first_template = eval_cfg_paths[0] if eval_cfg_paths else None
                if first_template and "online_plan_evals" in first_template:
                    # Extract environment name from path (e.g., "mz", "pt", "wall", "droid")
                    parts = first_template.split("online_plan_evals/")
                    if len(parts) > 1:
                        env_part = parts[1].split("/")[0]  # Get first directory after online_plan_evals/
                        dump_dir = f"configs/dump_online_evals/{env_part}"
                    else:
                        dump_dir = "configs/dump_online_evals"
                else:
                    dump_dir = "configs/dump_online_evals"
                os.makedirs(dump_dir, exist_ok=True)
                dumped_paths = []
                for i, cfg in enumerate(args_eval):
                    # Derive output filename from the config's tag field
                    # e.g., "online_gc_zeroshot/wall_L2_ng_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96/epoch-50-plan-only"
                    # -> "wall_L2_ng_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96.yaml"
                    tag = cfg.get("tag", None)
                    if tag:
                        tag_parts = tag.split("/")
                        if len(tag_parts) >= 2:
                            output_name = tag_parts[-2] + ".yaml"
                        else:
                            output_name = tag_parts[0] + ".yaml"
                    else:
                        output_name = f"eval_config_{i}.yaml"
                    output_path = os.path.join(dump_dir, output_name)
                    dump_yaml(cfg, output_path)
                    dumped_paths.append(output_path)
                logger.info(f"Dumped {len(args_eval)} eval configs:\n" + "\n".join(f"  - {p}" for p in dumped_paths))
            import sys

            sys.exit(0)

        for i, cfg in enumerate(args_eval):
            args_eval[i] = convert_to_dict_recursive(args_eval[i])

        if separate:
            if rank == 0:
                account, partition, qos = slurm_account_partition_and_qos(low_pri=eval_low_pri)
                logger.info(f"Launching online evals with {account=}, {partition=}, {qos=}")
                with submitit.helpers.clean_env():
                    launch_evals(
                        args_for_evals=args_eval,
                        nodes=eval_nodes,
                        tasks_per_node=eval_tasks_per_node,
                        submitit_folder=os.path.join(folder, "submitit-evals"),
                        account=account,
                        partition=partition,
                        qos=qos,
                        cpus_per_task=eval_cpus_per_task,
                        delay_seconds=5,
                        timeout=120,  # to schedule faster, could be insufficient if using old GPUs making eval slow
                    )
                logger.info(f"Launched online evals from templates {eval_cfg_paths}")
        else:
            from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
            from evals.simu_env_planning.eval import main_distributed_episodes_eval as gc_main_dist

            world_model.eval()
            for i, cfg in tqdm(enumerate(args_eval)):
                eval_tag = cfg.get("tag", None)
                pretrain_folder = cfg.get("folder", None)
                folder = os.path.join(pretrain_folder, "simu_env_planning/")
                if eval_tag is not None:
                    folder = os.path.join(folder, eval_tag)
                cfg["frameskip"] = cfg["model_kwargs"]["data"]["custom"]["frameskip"]
                cfg["work_dir"] = folder
                model = EncPredWM(
                    world_model,
                    action_dim=world_model.action_dim,
                    preprocessor=preprocessor,
                    ctxt_window=cfg["model_kwargs"]["wrapper_kwargs"]["ctxt_window"],
                )
                gc_main_dist(cfg, model=model, dset=dset, preprocessor=preprocessor, rank=rank)


def launch_unroll_decode_eval(
    rank,
    epoch,
    folder,
    checkpoint,
    cfgs_unroll_decode_evals,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
):
    """
    Launch unroll decode evaluations for counterfactual decoding of unrolled predictions.

    This evaluation hardcodes custom actions (e.g., open/close gripper + move up) to generate
    counterfactual decodings, allowing visual comparison of different action scenarios.

    Config options in cfgs_unroll_decode_evals:
    - dump_eval_configs (bool): If True, dump generated configs to disk and exit early
      without launching evals (useful for generating configs without training)
    - specific_video (bool): If True, use a specific video file instead of dataset samples
    - specific_video_path (str): Path to specific video file (npz format)
    - play_in_reverse (bool): If True, reverse the video sequence
    - obs (str): Observation type - "rgb" or "rgb_state"
    - save_decoding_only (bool): If True, only save decoded predictions (not ground truth comparison)
    - repeat_hardcode_act (int): Number of times to repeat the hardcoded action sequence
    - wrapper_kwargs (dict): Model wrapper configuration (same as evals.wrapper_kwargs)
        - ctxt_window (int): Context window size for the model wrapper
        - proprio_mode (str): Proprioception mode (e.g., "compute_new_pose")

    Args:
        rank: Process rank in distributed training
        epoch: Current training epoch
        folder: Output folder path for the training run
        checkpoint: Checkpoint filename to evaluate
        cfgs_unroll_decode_evals: Unroll decode evaluation configuration dict
        cfgs_model: Model configuration from training config
        cfgs_data: Data configuration from training config
        cfgs_data_aug: Data augmentation configuration from training config
    """
    # Build evaluation arguments
    args_eval = build_unroll_decode_eval_args(
        app_name="vjepa_wm",
        folder=folder,
        checkpoint=checkpoint,
        cfgs_model=cfgs_model,
        cfgs_data=cfgs_data,
        cfgs_data_aug=cfgs_data_aug,
        cfgs_unroll_decode_evals=cfgs_unroll_decode_evals,
        tag=f"epoch-{epoch}",
    )

    # Dump eval configs if in dump_eval_configs mode (useful for generating configs without training)
    dump_eval_configs = cfgs_unroll_decode_evals.get("dump_eval_configs", False)
    if dump_eval_configs:
        if rank == 0:
            dump_dir = "configs/dump_online_evals/vjepa_wm/unroll_decode"
            os.makedirs(dump_dir, exist_ok=True)
            for i, cfg in enumerate(args_eval):
                yaml_path = os.path.join(dump_dir, f"unroll_decode_{i}.yaml")
                dump_yaml(cfg, yaml_path)
            logger.info(f"Dumped {len(args_eval)} unroll_decode eval configs to {dump_dir}")
        # All ranks exit early after dumping configs (skip launching evals)
        return

    for i, cfg in enumerate(args_eval):
        args_eval[i] = convert_to_dict_recursive(args_eval[i])

    # Run eval directly on rank 0
    from evals.unroll_decode.eval import main as unroll_decode_main

    if rank == 0:
        for cfg in args_eval:
            unroll_decode_main(cfg)
