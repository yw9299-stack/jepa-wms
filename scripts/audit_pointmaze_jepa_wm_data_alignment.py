#!/usr/bin/env python3

"""Audit PointMaze train/eval alignment without training a model.

The audit always validates StableWM HDF5 statistics and the exact preprocessing
path.  If Meta's original PointMaze files are present, it additionally compares
action/state statistics and frozen DINOv2 latent distributions.
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from app.plan_common.datasets.transforms import make_transforms
from app.plan_common.models.dino import DinoEncoder


NORMALIZE = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def vector_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.shape[0]),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def image_stats(images):
    values = np.asarray(images, dtype=np.float64) / 255.0
    return {
        "count": int(values.shape[0]),
        "channel_mean": values.mean(axis=(0, 1, 2)).tolist(),
        "channel_std": values.std(axis=(0, 1, 2)).tolist(),
        "pixel_min": float(values.min()),
        "pixel_max": float(values.max()),
    }


def transform_images(images, scale):
    transform = make_transforms(
        img_size=224,
        normalize=NORMALIZE,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(float(scale), float(scale)),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    tensor = torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).permute(0, 3, 1, 2)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return transform(tensor)


def sample_h5(path, sample_count, seed):
    with h5py.File(path, "r", swmr=True) as source:
        rows = int(source["pixels"].shape[0])
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(rows, size=min(sample_count, rows), replace=False))
        images = np.asarray(source["pixels"][indices], dtype=np.uint8)
        actions = np.asarray(source["action"][:], dtype=np.float32)
        states = np.asarray(source["observation"][:], dtype=np.float32)
    return images, actions, states, indices


def sample_official_dataset(root, sample_count, seed):
    actions = torch.load(root / "actions.pth", map_location="cpu", weights_only=False).float()
    states = torch.load(root / "states.pth", map_location="cpu", weights_only=False).float()
    lengths = torch.load(root / "seq_lengths.pth", map_location="cpu", weights_only=False).long()
    valid_actions = []
    valid_states = []
    image_locations = []
    for episode, length in enumerate(lengths.tolist()):
        valid_actions.append(actions[episode, :length])
        valid_states.append(states[episode, :length])
        image_locations.extend((episode, frame) for frame in range(length))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(image_locations), size=min(sample_count, len(image_locations)), replace=False)
    by_episode = {}
    for position in chosen.tolist():
        episode, frame = image_locations[position]
        by_episode.setdefault(episode, []).append(frame)
    sampled = []
    for episode, frames in sorted(by_episode.items()):
        video = torch.load(root / "obses" / f"episode_{episode:03d}.pth", map_location="cpu", weights_only=False)
        for frame in frames:
            sampled.append(np.asarray(video[frame], dtype=np.uint8))
    return (
        np.stack(sampled),
        torch.cat(valid_actions).numpy(),
        torch.cat(valid_states).numpy(),
    )


@torch.inference_mode()
def encode_latents(images, device, batch_size):
    transformed = transform_images(images, 1.0)
    encoder = DinoEncoder("dinov2_vits14", "x_norm_patchtokens").to(device).eval()
    pooled = []
    for start in range(0, len(transformed), batch_size):
        batch = transformed[start : start + batch_size].to(device)
        pooled.append(encoder(batch).mean(dim=1).float().cpu())
    return torch.cat(pooled).numpy()


def latent_comparison(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1.0e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1.0e-12)
    similarities = left_norm @ right_norm.T
    centroid_left = left.mean(axis=0)
    centroid_right = right.mean(axis=0)
    centroid_cosine = float(
        centroid_left.dot(centroid_right)
        / max(np.linalg.norm(centroid_left) * np.linalg.norm(centroid_right), 1.0e-12)
    )
    pooled_std = np.sqrt(0.5 * (left.var(axis=0) + right.var(axis=0)) + 1.0e-12)
    return {
        "centroid_cosine_similarity": centroid_cosine,
        "h5_to_official_nearest_cosine_mean": float(similarities.max(axis=1).mean()),
        "official_to_h5_nearest_cosine_mean": float(similarities.max(axis=0).mean()),
        "standardized_centroid_shift_mean": float(
            np.mean(np.abs(centroid_left - centroid_right) / pooled_std)
        ),
        "h5_latent_norm_mean": float(np.linalg.norm(left, axis=1).mean()),
        "official_latent_norm_mean": float(np.linalg.norm(right, axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--current-config", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--official-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    for description, path in (
        ("StableWM HDF5", args.source_h5),
        ("current training config", args.current_config),
        ("official training config", args.official_config),
    ):
        if not path.is_file():
            raise SystemExit(f"[STOP] missing {description}: {path}")

    current = yaml.safe_load(args.current_config.read_text())
    official = yaml.safe_load(args.official_config.read_text())
    h5_images, h5_actions, h5_states, indices = sample_h5(
        args.source_h5, args.samples, args.seed
    )
    current_processed = transform_images(h5_images, 1.0)
    official_train_processed = transform_images(h5_images, 1.777)
    preprocessing_max_abs = float((current_processed - official_train_processed).abs().max())

    eval_utils = Path("evals/utils.py").read_text()
    eval_forces_identity_crop = "random_resize_scale=(1.0, 1.0)" in eval_utils
    result = {
        "schema_version": 1,
        "training": False,
        "sample_rows": indices.tolist(),
        "config_differences": {
            "current_dataset_type": current["data"].get("dataset_type"),
            "official_dataset_type": official["data"].get("dataset_type"),
            "current_dataset_seed": current["data"].get("seed"),
            "official_dataset_seed": official["data"].get("seed"),
            "current_train_resize_scale": current["data_aug"].get("random_resize_scale"),
            "official_train_resize_scale": official["data_aug"].get("random_resize_scale"),
            "online_eval_forces_resize_scale_1": eval_forces_identity_crop,
            "same_h5_frame_current_vs_official_train_preprocess_max_abs": preprocessing_max_abs,
        },
        "stablewm_h5": {
            "path": str(args.source_h5.resolve()),
            "images": image_stats(h5_images),
            "actions": vector_stats(h5_actions),
            "states": vector_stats(h5_states),
        },
        "official_dataset": {"available": False},
    }

    official_root = args.official_dataset
    required = ("actions.pth", "states.pth", "seq_lengths.pth", "obses")
    if official_root is not None and all((official_root / item).exists() for item in required):
        official_images, official_actions, official_states = sample_official_dataset(
            official_root, args.samples, args.seed
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h5_latents = encode_latents(h5_images, device, args.batch_size)
        official_latents = encode_latents(official_images, device, args.batch_size)
        result["official_dataset"] = {
            "available": True,
            "path": str(official_root.resolve()),
            "images": image_stats(official_images),
            "actions": vector_stats(official_actions),
            "states": vector_stats(official_states),
            "frozen_dinov2_vits14": latent_comparison(h5_latents, official_latents),
        }

    result["interpretation"] = {
        "preprocessing": (
            "The nominal 1.777 versus 1.0 training scale is not an image difference for square inputs "
            "when the crop implementation falls back to the whole frame."
            if preprocessing_max_abs == 0.0
            else "The two training preprocessing configurations produce different tensors."
        ),
        "missing_official_dataset": (
            "Action/state and latent-domain comparisons require Meta's point_maze dataset. "
            "The official-checkpoint online sanity test remains valid without it."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("[JEPA-WM PointMaze data/preprocessing alignment audit]")
    print(f"  online_eval_forces_scale1={eval_forces_identity_crop}")
    print(f"  same_frame scale1-vs-scale1.777 max_abs={preprocessing_max_abs:.8g}")
    print(f"  official_dataset_available={result['official_dataset']['available']}")
    if result["official_dataset"]["available"]:
        latent = result["official_dataset"]["frozen_dinov2_vits14"]
        print(f"  DINO centroid cosine={latent['centroid_cosine_similarity']:.6f}")
        print(f"  standardized centroid shift={latent['standardized_centroid_shift_mean']:.6f}")
    print(f"[saved] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
