# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Direct sliced access to StableWorldModel-style HDF5 trajectories.

The PointMaze PI-LTC experiments use the exact HDF5 source that produced the
counterfactual planner sidecar.  Reading only the four requested frames avoids
materializing a complete 100-frame episode for every randomly ordered clip.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class StableWMH5Metadata:
    """Dataset metadata exposed through the ordinary JEPA-WM training API."""

    def __init__(self, source_h5: str | Path, normalize_action: bool = True):
        self.source_h5_path = Path(source_h5)
        if not self.source_h5_path.is_file():
            raise FileNotFoundError(self.source_h5_path)
        with h5py.File(self.source_h5_path, "r", swmr=True) as source:
            required = (
                "ep_idx",
                "ep_len",
                "ep_offset",
                "pixels",
                "action",
                "observation",
            )
            missing = [key for key in required if key not in source]
            if missing:
                raise KeyError(f"StableWM HDF5 lacks required columns: {missing}")
            row_count = int(source["pixels"].shape[0])
            row_lengths = np.asarray(source["ep_len"][:], dtype=np.int64)
            row_offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
            row_episode_ids = np.asarray(source["ep_idx"][:], dtype=np.int64)
            if not (row_lengths.shape == row_offsets.shape == row_episode_ids.shape == (row_count,)):
                raise ValueError("StableWM episode metadata must be per-row vectors matching pixels")
            episode_starts = np.flatnonzero(np.arange(row_count, dtype=np.int64) == row_offsets)
            if episode_starts.size == 0:
                raise ValueError("StableWM HDF5 contains no physical episode starts")
            self.episode_lengths = row_lengths[episode_starts]
            self.episode_offsets = row_offsets[episode_starts]
            self.episode_ids = row_episode_ids[episode_starts]
            episode_ends = self.episode_offsets + self.episode_lengths
            if (
                np.any(self.episode_lengths <= 0)
                or np.any(episode_ends > row_count)
                or not np.array_equal(self.episode_offsets, episode_starts)
                or not np.array_equal(
                    self.episode_ids,
                    np.arange(len(self.episode_ids), dtype=np.int64),
                )
            ):
                raise ValueError("StableWM physical episode metadata is inconsistent")
            action = np.asarray(source["action"][:], dtype=np.float32)
            observation = np.asarray(source["observation"][:], dtype=np.float32)

        self.action_dim = int(action.shape[-1])
        self.proprio_dim = int(observation.shape[-1])
        self.state_dim = self.proprio_dim
        if normalize_action:
            self.action_mean = torch.from_numpy(action.mean(axis=0))
            self.action_std = torch.from_numpy(action.std(axis=0)).clamp_min(1.0e-6)
            self.proprio_mean = torch.from_numpy(observation.mean(axis=0))
            self.proprio_std = torch.from_numpy(observation.std(axis=0)).clamp_min(1.0e-6)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
        self.state_mean = self.proprio_mean.clone()
        self.state_std = self.proprio_std.clone()

    def __len__(self):
        return len(self.episode_lengths)


class StableWMH5SlicedDataset(Dataset):
    """Load one fixed-stride clip without reading the rest of its episode."""

    def __init__(
        self,
        metadata: StableWMH5Metadata,
        episode_indices,
        *,
        num_frames: int,
        frameskip: int,
        action_skip: int,
        transform,
        seed: int,
    ):
        if num_frames < 1 or frameskip < 1 or action_skip < 1:
            raise ValueError("num_frames, frameskip, and action_skip must be positive")
        if frameskip % action_skip:
            raise ValueError("StableWM concat actions require frameskip divisible by action_skip")
        self.metadata = metadata
        self.transform = transform
        self.num_frames = int(num_frames)
        self.frameskip = int(frameskip)
        self.action_skip = int(action_skip)
        self.action_dim = metadata.action_dim * self.frameskip // self.action_skip
        self.proprio_dim = metadata.proprio_dim
        self.state_dim = metadata.state_dim
        self._source = None

        slices = []
        for episode_id in np.asarray(episode_indices, dtype=np.int64).tolist():
            length = int(metadata.episode_lengths[episode_id])
            stop = length - self.num_frames * self.frameskip + 1
            slices.extend((episode_id, start) for start in range(max(stop, 0)))
        generator = np.random.default_rng(int(seed))
        generator.shuffle(slices)
        self.slices = slices
        if not self.slices:
            raise ValueError("StableWM split contains no valid clips")

    def __len__(self):
        return len(self.slices)

    def _open(self):
        if self._source is None:
            self._source = h5py.File(
                self.metadata.source_h5_path,
                "r",
                swmr=True,
                rdcc_nbytes=128 * 1024 * 1024,
            )
        return self._source

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_source"] = None
        return state

    def close(self):
        if self._source is not None:
            self._source.close()
            self._source = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index):
        source = self._open()
        episode_id, local_start = self.slices[int(index)]
        row_start = int(self.metadata.episode_offsets[episode_id]) + int(local_start)
        visual_rows = row_start + np.arange(self.num_frames, dtype=np.int64) * self.frameskip
        action_rows = (
            row_start
            + np.arange(
                self.num_frames * self.frameskip // self.action_skip,
                dtype=np.int64,
            )
            * self.action_skip
        )

        pixels = np.asarray(source["pixels"][visual_rows], dtype=np.float32) / 255.0
        visual = torch.from_numpy(pixels).permute(0, 3, 1, 2)
        if self.transform is not None:
            visual = self.transform(visual)

        proprio = torch.from_numpy(np.asarray(source["observation"][visual_rows], dtype=np.float32))
        proprio = (proprio - self.metadata.proprio_mean) / self.metadata.proprio_std
        action = torch.from_numpy(np.asarray(source["action"][action_rows], dtype=np.float32))
        action = (action - self.metadata.action_mean) / self.metadata.action_std
        action = action.reshape(self.num_frames, -1)
        state = proprio.clone()
        reward = torch.zeros(self.num_frames, dtype=torch.float32)
        return {"visual": visual, "proprio": proprio}, action, state, reward


def load_stablewm_h5_train_val(
    source_h5,
    *,
    transform,
    normalize_action,
    split_ratio,
    num_hist,
    num_pred,
    num_frames_val,
    frameskip,
    action_skip,
    random_seed,
):
    """Return train/validation clips and the shared normalization metadata."""

    metadata = StableWMH5Metadata(source_h5, normalize_action=normalize_action)
    episode_count = len(metadata)
    order = torch.randperm(
        episode_count,
        generator=torch.Generator().manual_seed(int(random_seed)),
    ).numpy()
    train_count = int(float(split_ratio) * episode_count)
    train_episode_indices = order[:train_count]
    valid_episode_indices = order[train_count:]
    frame_count = int(num_hist) + int(num_pred)
    train = StableWMH5SlicedDataset(
        metadata,
        train_episode_indices,
        num_frames=frame_count,
        frameskip=frameskip,
        action_skip=action_skip,
        transform=transform,
        seed=random_seed,
    )
    valid = StableWMH5SlicedDataset(
        metadata,
        valid_episode_indices,
        num_frames=num_frames_val or frame_count,
        frameskip=frameskip,
        action_skip=action_skip,
        transform=transform,
        seed=random_seed,
    )
    train_physical_episode_ids = metadata.episode_ids[train_episode_indices]
    return (
        {"train": train, "valid": valid},
        {"train": metadata, "valid": metadata},
        train_physical_episode_ids,
    )
