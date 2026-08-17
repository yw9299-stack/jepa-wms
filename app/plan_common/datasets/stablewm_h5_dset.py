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
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:  # Register Blosc and other HDF5 filters before a compressed pixel read.
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # Uncompressed sources remain usable; reads fail clearly otherwise.
    _hdf5plugin = None


def _resolve_proprio_keys(source: h5py.File, requested: Sequence[str] | str | None) -> tuple[str, ...]:
    if requested is None:
        requested = ("observation",)
    elif isinstance(requested, str):
        requested = (requested,)

    resolved: list[str] = []
    for item in requested:
        item = str(item)
        if item.endswith("*"):
            matches = sorted(key for key in source.keys() if key.startswith(item[:-1]))
            if not matches:
                raise KeyError(f"StableWM HDF5 has no columns matching {item!r}")
            resolved.extend(matches)
        elif item in source:
            resolved.append(item)
        else:
            raise KeyError(f"StableWM HDF5 lacks proprio column {item!r}")
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError(f"invalid proprio column selection: {resolved}")
    return tuple(resolved)


def _load_columns(source: h5py.File, keys: Sequence[str], rows) -> np.ndarray:
    values = []
    for key in keys:
        value = np.asarray(source[key][rows], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        else:
            value = value.reshape(value.shape[0], -1)
        values.append(value)
    return np.concatenate(values, axis=-1)


def _streaming_stats(
    source: h5py.File,
    keys: Sequence[str],
    row_count: int,
    *,
    allowed_full_nan_rows: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute population statistics while validating StableWM placeholders.

    StableWorldModel records the reset observation and then left-aligns actions.
    Consequently, the final observation of an episode has no outgoing action and
    its complete action row is an intentional NaN placeholder.  No other
    non-finite values are valid.  Callers may pass the physical terminal rows to
    preserve that native convention without hiding malformed interior data.
    """

    allowed_full_nan_rows = (
        np.empty(0, dtype=np.int64)
        if allowed_full_nan_rows is None
        else np.asarray(allowed_full_nan_rows, dtype=np.int64).reshape(-1)
    )
    if allowed_full_nan_rows.size and (
        np.any(allowed_full_nan_rows < 0)
        or np.any(allowed_full_nan_rows >= int(row_count))
        or np.any(allowed_full_nan_rows[1:] <= allowed_full_nan_rows[:-1])
    ):
        raise ValueError("allowed full-NaN rows must be sorted unique in-range indices")

    total = None
    total_square = None
    finite_count = 0
    rows_seen = 0
    placeholder_count = 0
    for start in range(0, int(row_count), 65536):
        stop = min(int(row_count), start + 65536)
        value = _load_columns(source, keys, slice(start, stop)).astype(np.float64)
        finite_rows = np.isfinite(value).all(axis=1)
        if not finite_rows.all():
            invalid_local = np.flatnonzero(~finite_rows)
            invalid_global = invalid_local + start
            invalid_values = value[invalid_local]
            if not np.isnan(invalid_values).all():
                raise ValueError(
                    f"non-terminal or partial non-finite values for columns {tuple(keys)}"
                )
            allowed_positions = np.searchsorted(
                allowed_full_nan_rows,
                invalid_global,
            )
            allowed = allowed_positions < len(allowed_full_nan_rows)
            if allowed.any():
                matched = np.zeros_like(allowed)
                matched[allowed] = (
                    allowed_full_nan_rows[allowed_positions[allowed]]
                    == invalid_global[allowed]
                )
                allowed = matched
            if not allowed.all():
                raise ValueError(
                    f"full-NaN values outside physical episode terminals for columns {tuple(keys)}"
                )
            placeholder_count += len(invalid_global)

        finite_value = value[finite_rows]
        if len(finite_value):
            chunk_sum = finite_value.sum(axis=0)
            chunk_square = np.square(finite_value).sum(axis=0)
            total = chunk_sum if total is None else total + chunk_sum
            total_square = chunk_square if total_square is None else total_square + chunk_square
            finite_count += len(finite_value)
        rows_seen += len(value)
    if (
        rows_seen != int(row_count)
        or finite_count < 1
        or total is None
        or total_square is None
        or not np.isfinite(total).all()
        or not np.isfinite(total_square).all()
    ):
        raise ValueError(f"non-finite or incomplete statistics for columns {tuple(keys)}")
    mean = total / finite_count
    variance = np.maximum(total_square / finite_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1.0e-6)
    return (
        torch.from_numpy(mean.astype(np.float32)),
        torch.from_numpy(std.astype(np.float32)),
        placeholder_count,
    )


class StableWMH5Metadata:
    """Dataset metadata exposed through the ordinary JEPA-WM training API."""

    def __init__(
        self,
        source_h5: str | Path,
        normalize_action: bool = True,
        *,
        proprio_keys: Sequence[str] | str | None = None,
        expected_action_dim: int | None = None,
        expected_proprio_dim: int | None = None,
        expected_row_count: int | None = None,
        expected_episode_count: int | None = None,
    ):
        self.source_h5_path = Path(source_h5)
        if not self.source_h5_path.is_file():
            raise FileNotFoundError(self.source_h5_path)
        with h5py.File(self.source_h5_path, "r", swmr=True) as source:
            required = ("ep_len", "ep_offset", "pixels", "action")
            missing = [key for key in required if key not in source]
            if missing:
                raise KeyError(f"StableWM HDF5 lacks required columns: {missing}")
            row_count = int(source["pixels"].shape[0])
            if source["action"].shape[0] != row_count:
                raise ValueError("StableWM action and pixel row counts differ")
            raw_lengths = np.asarray(source["ep_len"][:], dtype=np.int64).reshape(-1)
            raw_offsets = np.asarray(source["ep_offset"][:], dtype=np.int64).reshape(-1)
            if raw_lengths.shape != raw_offsets.shape or not len(raw_lengths):
                raise ValueError("StableWM episode lengths and offsets must be matching vectors")

            if len(raw_lengths) == row_count:
                episode_starts = np.flatnonzero(np.arange(row_count, dtype=np.int64) == raw_offsets)
                if episode_starts.size == 0:
                    raise ValueError("StableWM HDF5 contains no physical episode starts")
                self.episode_lengths = raw_lengths[episode_starts]
                self.episode_offsets = raw_offsets[episode_starts]
                self.metadata_episode_ids = episode_starts.astype(np.int64)
                self.metadata_layout = "per_row"
            else:
                self.episode_lengths = raw_lengths
                self.episode_offsets = raw_offsets
                self.metadata_episode_ids = np.arange(len(raw_lengths), dtype=np.int64)
                self.metadata_layout = "per_episode"

            episode_ends = self.episode_offsets + self.episode_lengths
            if (
                np.any(self.episode_lengths <= 0)
                or np.any(episode_ends > row_count)
                or self.episode_offsets[0] != 0
                or episode_ends[-1] != row_count
                or not np.array_equal(self.episode_offsets[1:], episode_ends[:-1])
            ):
                raise ValueError("StableWM physical episode metadata is inconsistent")

            row_episode_key = next(
                (
                    key
                    for key in ("ep_idx", "episode_idx")
                    if key in source and source[key].shape == (row_count,)
                ),
                None,
            )
            if row_episode_key is None:
                self.episode_ids = np.arange(len(self.episode_lengths), dtype=np.int64)
            else:
                row_episode_ids = np.asarray(source[row_episode_key][:], dtype=np.int64)
                self.episode_ids = row_episode_ids[self.episode_offsets]
                expected_ids = np.repeat(self.episode_ids, self.episode_lengths)
                if not np.array_equal(row_episode_ids, expected_ids):
                    raise ValueError(f"StableWM {row_episode_key} is not constant within episodes")
            if len(np.unique(self.episode_ids)) != len(self.episode_ids):
                raise ValueError("StableWM physical episode IDs must be unique")

            self.proprio_keys = _resolve_proprio_keys(source, proprio_keys)
            for key in self.proprio_keys:
                if source[key].shape[0] != row_count:
                    raise ValueError(f"StableWM proprio column {key!r} has the wrong row count")
            self.action_dim = int(np.prod(source["action"].shape[1:]))
            self.proprio_dim = int(
                sum(np.prod(source[key].shape[1:]) if source[key].ndim > 1 else 1 for key in self.proprio_keys)
            )
            if expected_row_count is not None and row_count != int(expected_row_count):
                raise ValueError(f"StableWM rows={row_count}, expected {expected_row_count}")
            if expected_episode_count is not None and len(self.episode_lengths) != int(expected_episode_count):
                raise ValueError(
                    f"StableWM episodes={len(self.episode_lengths)}, expected {expected_episode_count}"
                )
            if expected_action_dim is not None and self.action_dim != int(expected_action_dim):
                raise ValueError(f"StableWM action_dim={self.action_dim}, expected {expected_action_dim}")
            if expected_proprio_dim is not None and self.proprio_dim != int(expected_proprio_dim):
                raise ValueError(f"StableWM proprio_dim={self.proprio_dim}, expected {expected_proprio_dim}")

            if normalize_action:
                terminal_rows = episode_ends - 1
                (
                    self.action_mean,
                    self.action_std,
                    self.action_placeholder_count,
                ) = _streaming_stats(
                    source,
                    ("action",),
                    row_count,
                    allowed_full_nan_rows=terminal_rows,
                )
                self.proprio_mean, self.proprio_std, _ = _streaming_stats(
                    source, self.proprio_keys, row_count
                )

        self.state_dim = self.proprio_dim
        self.row_count = row_count
        if not normalize_action:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.action_placeholder_count = 0
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
        self.state_mean = self.proprio_mean.clone()
        self.state_std = self.proprio_std.clone()

    def __len__(self):
        return len(self.episode_lengths)

    def episode_positions_for_rows(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        positions = np.searchsorted(self.episode_offsets, rows, side="right") - 1
        if np.any(positions < 0) or np.any(rows >= self.episode_offsets[positions] + self.episode_lengths[positions]):
            raise ValueError("row is outside StableWM physical episodes")
        return positions

    def physical_episode_ids_for_rows(self, rows) -> np.ndarray:
        return self.episode_ids[self.episode_positions_for_rows(rows)]

    def metadata_episode_ids_for_rows(self, rows) -> np.ndarray:
        return self.metadata_episode_ids[self.episode_positions_for_rows(rows)]

    def episode_ends_for_rows(self, rows) -> np.ndarray:
        positions = self.episode_positions_for_rows(rows)
        return self.episode_offsets[positions] + self.episode_lengths[positions]

    def load_proprio(self, source: h5py.File, rows) -> np.ndarray:
        return _load_columns(source, self.proprio_keys, rows)


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

        proprio = torch.from_numpy(self.metadata.load_proprio(source, visual_rows))
        proprio = (proprio - self.metadata.proprio_mean) / self.metadata.proprio_std
        action = torch.from_numpy(np.asarray(source["action"][action_rows], dtype=np.float32))
        action = (action - self.metadata.action_mean) / self.metadata.action_std
        # Match native LeWM: terminal action NaNs are normalized first, then
        # replaced by the neutral value in normalized action space.
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
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
    proprio_keys=None,
    expected_action_dim=None,
    expected_proprio_dim=None,
    expected_row_count=None,
    expected_episode_count=None,
    expected_total_clips=None,
):
    """Return train/validation clips and the shared normalization metadata."""

    metadata = StableWMH5Metadata(
        source_h5,
        normalize_action=normalize_action,
        proprio_keys=proprio_keys,
        expected_action_dim=expected_action_dim,
        expected_proprio_dim=expected_proprio_dim,
        expected_row_count=expected_row_count,
        expected_episode_count=expected_episode_count,
    )
    episode_count = len(metadata)
    order = torch.randperm(
        episode_count,
        generator=torch.Generator().manual_seed(int(random_seed)),
    ).numpy()
    train_count = int(float(split_ratio) * episode_count)
    train_episode_indices = order[:train_count]
    valid_episode_indices = order[train_count:]
    frame_count = int(num_hist) + int(num_pred)
    total_clips = int(np.maximum(metadata.episode_lengths - frame_count * int(frameskip) + 1, 0).sum())
    if expected_total_clips is not None and total_clips != int(expected_total_clips):
        raise ValueError(f"StableWM clips={total_clips}, expected {expected_total_clips}")
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
