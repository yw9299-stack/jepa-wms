# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Same-state planner-landscape groups backed by the existing PI-LTC HDF5."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .stablewm_h5_dset import StableWMH5Metadata

POINTMAZE_PROTOCOL = "pointmaze_planner_counterfactual_v1"


class PlannerLandscapeH5Dataset(Dataset):
    def __init__(
        self,
        *,
        source_h5,
        sidecar_h5,
        transform,
        action_mean,
        action_std,
        proprio_mean,
        proprio_std,
        source_metadata: StableWMH5Metadata | None = None,
        proprio_keys=None,
        expected_protocol=POINTMAZE_PROTOCOL,
        expected_groups=None,
        expected_branches_per_group=None,
        expected_config_sha256=None,
        frameskip=5,
        goal_offset_steps=25,
    ):
        self.source_h5_path = Path(source_h5)
        self.sidecar_h5_path = Path(sidecar_h5)
        self.transform = transform
        self.frameskip = int(frameskip)
        self.goal_offset_steps = int(goal_offset_steps)
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)
        self.proprio_mean = torch.as_tensor(proprio_mean, dtype=torch.float32)
        self.proprio_std = torch.as_tensor(proprio_std, dtype=torch.float32)
        self.source_metadata = source_metadata or StableWMH5Metadata(
            self.source_h5_path,
            normalize_action=False,
            proprio_keys=proprio_keys,
        )
        if self.source_metadata.source_h5_path.resolve() != self.source_h5_path.resolve():
            raise ValueError("planner source metadata belongs to a different HDF5")
        self._source = None
        self._sidecar = None

        if not self.source_h5_path.is_file():
            raise FileNotFoundError(self.source_h5_path)
        if not self.sidecar_h5_path.is_file():
            raise FileNotFoundError(self.sidecar_h5_path)
        with h5py.File(self.sidecar_h5_path, "r", swmr=True) as sidecar:
            protocol = sidecar.attrs.get("protocol", "")
            if isinstance(protocol, bytes):
                protocol = protocol.decode("utf-8")
            if protocol != expected_protocol:
                raise ValueError(f"sidecar protocol={protocol!r}, expected {expected_protocol!r}")
            config_sha256 = sidecar.attrs.get("config_sha256", "")
            if isinstance(config_sha256, bytes):
                config_sha256 = config_sha256.decode("utf-8")
            if (
                expected_config_sha256 is not None
                and config_sha256 != expected_config_sha256
            ):
                raise ValueError("planner sidecar config SHA-256 does not match training config")
            if not bool(sidecar.attrs.get("complete", False)):
                raise ValueError(f"incomplete planner sidecar: {self.sidecar_h5_path}")
            self.branches_per_group = int(sidecar.attrs["branches_per_state"])
            if self.branches_per_group < 2:
                raise ValueError("planner landscape requires at least two branches per group")
            if (
                expected_branches_per_group is not None
                and self.branches_per_group != int(expected_branches_per_group)
            ):
                raise ValueError(
                    f"planner branches/group={self.branches_per_group}, "
                    f"expected {expected_branches_per_group}"
                )
            if int(sidecar.attrs["frameskip"]) != self.frameskip:
                raise ValueError("planner sidecar frameskip does not match training config")
            for key in (
                "selected_context_rows",
                "selected_episode_id",
                "group_id",
                "branch_slot",
                "raw_action",
                "next_pixels",
            ):
                if key not in sidecar:
                    raise KeyError(f"planner sidecar lacks {key!r}")
            self.context_rows = np.asarray(sidecar["selected_context_rows"][:], dtype=np.int64)
            self.episode_ids = np.asarray(sidecar["selected_episode_id"][:], dtype=np.int64)
            if "selected_physical_episode_id" in sidecar:
                self.physical_episode_ids = np.asarray(sidecar["selected_physical_episode_id"][:], dtype=np.int64)
            else:
                self.physical_episode_ids = self.episode_ids.copy()
            group_ids = np.asarray(sidecar["group_id"][:], dtype=np.int64)
            slots = np.asarray(sidecar["branch_slot"][:], dtype=np.int64)
            groups = len(self.context_rows)
            if expected_groups is not None and groups != int(expected_groups):
                raise ValueError(f"planner groups={groups}, expected {expected_groups}")
            branch_count = groups * self.branches_per_group
            if len(self.episode_ids) != groups:
                raise ValueError("planner sidecar episode IDs do not match selected groups")
            if not np.array_equal(
                group_ids,
                np.repeat(np.arange(groups, dtype=np.int64), self.branches_per_group),
            ):
                raise ValueError("planner branches are not contiguous by group")
            if not np.array_equal(
                slots,
                np.tile(np.arange(self.branches_per_group), groups),
            ):
                raise ValueError("planner branch slots are not canonical")
            if sidecar["raw_action"].shape != (
                branch_count,
                self.frameskip,
                self.source_metadata.action_dim,
            ):
                raise ValueError(
                    "planner raw_action shape does not match branches/frameskip/action_dim"
                )
            if sidecar["next_pixels"].shape[0] != branch_count:
                raise ValueError("planner next_pixels does not match branch count")
            sidecar_pixel_shape = tuple(sidecar["next_pixels"].shape[1:])

        with h5py.File(self.source_h5_path, "r", swmr=True) as source:
            for key in ("pixels", "action", *self.source_metadata.proprio_keys):
                if key not in source:
                    raise KeyError(f"source HDF5 lacks {key!r}")
            row_count = int(source["pixels"].shape[0])
            if sidecar_pixel_shape != tuple(source["pixels"].shape[1:]):
                raise ValueError("planner next_pixels shape does not match source pixels")
            if np.any(self.context_rows < 0) or np.any(self.context_rows >= row_count):
                raise ValueError("planner sidecar context row is outside source HDF5")
            source_physical_ids = self.source_metadata.physical_episode_ids_for_rows(
                self.context_rows
            )
            if not np.array_equal(source_physical_ids, self.physical_episode_ids):
                raise ValueError("planner sidecar physical episode IDs do not match source rows")
            source_metadata_ids = self.source_metadata.metadata_episode_ids_for_rows(
                self.context_rows
            )
            if not np.array_equal(source_metadata_ids, self.episode_ids):
                raise ValueError("planner sidecar metadata episode IDs do not match source rows")
            episode_end = self.source_metadata.episode_ends_for_rows(self.context_rows)
            goal_rows = self.context_rows + self.goal_offset_steps
            valid = goal_rows < episode_end
            self.group_indices = np.flatnonzero(valid).astype(np.int64)
            self.context_rows = self.context_rows[valid]
            self.episode_ids = self.episode_ids[valid]
            self.physical_episode_ids = self.physical_episode_ids[valid]
            self.goal_rows = goal_rows[valid]
            self.raw_action_dim = self.source_metadata.action_dim

    def __len__(self):
        return len(self.group_indices)

    def _open(self):
        if self._source is None:
            self._source = h5py.File(self.source_h5_path, "r", swmr=True, rdcc_nbytes=128 * 1024 * 1024)
        if self._sidecar is None:
            self._sidecar = h5py.File(self.sidecar_h5_path, "r", swmr=True, rdcc_nbytes=128 * 1024 * 1024)
        return self._source, self._sidecar

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_source"] = None
        state["_sidecar"] = None
        return state

    def close(self):
        for name in ("_source", "_sidecar"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index):
        source, sidecar = self._open()
        local_index = int(index)
        group_index = int(self.group_indices[local_index])
        context_row = int(self.context_rows[local_index])
        start = group_index * self.branches_per_group
        stop = start + self.branches_per_group

        context = np.asarray(source["pixels"][context_row])
        if "goal_pixels" in source:
            goal = np.asarray(source["goal_pixels"][context_row])
        else:
            goal = np.asarray(source["pixels"][int(self.goal_rows[local_index])])
        next_pixels = np.asarray(sidecar["next_pixels"][start:stop])
        pixels = np.concatenate((context[None], goal[None], next_pixels), axis=0)
        pixels = torch.from_numpy(pixels.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        if self.transform is not None:
            pixels = self.transform(pixels)

        action = torch.from_numpy(np.asarray(sidecar["raw_action"][start:stop], dtype=np.float32))
        action = (action - self.action_mean) / self.action_std
        action = action.reshape(self.branches_per_group, 1, -1)
        proprio = torch.from_numpy(
            self.source_metadata.load_proprio(
                source,
                np.asarray([context_row], dtype=np.int64),
            )[0]
        )
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        return {
            "context_visual": pixels[0:1],
            "goal_visual": pixels[1:2],
            "next_visual": pixels[2:],
            "action": action,
            "context_proprio": proprio.unsqueeze(0),
            "group_id": torch.tensor(group_index, dtype=torch.int64),
        }


def split_planner_groups_by_physical_episode(dataset, train_episode_ids):
    membership = np.isin(dataset.physical_episode_ids, np.asarray(train_episode_ids, dtype=np.int64))
    train = np.flatnonzero(membership).tolist()
    valid = np.flatnonzero(~membership).tolist()
    if not train or not valid:
        raise ValueError("planner physical-episode split produced an empty arm")
    return Subset(dataset, train), Subset(dataset, valid)
