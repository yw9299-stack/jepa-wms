# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from app.plan_common.datasets.planner_landscape_h5 import (
    PlannerLandscapeH5Dataset,
    split_planner_groups_by_physical_episode,
)
from app.plan_common.datasets.stablewm_h5_dset import (
    load_stablewm_h5_train_val,
)


class TestStableWmPiLtcH5(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.source_path = root / "source.h5"
        self.sidecar_path = root / "sidecar.h5"
        episode_count, episode_length = 4, 30
        rows = episode_count * episode_length
        with h5py.File(self.source_path, "w") as source:
            source.create_dataset(
                "ep_idx",
                data=np.repeat(np.arange(episode_count, dtype=np.int64), episode_length),
            )
            source.create_dataset(
                "step_idx",
                data=np.tile(np.arange(episode_length, dtype=np.int64), episode_count),
            )
            source.create_dataset(
                "ep_len",
                data=np.full(rows, episode_length, dtype=np.int64),
            )
            source.create_dataset(
                "ep_offset",
                data=np.repeat(
                    np.arange(episode_count, dtype=np.int64) * episode_length,
                    episode_length,
                ),
            )
            pixels = np.arange(rows * 4 * 4 * 3, dtype=np.uint8).reshape(rows, 4, 4, 3)
            source.create_dataset("pixels", data=pixels)
            source.create_dataset(
                "goal_pixels",
                data=np.flip(pixels, axis=1),
            )
            source.create_dataset(
                "action",
                data=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
            )
            source.create_dataset(
                "observation",
                data=np.arange(rows * 4, dtype=np.float32).reshape(rows, 4),
            )

        groups, branches = 4, 4
        with h5py.File(self.sidecar_path, "w") as sidecar:
            sidecar.attrs["protocol"] = "pointmaze_planner_counterfactual_v1"
            sidecar.attrs["complete"] = True
            sidecar.attrs["branches_per_state"] = branches
            sidecar.attrs["frameskip"] = 5
            sidecar.create_dataset(
                "selected_context_rows",
                data=np.arange(groups, dtype=np.int64) * episode_length,
            )
            sidecar.create_dataset(
                "selected_episode_id",
                data=np.arange(groups, dtype=np.int64) * episode_length,
            )
            sidecar.create_dataset(
                "selected_physical_episode_id",
                data=np.arange(groups, dtype=np.int64),
            )
            sidecar.create_dataset(
                "group_id",
                data=np.repeat(np.arange(groups, dtype=np.int64), branches),
            )
            sidecar.create_dataset(
                "branch_slot",
                data=np.tile(np.arange(branches, dtype=np.int64), groups),
            )
            sidecar.create_dataset(
                "raw_action",
                data=np.ones((groups * branches, 5, 2), dtype=np.float32),
            )
            sidecar.create_dataset(
                "next_pixels",
                data=np.zeros((groups * branches, 4, 4, 3), dtype=np.uint8),
            )

    def tearDown(self):
        self.directory.cleanup()

    def test_exact_span_and_action_concatenation(self):
        datasets, metadata, train_episode_ids = load_stablewm_h5_train_val(
            self.source_path,
            transform=None,
            normalize_action=True,
            split_ratio=0.5,
            num_hist=3,
            num_pred=1,
            num_frames_val=4,
            frameskip=5,
            action_skip=1,
            random_seed=7,
        )
        self.assertEqual(len(datasets["train"]), 22)
        self.assertEqual(metadata["train"].action_dim, 2)
        self.assertEqual(len(metadata["train"]), 4)
        np.testing.assert_array_equal(
            metadata["train"].episode_offsets,
            np.asarray([0, 30, 60, 90], dtype=np.int64),
        )
        self.assertEqual(len(train_episode_ids), 2)
        obs, action, state, reward = datasets["train"][0]
        self.assertEqual(obs["visual"].shape, (4, 3, 4, 4))
        self.assertEqual(obs["proprio"].shape, (4, 4))
        self.assertEqual(action.shape, (4, 10))
        self.assertEqual(state.shape, (4, 4))
        self.assertEqual(reward.shape, (4,))

    def test_planner_group_shapes_and_episode_split(self):
        datasets, metadata, train_episode_ids = load_stablewm_h5_train_val(
            self.source_path,
            transform=None,
            normalize_action=True,
            split_ratio=0.5,
            num_hist=3,
            num_pred=1,
            num_frames_val=4,
            frameskip=5,
            action_skip=1,
            random_seed=7,
        )
        del datasets
        planner = PlannerLandscapeH5Dataset(
            source_h5=self.source_path,
            sidecar_h5=self.sidecar_path,
            transform=None,
            action_mean=metadata["train"].action_mean,
            action_std=metadata["train"].action_std,
            proprio_mean=metadata["train"].proprio_mean,
            proprio_std=metadata["train"].proprio_std,
            frameskip=5,
            goal_offset_steps=25,
        )
        sample = planner[0]
        self.assertEqual(sample["context_visual"].shape, (1, 3, 4, 4))
        self.assertEqual(sample["goal_visual"].shape, (1, 3, 4, 4))
        self.assertEqual(sample["next_visual"].shape, (4, 3, 4, 4))
        self.assertEqual(sample["action"].shape, (4, 1, 10))
        self.assertEqual(sample["context_proprio"].shape, (1, 4))
        train_groups, valid_groups = split_planner_groups_by_physical_episode(
            planner,
            train_episode_ids,
        )
        self.assertEqual(len(train_groups), 2)
        self.assertEqual(len(valid_groups), 2)
        self.assertEqual(len(train_groups) + len(valid_groups), len(planner))


if __name__ == "__main__":
    unittest.main()
