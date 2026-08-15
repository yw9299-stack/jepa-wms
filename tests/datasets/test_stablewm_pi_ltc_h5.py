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
    StableWMH5Metadata,
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

    def _write_episode_level_source(
        self,
        name,
        *,
        action_dim,
        proprio_columns,
        episode_id_key="episode_idx",
    ):
        path = Path(self.directory.name) / name
        episode_count, episode_length = 4, 30
        rows = episode_count * episode_length
        with h5py.File(path, "w") as source:
            source.create_dataset(
                episode_id_key,
                data=np.repeat(np.arange(episode_count, dtype=np.int64), episode_length),
            )
            source.create_dataset(
                "ep_len",
                data=np.full(episode_count, episode_length, dtype=np.int64),
            )
            source.create_dataset(
                "ep_offset",
                data=np.arange(episode_count, dtype=np.int64) * episode_length,
            )
            source.create_dataset(
                "pixels",
                data=np.zeros((rows, 4, 4, 3), dtype=np.uint8),
            )
            source.create_dataset(
                "action",
                data=np.arange(rows * action_dim, dtype=np.float32).reshape(rows, action_dim),
            )
            for key, dim, value in proprio_columns:
                source.create_dataset(
                    key,
                    data=np.full((rows, dim), value, dtype=np.float32),
                )
        return path

    def _write_task_sidecar(self, name, *, protocol, action_dim, physical_ids):
        path = Path(self.directory.name) / name
        groups, branches, episode_length = 4, 4, 30
        with h5py.File(path, "w") as sidecar:
            sidecar.attrs["protocol"] = protocol
            sidecar.attrs["complete"] = True
            sidecar.attrs["branches_per_state"] = branches
            sidecar.attrs["frameskip"] = 5
            sidecar.create_dataset(
                "selected_context_rows",
                data=np.arange(groups, dtype=np.int64) * episode_length,
            )
            sidecar.create_dataset(
                "selected_episode_id",
                data=np.arange(groups, dtype=np.int64),
            )
            if physical_ids:
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
                data=np.ones((groups * branches, 5, action_dim), dtype=np.float32),
            )
            sidecar.create_dataset(
                "next_pixels",
                data=np.zeros((groups * branches, 4, 4, 3), dtype=np.uint8),
            )
        return path

    def test_pusht_episode_level_metadata_and_proprio_column(self):
        source_path = self._write_episode_level_source(
            "pusht.h5",
            action_dim=2,
            proprio_columns=(("proprio", 4, 2.0),),
        )
        sidecar_path = self._write_task_sidecar(
            "pusht_sidecar.h5",
            protocol="pusht_planner_counterfactual_v1",
            action_dim=2,
            physical_ids=True,
        )
        datasets, metadata, train_episode_ids = load_stablewm_h5_train_val(
            source_path,
            transform=None,
            normalize_action=True,
            split_ratio=0.5,
            num_hist=3,
            num_pred=1,
            num_frames_val=4,
            frameskip=5,
            action_skip=1,
            random_seed=7,
            proprio_keys=("proprio",),
            expected_action_dim=2,
            expected_proprio_dim=4,
            expected_row_count=120,
            expected_episode_count=4,
            expected_total_clips=44,
        )
        self.assertEqual(metadata["train"].metadata_layout, "per_episode")
        self.assertEqual(len(datasets["train"]), 22)
        planner = PlannerLandscapeH5Dataset(
            source_h5=source_path,
            sidecar_h5=sidecar_path,
            transform=None,
            action_mean=metadata["train"].action_mean,
            action_std=metadata["train"].action_std,
            proprio_mean=metadata["train"].proprio_mean,
            proprio_std=metadata["train"].proprio_std,
            source_metadata=metadata["train"],
            expected_protocol="pusht_planner_counterfactual_v1",
        )
        self.assertEqual(planner[0]["context_proprio"].shape, (1, 4))
        train_groups, valid_groups = split_planner_groups_by_physical_episode(
            planner, train_episode_ids
        )
        self.assertEqual((len(train_groups), len(valid_groups)), (2, 2))

    def test_cube_episode_level_metadata_and_merged_proprio(self):
        source_path = self._write_episode_level_source(
            "cube.h5",
            action_dim=5,
            episode_id_key="ep_idx",
            proprio_columns=(
                ("proprio_effector_pos", 3, 1.0),
                ("proprio_gripper_opening", 1, 2.0),
                ("proprio_joint_pos", 2, 3.0),
            ),
        )
        sidecar_path = self._write_task_sidecar(
            "cube_sidecar.h5",
            protocol="cube_planner_counterfactual_v1",
            action_dim=5,
            physical_ids=False,
        )
        proprio_keys = (
            "proprio_effector_pos",
            "proprio_gripper_opening",
            "proprio_joint_pos",
        )
        metadata = StableWMH5Metadata(
            source_path,
            normalize_action=False,
            proprio_keys=proprio_keys,
            expected_action_dim=5,
            expected_proprio_dim=6,
        )
        self.assertEqual(metadata.metadata_layout, "per_episode")
        with h5py.File(source_path, "r", swmr=True) as source:
            merged = metadata.load_proprio(source, np.asarray([0], dtype=np.int64))[0]
        np.testing.assert_array_equal(
            merged,
            np.asarray([1.0, 1.0, 1.0, 2.0, 3.0, 3.0], dtype=np.float32),
        )
        planner = PlannerLandscapeH5Dataset(
            source_h5=source_path,
            sidecar_h5=sidecar_path,
            transform=None,
            action_mean=np.zeros(5, dtype=np.float32),
            action_std=np.ones(5, dtype=np.float32),
            proprio_mean=np.zeros(6, dtype=np.float32),
            proprio_std=np.ones(6, dtype=np.float32),
            source_metadata=metadata,
            expected_protocol="cube_planner_counterfactual_v1",
        )
        self.assertEqual(planner[0]["action"].shape, (4, 1, 25))
        np.testing.assert_array_equal(
            planner[0]["context_proprio"][0].numpy(),
            merged,
        )

    def test_expected_dimensions_are_hard_stops(self):
        source_path = self._write_episode_level_source(
            "bad_dim.h5",
            action_dim=2,
            proprio_columns=(("proprio", 4, 0.0),),
        )
        with self.assertRaisesRegex(ValueError, "action_dim=2, expected 5"):
            StableWMH5Metadata(
                source_path,
                proprio_keys=("proprio",),
                expected_action_dim=5,
            )


if __name__ == "__main__":
    unittest.main()
