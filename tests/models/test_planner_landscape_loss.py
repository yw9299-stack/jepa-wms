# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import torch

from app.plan_common.models.planner_identified_scale import (
    PlannerIdentifiedInputScale,
)
from app.vjepa_wm.planner_landscape import (
    clip_grad_norm_with_isolated_parameters,
    install_planner_scale_gradient,
    normalized_pairwise_landscape_error,
    pairwise_sign_accuracy,
    planner_landscape_result,
)
from app.vjepa_wm.utils import init_opt
from src.utils.schedulers import resolve_optimizer_schedule_steps


class TestPlannerLandscapeLoss(unittest.TestCase):
    def test_planner_forward_uses_full_history_and_only_final_endpoint_cost(self):
        class Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.planner_input_scale = PlannerIdentifiedInputScale()

        class WorldModel:
            def __init__(self):
                self.encoder = torch.nn.Identity()
                self.predictor = Predictor()
                self.action_encoder = torch.nn.Identity()
                self.proprio_encoder = torch.nn.Identity()

            def encode_obs(self, obs):
                return {"visual": obs["visual"].unsqueeze(-1)}

            def encode_act(self, action):
                return action

            def encode_proprio(self, proprio):
                return proprio

            def forward_pred(
                self,
                video_features,
                action_features,
                proprio_features,
                predictor_override=None,
            ):
                del proprio_features
                self.seen_shapes = (
                    tuple(video_features.shape),
                    tuple(action_features.shape),
                )
                scale = predictor_override.planner_input_scale
                early = scale(video_features[:, :-1] * 0.0) + 1000.0
                endpoint = action_features[:, -1:, 0]
                endpoint = scale(endpoint).reshape(-1, 1, 1, 1, 1, 1)
                return torch.cat((early, endpoint), dim=1), None, None

        world_model = WorldModel()
        batch = {
            "context_visual": torch.zeros(1, 3, 1, 1, 1),
            "goal_visual": torch.zeros(1, 1, 1, 1, 1),
            "next_visual": torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1, 1),
            "action": torch.tensor(
                [[[[0.0], [0.0], [1.0]], [[0.0], [0.0], [2.0]], [[0.0], [0.0], [3.0]]]]
            ),
            "context_proprio": torch.zeros(1, 3, 1),
        }
        result = planner_landscape_result(
            world_model,
            batch,
            device=torch.device("cpu"),
            dtype=torch.float32,
            mixed_precision=False,
        )
        self.assertEqual(result.loss.item(), 0.0)
        self.assertEqual(result.stats["planner_context_frames"].item(), 3.0)
        self.assertEqual(world_model.seen_shapes[0][1], 3)
        self.assertEqual(world_model.seen_shapes[1][1], 3)

    def test_planner_scale_has_its_own_lr_and_no_weight_decay(self):
        predictor = torch.nn.Module()
        predictor.projection = torch.nn.Linear(2, 2)
        predictor.planner_input_scale = PlannerIdentifiedInputScale()
        optimizer, _, scheduler, wd_scheduler = init_opt(
            predictor=predictor,
            action_encoder=None,
            proprio_encoder=None,
            encoder=None,
            iterations_per_epoch=10,
            start_lr=1.0e-3,
            ref_lr=1.0e-3,
            final_lr=1.0e-3,
            warmup=0,
            num_epochs=1,
            ipe_scale=1.0,
            weight_decay=1.0e-2,
            final_weight_decay=1.0e-2,
            planner_scale_lr_multiplier=0.1,
        )
        scale_groups = [
            group
            for group in optimizer.param_groups
            if group.get("group_name") == "planner_scale"
        ]
        self.assertEqual(len(scale_groups), 1)
        base_lr = scheduler.step()
        wd_scheduler.step()
        self.assertAlmostEqual(scale_groups[0]["lr"], base_lr * 0.1)
        self.assertEqual(scale_groups[0]["weight_decay"], 0.0)
        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

    def test_exact_optimizer_budget_owns_scheduler_horizon(self):
        self.assertEqual(
            resolve_optimizer_schedule_steps(9, 10, 83),
            83,
        )
        self.assertEqual(
            resolve_optimizer_schedule_steps(9, 10),
            90,
        )

    def test_matching_landscapes_have_zero_error_and_full_sign_accuracy(self):
        real = torch.tensor([[1.0, 3.0, 2.0], [5.0, 2.0, 7.0]])
        predicted = real.clone().requires_grad_(True)
        loss = normalized_pairwise_landscape_error(predicted, real)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(pairwise_sign_accuracy(predicted, real).item(), 1.0)
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))

    def test_identifiable_groups_preserve_the_previous_objective_exactly(self):
        real = torch.tensor([[1.0, 3.0, 2.0], [5.0, 2.0, 7.0]])
        predicted = torch.tensor(
            [[1.5, 2.0, 4.0], [4.0, 3.0, 8.0]],
            requires_grad=True,
        )
        predicted_difference = torch.stack(
            (
                predicted[:, 0] - predicted[:, 1],
                predicted[:, 0] - predicted[:, 2],
                predicted[:, 1] - predicted[:, 2],
            ),
            dim=1,
        )
        real_difference = torch.stack(
            (
                real[:, 0] - real[:, 1],
                real[:, 0] - real[:, 2],
                real[:, 1] - real[:, 2],
            ),
            dim=1,
        )
        previous = (
            (predicted_difference - real_difference).square().mean(dim=1)
            / real_difference.square().mean(dim=1).clamp_min(1.0e-8)
        ).mean()
        current, diagnostics = normalized_pairwise_landscape_error(
            predicted,
            real,
            return_diagnostics=True,
        )
        torch.testing.assert_close(current, previous, rtol=0.0, atol=0.0)
        self.assertEqual(diagnostics["valid_group_count"].item(), 2)
        self.assertEqual(diagnostics["valid_group_fraction"].item(), 1.0)

    def test_unidentifiable_group_has_no_loss_or_gradient_ownership(self):
        real = torch.tensor(
            [[2.0, 2.0, 2.0], [1.0, 2.0, 4.0]],
        )
        predicted = torch.tensor(
            [[-100.0, 0.0, 100.0], [1.5, 2.5, 3.5]],
            requires_grad=True,
        )
        loss, diagnostics = normalized_pairwise_landscape_error(
            predicted,
            real,
            return_diagnostics=True,
        )
        expected = normalized_pairwise_landscape_error(
            predicted[1:],
            real[1:],
        )
        torch.testing.assert_close(loss, expected, rtol=0.0, atol=0.0)
        loss.backward()
        torch.testing.assert_close(predicted.grad[0], torch.zeros(3))
        self.assertEqual(diagnostics["valid_group_count"].item(), 1)
        self.assertEqual(diagnostics["valid_group_fraction"].item(), 0.5)

    def test_relative_floor_rejects_group_far_below_batch_median_energy(self):
        real = torch.tensor(
            [
                [0.0, 1.0e-4, 2.0e-4],
                [0.0, 1.0, 2.0],
                [0.0, 2.0, 4.0],
            ]
        )
        predicted = torch.tensor(
            [
                [-100.0, 0.0, 100.0],
                [0.0, 1.5, 2.5],
                [0.0, 2.5, 3.5],
            ],
            requires_grad=True,
        )
        loss, diagnostics = normalized_pairwise_landscape_error(
            predicted,
            real,
            relative_energy_floor=1.0e-4,
            return_diagnostics=True,
        )
        expected = normalized_pairwise_landscape_error(
            predicted[1:],
            real[1:],
        )

        torch.testing.assert_close(loss, expected, rtol=0.0, atol=0.0)
        loss.backward()
        torch.testing.assert_close(predicted.grad[0], torch.zeros(3))
        self.assertEqual(diagnostics["valid_group_count"].item(), 2)
        self.assertAlmostEqual(
            diagnostics["target_energy_effective_floor"].item(),
            diagnostics["target_energy_median"].item() * 1.0e-4,
        )

    def test_relative_floor_must_be_finite_and_non_negative(self):
        predicted = torch.zeros(1, 2)
        real = torch.ones(1, 2)
        for invalid in (-1.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    normalized_pairwise_landscape_error(
                        predicted,
                        real,
                        relative_energy_floor=invalid,
                    )

    def test_all_unidentifiable_groups_produce_safe_zero_gradient(self):
        real = torch.ones(2, 4)
        predicted = torch.randn(2, 4, requires_grad=True)
        loss, diagnostics = normalized_pairwise_landscape_error(
            predicted,
            real,
            return_diagnostics=True,
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))
        self.assertEqual(diagnostics["valid_group_count"].item(), 0)
        self.assertEqual(diagnostics["valid_group_fraction"].item(), 0.0)

    def test_gradient_clipping_isolates_planner_scale_from_transition_norm(self):
        transition = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        scale = torch.nn.Parameter(torch.tensor(0.0))
        transition.grad = torch.tensor([3.0, 4.0])
        scale.grad = torch.tensor(1000.0)

        transition_norm, scale_norm = clip_grad_norm_with_isolated_parameters(
            (transition, scale),
            (scale,),
            max_norm=1.0,
        )

        self.assertAlmostEqual(float(transition_norm), 5.0, places=5)
        self.assertAlmostEqual(float(scale_norm), 1000.0, places=3)
        self.assertAlmostEqual(float(transition.grad.norm()), 1.0, places=5)
        self.assertAlmostEqual(float(scale.grad.abs()), 1.0, places=5)
        expected_transition = torch.tensor([0.6, 0.8])
        torch.testing.assert_close(transition.grad, expected_transition)

    def test_scale_gradient_installer_rejects_ordinary_gradient(self):
        predictor = torch.nn.Module()
        predictor.planner_input_scale = PlannerIdentifiedInputScale()
        world_model = SimpleNamespace(
            predictor=predictor,
            mixed_precision=False,
            scaler=None,
        )
        install_planner_scale_gradient(world_model, torch.tensor(2.5))
        self.assertEqual(
            predictor.planner_input_scale.log_scale.grad.item(),
            2.5,
        )
        predictor.planner_input_scale.log_scale.grad = torch.tensor(1.0)
        with self.assertRaises(RuntimeError):
            install_planner_scale_gradient(world_model, torch.tensor(2.5))


if __name__ == "__main__":
    unittest.main()
