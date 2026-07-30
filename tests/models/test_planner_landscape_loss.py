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
    install_planner_scale_gradient,
    normalized_pairwise_landscape_error,
    pairwise_sign_accuracy,
)


class TestPlannerLandscapeLoss(unittest.TestCase):
    def test_matching_landscapes_have_zero_error_and_full_sign_accuracy(self):
        real = torch.tensor([[1.0, 3.0, 2.0], [5.0, 2.0, 7.0]])
        predicted = real.clone().requires_grad_(True)
        loss = normalized_pairwise_landscape_error(predicted, real)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(pairwise_sign_accuracy(predicted, real).item(), 1.0)
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))

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
