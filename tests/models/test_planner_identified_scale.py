# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import unittest
from contextlib import nullcontext

import torch
from torch.func import functional_call, jvp

from app.plan_common.models.AdaLN_vit import VisionTransformerAdaLN
from app.plan_common.models.planner_identified_scale import PlannerIdentifiedInputScale
from app.plan_common.models.vit import ViTPredictor


def _math_sdpa():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


def _directional_jvp(model, args):
    parameters = dict(model.named_parameters())
    log_scale_name = "planner_input_scale.log_scale"
    log_scale = parameters[log_scale_name]

    def forward_with_log_scale(value):
        overridden = dict(parameters)
        overridden[log_scale_name] = value
        output = functional_call(model, overridden, args)
        if isinstance(output, tuple):
            output = output[0]
        return output

    # Flash/efficient SDPA does not implement forward-mode AD on every
    # backend (notably CPU). The math kernel is equivalent for this test and
    # supports the JVP used by the mechanism preflight.
    with _math_sdpa():
        output, derivative = jvp(
            forward_with_log_scale,
            (log_scale,),
            (torch.ones_like(log_scale),),
        )
    output = output.flatten(1)
    derivative = derivative.flatten(1)
    projection = (derivative * output).sum(dim=1, keepdim=True)
    projection = projection / output.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    directional = derivative - projection * output
    return directional.norm(dim=1).mean()


class TestPlannerIdentifiedInputScale(unittest.TestCase):
    def test_gradient_is_detached_by_default_and_enabled_explicitly(self):
        scale = PlannerIdentifiedInputScale()
        x = torch.randn(3, 5, requires_grad=True)

        scale(x).square().mean().backward()
        self.assertIsNone(scale.log_scale.grad)
        self.assertIsNotNone(x.grad)

        x.grad = None
        with scale.landscape_gradient():
            scale(x).square().mean().backward()
        self.assertIsNotNone(scale.log_scale.grad)
        self.assertGreater(abs(scale.log_scale.grad.item()), 0.0)

    def test_runtime_override_is_positive_and_reversible(self):
        scale = PlannerIdentifiedInputScale()
        x = torch.ones(2, 3)
        scale.set_runtime_override(0.4)
        self.assertTrue(torch.equal(scale(x), torch.full_like(x, 0.4)))
        scale.clear_runtime_override()
        self.assertTrue(torch.equal(scale(x), x))
        with self.assertRaises(ValueError):
            scale.set_runtime_override(0.0)


class TestDinoWmPlannerIdentifiedScale(unittest.TestCase):
    def _predictor(self, enabled):
        return ViTPredictor(
            num_patches=4,
            num_frames=2,
            dim=12,
            depth=1,
            heads=3,
            mlp_dim=24,
            dim_head=4,
            dropout=0.0,
            emb_dropout=0.0,
            use_sdpa=False,
            planner_identified_input_scale=enabled,
            planner_identified_visual_dim=8 if enabled else None,
        ).eval()

    def test_identity_is_native_and_action_features_are_not_scaled(self):
        torch.manual_seed(7)
        native = self._predictor(enabled=False)
        calibrated = self._predictor(enabled=True)
        calibrated.load_state_dict(native.state_dict(), strict=False)
        x = torch.randn(2, 8, 12)

        self.assertTrue(torch.equal(native(x), calibrated(x)))

        calibrated.planner_input_scale.set_runtime_override(0.4)
        scaled = calibrated.scale_visual_input(x)
        self.assertTrue(torch.equal(scaled[..., :8], x[..., :8] * 0.4))
        self.assertTrue(torch.equal(scaled[..., 8:], x[..., 8:]))

    def test_log_scale_changes_prediction_direction(self):
        torch.manual_seed(11)
        model = self._predictor(enabled=True)
        x = torch.randn(2, 8, 12)
        with model.planner_input_scale.landscape_gradient():
            directional_norm = _directional_jvp(model, (x,))
        self.assertTrue(torch.isfinite(directional_norm))
        self.assertGreater(directional_norm.item(), 1.0e-6)


class TestAdaLnPlannerIdentifiedScale(unittest.TestCase):
    def _predictor(self, enabled):
        return VisionTransformerAdaLN(
            img_size=(28, 28),
            patch_size=14,
            num_frames=2,
            tubelet_size=1,
            embed_dim=12,
            predictor_embed_dim=12,
            depth=1,
            num_heads=3,
            mlp_ratio=2.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            use_rope=False,
            action_dim=3,
            use_proprio=False,
            action_encoder_inpred=True,
            init_scale_factor_adaln=10,
            planner_identified_input_scale=enabled,
        ).eval()

    def test_identity_is_native(self):
        torch.manual_seed(13)
        native = self._predictor(enabled=False)
        calibrated = self._predictor(enabled=True)
        calibrated.load_state_dict(native.state_dict(), strict=False)
        x = torch.randn(2, 2, 1, 2, 2, 12)
        actions = torch.randn(2, 2, 3)

        native_output = native(x, actions)[0]
        calibrated_output = calibrated(x, actions)[0]
        self.assertTrue(torch.equal(native_output, calibrated_output))

    def test_log_scale_changes_prediction_direction(self):
        torch.manual_seed(17)
        model = self._predictor(enabled=True)
        x = torch.randn(2, 2, 1, 2, 2, 12)
        actions = torch.randn(2, 2, 3)
        with model.planner_input_scale.landscape_gradient():
            directional_norm = _directional_jvp(model, (x, actions))
        self.assertTrue(torch.isfinite(directional_norm))
        self.assertGreater(directional_norm.item(), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
