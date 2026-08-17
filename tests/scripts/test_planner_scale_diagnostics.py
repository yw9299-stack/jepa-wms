import unittest

from app.vjepa_wm.planner_scale_diagnostics import (
    build_scale_sweep_diagnostic,
    normalize_scale_sweep_values,
    require_deterministic_scale_sweep_augmentation,
)


def metrics(loss, sign, ratio):
    return {
        "planner_landscape_loss": loss,
        "planner_valid_pairwise_sign_accuracy": sign,
        "planner_predicted_to_real_cost_std_ratio": ratio,
        "planner_valid_group_count": 10,
    }


class TestPlannerScaleDiagnostics(unittest.TestCase):
    def test_scale_sweep_requires_deterministic_augmentation(self):
        deterministic = {
            "auto_augment": False,
            "random_horizontal_flip": False,
            "motion_shift": False,
            "random_resize_aspect_ratio": [1.0, 1.0],
            "random_resize_scale": [1.0, 1.0],
            "reprob": 0.0,
        }
        require_deterministic_scale_sweep_augmentation(deterministic)
        for key, value in (
            ("random_horizontal_flip", True),
            ("random_resize_scale", [0.8, 1.0]),
            ("reprob", 0.1),
        ):
            invalid = dict(deterministic)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_deterministic_scale_sweep_augmentation(invalid)

    def test_grid_requires_unique_positive_identity_intervention(self):
        self.assertEqual(
            normalize_scale_sweep_values([0.5, 1, 2.0]),
            (0.5, 1.0, 2.0),
        )
        for invalid in ([0.5, 2.0], [0.0, 1.0], [1.0, 1.0], []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_scale_sweep_values(invalid)

    def test_summary_separates_learned_identity_and_best_fixed(self):
        result = build_scale_sweep_diagnostic(
            learned_scale=1.1,
            learned_metrics=metrics(1.16, 0.77, 0.73),
            fixed_metrics=[
                (0.5, metrics(0.90, 0.70, 0.30)),
                (1.0, metrics(1.10, 0.76, 0.68)),
                (2.0, metrics(2.00, 0.80, 1.40)),
            ],
        )

        self.assertEqual(result["learned"]["mode"], "learned")
        self.assertEqual(result["identity"]["scale"], 1.0)
        self.assertEqual(result["best_fixed"]["scale"], 0.5)
        self.assertEqual(result["best_observed"]["scale"], 0.5)
        self.assertAlmostEqual(result["learned_minus_identity_loss"], 0.06)
        self.assertAlmostEqual(result["learned_improvement_over_identity"], -0.06)
        self.assertAlmostEqual(result["best_fixed_minus_identity_loss"], -0.20)
        self.assertAlmostEqual(result["best_fixed_improvement_over_identity"], 0.20)
        self.assertTrue(result["observed_any_loss_below_zero_difference_baseline"])
        self.assertEqual(len(result["records"]), 4)

    def test_summary_rejects_nonfinite_metrics(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            build_scale_sweep_diagnostic(
                learned_scale=1.0,
                learned_metrics=metrics(float("nan"), 0.5, 0.1),
                fixed_metrics=[(1.0, metrics(1.0, 0.5, 0.1))],
            )


if __name__ == "__main__":
    unittest.main()
