import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from scripts.generate_stablewm_task_training_configs import TASKS, derive_task_configs
from src.utils.schedulers import resolve_optimizer_schedule_steps


class TestStableWmTaskTrainingProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = yaml.safe_load(
            Path("configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml").read_text()
        )

    def test_task_dimensions_and_source_provenance_are_pinned(self):
        for task, spec in TASKS.items():
            with self.subTest(task=task):
                pi, _ = derive_task_configs(self.base, task)
                custom = pi["data"]["custom"]
                self.assertEqual(custom["proprio_keys"], spec["proprio_keys"])
                self.assertEqual(custom["expected_action_dim"], spec["action_dim"])
                self.assertEqual(custom["expected_proprio_dim"], spec["proprio_dim"])
                self.assertEqual(custom["expected_total_clips"], spec["total_clips"])
                self.assertEqual(
                    pi["planner_identified"]["expected_config_sha256"],
                    spec["sidecar_config_sha256"],
                )
                self.assertEqual(
                    pi["planner_identified"]["target_energy_eps"],
                    1.0e-8,
                )
                self.assertEqual(
                    pi["planner_identified"]["target_energy_relative_floor"],
                    1.0e-4,
                )
                self.assertEqual(
                    pi["planner_identified"]["target_energy_eligibility"],
                    "strictly_greater_than_max_eps_or_1e-4_batch_median",
                )
                self.assertEqual(
                    pi["planner_identified"]["gradient_clip_ownership"],
                    "separate_transition_and_scale",
                )

    def test_vanilla_restores_exact_pi_config(self):
        for task in TASKS:
            with self.subTest(task=task):
                pi, vanilla = derive_task_configs(self.base, task)
                self.assertFalse(
                    vanilla["model"]["predictor"]["planner_identified_input_scale"]
                )
                self.assertEqual(vanilla["planner_identified"], {"enabled": False})
                restored = deepcopy(vanilla)
                restored["folder"] = pi["folder"]
                restored["model"]["predictor"]["planner_identified_input_scale"] = True
                restored["planner_identified"] = deepcopy(pi["planner_identified"])
                self.assertEqual(restored, pi)

    def test_task_specific_lewm_step_budgets_use_effective_batch_128(self):
        expected = {
            "pusht": {
                "steps": 111464,
                "steps_per_pass": 13923,
                "driver_epochs": 9,
                "passes": 8,
            },
            "cube": {
                "steps": 51184,
                "steps_per_pass": 12796,
                "driver_epochs": 4,
                "passes": 4,
            },
        }
        for task, spec in TASKS.items():
            with self.subTest(task=task):
                pi, vanilla = derive_task_configs(self.base, task)
                for config in (pi, vanilla):
                    optimization = config["optimization"]["transition_model"]
                    self.assertEqual(
                        optimization["num_epochs"],
                        expected[task]["driver_epochs"],
                    )
                    self.assertEqual(
                        optimization["total_optimizer_steps"],
                        expected[task]["steps"],
                    )
                    self.assertEqual(
                        optimization["expected_optimizer_steps_per_epoch"],
                        expected[task]["steps_per_pass"],
                    )
                    self.assertEqual(optimization["gradient_accumulation_steps"], 8)
                    self.assertEqual(config["data"]["loader"]["batch_size"], 16)
                self.assertEqual(
                    spec["lewm_reference_passes"],
                    expected[task]["passes"],
                )
                self.assertIn(
                    f"step{expected[task]['steps']}",
                    spec["pi_run"],
                )
                full_passes, tail_steps = divmod(
                    expected[task]["steps"],
                    expected[task]["steps_per_pass"],
                )
                self.assertEqual(full_passes, expected[task]["passes"])
                self.assertEqual(tail_steps, 80 if task == "pusht" else 0)

    def test_optimizer_schedule_uses_exact_step_budget(self):
        self.assertEqual(resolve_optimizer_schedule_steps(9, 13923, 111464), 111464)
        self.assertEqual(resolve_optimizer_schedule_steps(4, 12796, 51184), 51184)
        self.assertEqual(resolve_optimizer_schedule_steps(5, 10), 50)

    def test_launch_and_evaluation_names_are_step_matched(self):
        launcher = Path("scripts/stablewm_task_5pass_autodl.sh").read_text()
        evaluator = Path("scripts/stablewm_task_eval_autodl.sh").read_text()
        summary = Path("scripts/summarize_stablewm_task_multiseed.py").read_text()
        self.assertIn("OPTIMIZER_STEP_BUDGET=111464", launcher)
        self.assertIn("OPTIMIZER_STEP_BUDGET=51184", launcher)
        self.assertIn("LEWM_REFERENCE_PASSES=8", launcher)
        self.assertIn("LEWM_REFERENCE_PASSES=4", launcher)
        self.assertIn("vanilla_stepmatched", launcher)
        self.assertIn("vanilla_stepmatched", evaluator)
        self.assertIn('"vanilla_stepmatched": "vanilla"', summary)


if __name__ == "__main__":
    unittest.main()
