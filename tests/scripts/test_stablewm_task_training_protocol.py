import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from scripts.generate_stablewm_task_training_configs import TASKS, derive_task_configs


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
                    pi["planner_identified"]["target_energy_eligibility"],
                    "strictly_greater_than_eps",
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

    def test_five_pass_effective_batch_schedule_is_unchanged(self):
        for task in TASKS:
            with self.subTest(task=task):
                pi, vanilla = derive_task_configs(self.base, task)
                for config in (pi, vanilla):
                    optimization = config["optimization"]["transition_model"]
                    self.assertEqual(optimization["num_epochs"], 5)
                    self.assertEqual(optimization["gradient_accumulation_steps"], 8)
                    self.assertEqual(config["data"]["loader"]["batch_size"], 16)


if __name__ == "__main__":
    unittest.main()
