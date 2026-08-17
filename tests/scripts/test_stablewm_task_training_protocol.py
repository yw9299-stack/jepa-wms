import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.generate_stablewm_task_training_configs import TASKS, derive_task_configs
from src.utils.schedulers import WarmupCosineSchedule, resolve_optimizer_schedule_steps


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
                self.assertEqual(
                    pi["planner_identified"]["history_size"],
                    custom["num_hist"],
                )
                self.assertEqual(
                    pi["planner_identified"]["scale_lr_multiplier"],
                    0.1,
                )
                self.assertEqual(
                    pi["planner_identified"]["groups_per_batch"],
                    4,
                )
                self.assertEqual(
                    pi["planner_identified"][
                        "validation_groups_per_batch"
                    ],
                    4,
                )
                self.assertTrue(
                    pi["planner_identified"]["validation_at_start"]
                )
                self.assertEqual(
                    pi["planner_identified"]["canary_scale_sweep_values"],
                    [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
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
                self.assertIn("_v6_", spec["pi_run"])
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

    def test_cosine_scheduler_preserves_scale_lr_multiplier(self):
        optimizer = SimpleNamespace(
            param_groups=[{}, {"lr_scale": 0.1, "group_name": "planner_scale"}]
        )
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=0,
            start_lr=5.0e-4,
            ref_lr=5.0e-4,
            final_lr=5.0e-4,
            T_max=10,
        )
        base_lr = scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], base_lr)
        self.assertEqual(optimizer.param_groups[1]["lr"], base_lr * 0.1)

    def test_launch_and_evaluation_names_are_step_matched(self):
        launcher = Path("scripts/stablewm_task_5pass_autodl.sh").read_text()
        trainer = Path("app/vjepa_wm/train.py").read_text(encoding="utf-8")
        evaluator = Path("scripts/stablewm_task_eval_autodl.sh").read_text()
        evaluator_protocol = Path(
            "scripts/eval_stablewm_task_protocol.py"
        ).read_text()
        summary = Path("scripts/summarize_stablewm_task_multiseed.py").read_text()
        self.assertIn("OPTIMIZER_STEP_BUDGET=111464", launcher)
        self.assertIn("OPTIMIZER_STEP_BUDGET=51184", launcher)
        self.assertIn("LEWM_REFERENCE_PASSES=8", launcher)
        self.assertIn("LEWM_REFERENCE_PASSES=4", launcher)
        self.assertIn("DEFAULT_TARGET_COMPLETE_PASSES=1", launcher)
        self.assertIn("DEFAULT_TARGET_COMPLETE_PASSES=4", launcher)
        self.assertIn("PI_LTC_STOP_AFTER_COMPLETE_PASSES", launcher)
        self.assertIn("not_scanned_user_confirmed", launcher)
        self.assertIn("[canary diagnostic]", launcher)
        self.assertIn("canary diagnostic checkpoint audit is missing", launcher)
        self.assertIn("checkpoint_role=\"canary_diagnostic\"", trainer)
        self.assertIn("canary_scale_sweep.json", trainer)
        self.assertIn("evaluate_canary_scale_sweep", trainer)
        self.assertIn("vanilla_stepmatched", launcher)
        self.assertIn("vanilla_stepmatched", evaluator)
        self.assertIn("planner_validation_history.json", evaluator)
        self.assertIn("PI_CHECKPOINT_NAME", evaluator)
        self.assertIn("AUDITED_HASHES", evaluator)
        self.assertIn("heldout_complete_pass_v1", evaluator_protocol)
        self.assertIn('"vanilla_stepmatched": "vanilla"', summary)
        for spec in TASKS.values():
            self.assertIn(spec["pi_run"], launcher)
            self.assertIn(spec["vanilla_run"], launcher)
            self.assertIn(spec["pi_run"], evaluator)
            self.assertIn(spec["vanilla_run"], evaluator)


if __name__ == "__main__":
    unittest.main()
