import json
import tempfile
import unittest
from pathlib import Path

from scripts.stablewm_checkpoint_selection import expected_checkpoint_schedule
from scripts.summarize_stablewm_task_multiseed import EXACT_PROTOCOL
from scripts.summarize_stablewm_task_seed_pair import summarize


class TestCheckpointSelection(unittest.TestCase):
    def test_native_pass_one_is_distinct_from_final_horizon(self):
        selected = expected_checkpoint_schedule(
            optimizer_steps_per_pass=13923,
            configured_optimizer_step_budget=111464,
            expected_completed_passes=1,
        )
        self.assertEqual(
            selected,
            {
                "epoch": 1,
                "total_optimizer_steps": 13923,
                "optimizer_step_in_epoch": 0,
                "training_complete": False,
                "selection_policy": "native_complete_pass_boundary",
            },
        )
        final = expected_checkpoint_schedule(
            optimizer_steps_per_pass=13923,
            configured_optimizer_step_budget=111464,
            expected_completed_passes=None,
        )
        self.assertEqual(final["epoch"], 8)
        self.assertEqual(final["optimizer_step_in_epoch"], 80)
        self.assertTrue(final["training_complete"])

    def test_complete_final_pass_is_a_valid_selected_boundary(self):
        selected = expected_checkpoint_schedule(
            optimizer_steps_per_pass=12796,
            configured_optimizer_step_budget=51184,
            expected_completed_passes=4,
        )
        self.assertEqual(selected["epoch"], 4)
        self.assertEqual(selected["total_optimizer_steps"], 51184)
        self.assertEqual(selected["optimizer_step_in_epoch"], 0)
        self.assertTrue(selected["training_complete"])
        self.assertEqual(
            selected["selection_policy"],
            "configured_final_horizon",
        )

    def test_pass_boundary_cannot_exceed_final_horizon(self):
        with self.assertRaisesRegex(ValueError, "exceed"):
            expected_checkpoint_schedule(
                optimizer_steps_per_pass=10,
                configured_optimizer_step_budget=20,
                expected_completed_passes=3,
            )


class TestSeed42PairSummary(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        common = {
            "status": "complete",
            "task": "pusht",
            "eval_seed": 42,
            "protocol": "pusht_stablewm_exact_jepa_v1",
            "repository_commit": "evaluator-commit",
            "training_repository_commit": "training-commit",
            "lewm_repository_commit": "lewm-commit",
            "source_h5": {"sha256": "source-marker"},
            "evaluation": dict(EXACT_PROTOCOL),
            "candidate_rng_audit": {"generator_state_sequence_sha256": "candidate-rng"},
        }
        for arm, owner, outcomes, checkpoint in (
            ("learned", "pi", [True] * 40 + [False] * 10, "pi-checkpoint"),
            (
                "vanilla_stepmatched",
                "vanilla",
                [True] * 35 + [False] * 15,
                "vanilla-checkpoint",
            ),
        ):
            audit = {
                **common,
                "arm": arm,
                "owner": owner,
                "checkpoint": {
                    "path": f"/{arm}.pth.tar",
                    "sha256": checkpoint,
                    "selection_policy": "native_complete_pass_boundary",
                    "selected_completed_passes": 1,
                    "optimizer_steps_per_epoch": 13923,
                    "total_optimizer_steps": 13923,
                    "optimizer_step_in_epoch": 0,
                    "common_trainable_initialization_sha256": "a" * 64,
                },
                "results": {
                    "episode_successes": outcomes,
                    "paired_start_sha256": "paired-starts",
                    "environment_state_sha256": "paired-states",
                    "seeds": list(range(50)),
                },
            }
            path = self.root / "pusht" / "seed42" / arm / "arm_audit.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(audit), encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def test_seed42_clean_pair_is_summarized_without_identity_arm(self):
        result = summarize(
            self.root,
            "pusht",
            42,
            expected_completed_passes=1,
            bootstrap_draws=200,
            bootstrap_seed=7,
        )
        self.assertEqual(result["pi_success_rate"], 80.0)
        self.assertEqual(result["vanilla_success_rate"], 70.0)
        self.assertAlmostEqual(result["difference_pp"], 10.0)
        self.assertEqual(result["completed_training_passes_per_arm"], 1)
        self.assertNotIn("identity", json.dumps(result))

    def test_pairing_mismatch_is_a_hard_stop(self):
        path = self.root / "pusht" / "seed42" / "vanilla_stepmatched" / "arm_audit.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit["results"]["paired_start_sha256"] = "wrong"
        path.write_text(json.dumps(audit), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "paired paired_start_sha256"):
            summarize(
                self.root,
                "pusht",
                42,
                expected_completed_passes=1,
                bootstrap_draws=20,
                bootstrap_seed=7,
            )

    def test_pass_two_pair_uses_existing_pass_two_audits(self):
        for path in self.root.rglob("arm_audit.json"):
            audit = json.loads(path.read_text(encoding="utf-8"))
            audit["checkpoint"]["selected_completed_passes"] = 2
            audit["checkpoint"]["total_optimizer_steps"] = 27846
            path.write_text(json.dumps(audit), encoding="utf-8")

        result = summarize(
            self.root,
            "pusht",
            42,
            expected_completed_passes=2,
            bootstrap_draws=200,
            bootstrap_seed=7,
        )

        self.assertEqual(result["completed_training_passes_per_arm"], 2)
        self.assertEqual(result["optimizer_steps_per_pass"], 13923)
        self.assertEqual(result["optimizer_steps_per_arm"], 27846)
        self.assertIn("matched 2-pass", result["estimand"])


def test_launcher_runs_only_two_clean_seed42_arms():
    text = Path("scripts/pusht_one_pass_seed42_clean_eval_autodl.sh").read_text(encoding="utf-8")
    summary_command = text.split("summarize_stablewm_task_seed_pair.py", 1)[1]
    assert "--eval-seed 42" in text
    assert "--episodes 50" in text
    assert "--expected-completed-passes 1" in text
    assert "--expected-completed-passes 1" in summary_command
    assert "run_arm pi learned" in text
    assert "run_arm vanilla vanilla_stepmatched" in text
    assert "run_arm pi identity" not in text
    assert "fixed06" not in text
    assert "fixed04" not in text
    assert "jepa-e0.pth.tar" in text


def test_two_pass_launcher_selects_atomic_pass_two_pair():
    text = Path("scripts/pusht_two_pass_seed42_clean_eval_autodl.sh").read_text(encoding="utf-8")
    summary_command = text.split("summarize_stablewm_task_seed_pair.py", 1)[1]
    assert "pusht_second_pass_relay/relay_summary.json" in text
    assert "pusht_twopass_seed42_clean_v1" in text
    assert "--eval-seed 42" in text
    assert "--episodes 50" in text
    assert "--expected-completed-passes 2" in text
    assert "--expected-completed-passes 2" in summary_command
    assert "run_arm pi learned" in text
    assert "run_arm vanilla vanilla_stepmatched" in text
    assert "run_arm pi identity" not in text
    assert "fixed06" not in text
    assert "fixed04" not in text
    assert "jepa-e1.pth.tar" in text


def test_preflight_accepts_only_audited_pass_boundaries():
    text = Path("scripts/preflight_pusht_one_pass_seed42_eval.py").read_text(encoding="utf-8")
    assert 'choices=(1, 2)' in text
    assert '1: "PUSHT_MATCHED_ONE_PASS_COMPLETE"' in text
    assert '2: "PUSHT_MATCHED_TWO_PASS_COMPLETE"' in text
    assert 'expected_completed_passes=expected_completed_passes' in text


def test_evaluator_separates_training_and_evaluation_commits():
    text = Path("scripts/eval_stablewm_task_protocol.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--expected-training-commit")' in text
    assert 'parser.add_argument("--expected-completed-passes", type=int)' in text
    assert '"training_repository_commit": expected_training_commit' in text
