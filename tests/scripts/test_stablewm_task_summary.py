import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_stablewm_task_multiseed import EXACT_PROTOCOL, summarize


class TestStableWmTaskSummary(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for seed in (42, 43, 44):
            learned = [True] * 40 + [False] * 10
            identity = [True] * 35 + [False] * 15
            vanilla = [True] * 37 + [False] * 13
            for arm, owner, outcomes, checkpoint in (
                ("learned", "pi", learned, "pi-checkpoint"),
                ("identity", "pi", identity, "pi-checkpoint"),
                ("vanilla_stepmatched", "vanilla", vanilla, "vanilla-checkpoint"),
            ):
                path = self.root / "pusht" / f"seed{seed}" / arm / "arm_audit.json"
                path.parent.mkdir(parents=True)
                audit = {
                    "status": "complete",
                    "task": "pusht",
                    "protocol": "pusht_stablewm_exact_jepa_v1",
                    "eval_seed": seed,
                    "arm": arm,
                    "owner": owner,
                    "repository_commit": "jepa-commit",
                    "lewm_repository_commit": "lewm-commit",
                    "source_h5": {"sha256": "source-hash"},
                    "evaluation": dict(EXACT_PROTOCOL),
                    "checkpoint": {
                        "sha256": checkpoint,
                        "common_trainable_initialization_sha256": "a" * 64,
                    },
                    "candidate_rng_audit": {
                        "generator_state_sequence_sha256": f"rng-{seed}"
                    },
                    "results": {
                        "episode_successes": outcomes,
                        "paired_start_sha256": f"start-{seed}",
                        "environment_state_sha256": f"state-{seed}",
                    },
                }
                path.write_text(json.dumps(audit), encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def test_estimands_remain_separate(self):
        result = summarize(self.root, "pusht", draws=200, bootstrap_seed=7)
        same = result["same_checkpoint_estimand_A"]["pooled"]
        cross = result["cross_checkpoint_estimand_B"]["pooled"]
        self.assertAlmostEqual(same["difference_pp"], 10.0)
        self.assertAlmostEqual(cross["difference_pp"], 6.0)
        self.assertEqual(same["episodes"], 150)
        self.assertEqual(cross["episodes"], 150)

    def test_pairing_hash_mismatch_is_a_hard_stop(self):
        path = self.root / "pusht" / "seed43" / "identity" / "arm_audit.json"
        audit = json.loads(path.read_text())
        audit["results"]["environment_state_sha256"] = "wrong"
        path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "environment_state_sha256 differs"):
            summarize(self.root, "pusht", draws=20, bootstrap_seed=7)


if __name__ == "__main__":
    unittest.main()
