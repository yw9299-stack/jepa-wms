from pathlib import Path

import numpy as np

from scripts.summarize_pointmaze_medium_stablewm_protocol import (
    _mcnemar_exact,
    _paired_ci,
)


ROOT = Path(__file__).resolve().parents[2]


def test_runner_uses_medium_stablewm_protocol_not_native_umaze():
    runner = (ROOT / "scripts/eval_pointmaze_medium_stablewm_protocol.py").read_text()
    assert "sys.path.insert(0, str(REPOSITORY))" in runner
    assert 'config_name="pointmaze_topdown_medium"' in runner
    assert "ogbench/temporal_pointmaze_medium_topdown" in runner
    assert '"solver.n_steps=16"' in runner
    assert '"solver.num_samples=256"' in runner
    assert '"solver.topk=64"' in runner
    assert '"plan_config.horizon=6"' in runner
    assert '"plan_config.receding_horizon=6"' in runner
    assert '"plan_config.action_block=5"' in runner
    assert "maze-base" not in runner


def test_launcher_covers_pi_interventions_and_matched_vanilla():
    launcher = (ROOT / "scripts/pointmaze_jepa_wm_medium_protocol_eval_autodl.sh").read_text()
    for invocation in (
        "run_arm learned pi",
        "run_arm identity pi",
        "run_arm fixed06 pi",
        "run_arm fixed04 pi",
        "run_arm vanilla_5pass vanilla",
    ):
        assert invocation in launcher
    assert "training=false" in launcher
    assert "native_pointmaze" not in launcher
    assert 'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"' in launcher
    assert "evaluator import preflight failed" in launcher
    assert "eval_pointmaze_medium_stablewm_protocol.py --help" in launcher
    assert "elif ! run_arm identity pi" in launcher
    assert "|| FAILED=$((FAILED + 1))" not in launcher


def test_adapter_costs_visual_endpoint_and_preserves_proprio_conditioning():
    adapter = (ROOT / "app/vjepa_wm/modelcustom/stablewm_pointmaze_cost.py").read_text()
    assert 'TensorDict({"visual": pixels, "proprio": proprio}' in adapter
    assert "goal_latent = self.model.encode(goal_pixels)" in adapter
    assert ".square().flatten(1).sum(dim=1)" in adapter
    assert "action_candidates" in adapter
    assert "encoding_cache_audit" in adapter


def test_paired_statistics_are_exact_for_simple_cases():
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(4, 0) == 0.125
    low, high = _paired_ci(
        np.ones(8, dtype=np.int8),
        samples=1_000,
        seed=7,
    )
    assert low == 100.0
    assert high == 100.0
