from pathlib import Path


SCRIPT = Path("scripts/pusht_second_pass_relay_autodl.sh")


def test_relay_resumes_both_accepted_arms_to_pass_two():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "pusht_jepa_wm_pi_ltc_step111464_v4_seed3072" in text
    assert "pusht_jepa_wm_vanilla_step111464_v4_seed3072" in text
    assert 'PI_PASS1="$PI_DIR/jepa-e0.pth.tar"' in text
    assert 'VANILLA_PASS1="$VANILLA_DIR/jepa-e0.pth.tar"' in text
    assert 'PI_PASS2="$PI_DIR/jepa-e1.pth.tar"' in text
    assert 'VANILLA_PASS2="$VANILLA_DIR/jepa-e1.pth.tar"' in text
    assert "EXPECTED_COMPLETED_PASSES=2" in text
    assert '"optimizer_steps_per_arm": 27846' in text
    assert '"microbatches_per_arm": 222768' in text


def test_relay_preserves_original_resume_and_scheduler_state():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "assert_same_resume_state" in text
    assert "torch.equal(left, right)" in text
    assert 'meta.get("load_checkpoint") is True' in text
    assert 'meta.get("load_opt_scale_epoch") is True' in text
    assert 'optimization["total_optimizer_steps"] == 111464' in text
    assert '"scheduler_advanced_to_optimizer_step": 13923' in text
    assert '"model_optimizer_scaler_state_restored": True' in text


def test_relay_validates_before_launch_and_stops_at_atomic_boundary():
    text = SCRIPT.read_text(encoding="utf-8")

    resume_validation = text.index("assert_same_resume_state")
    launch = text.index('echo "[launch] owner=$owner')
    wait = text.index('wait_for_pass2_checkpoint "$pass2"')
    stop = text.index('stop_at_boundary "$pid"')
    assert resume_validation < launch < wait < stop
    assert '"training_complete": False' in text
    assert '"optimizer_step_in_epoch": 0' in text
    assert "pass-2 checkpoint exists but failed validation" in text
    assert "tmux" not in text.lower()


def test_relay_pins_training_commit_and_pass_one_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "90bb004351a1e4c42f0e99d20601fb69dcd22fdf" in text
    assert "3933f39514ae6175c8186682b46f5da2430f8c6d96beff0a55fdca70a35b5869" in text
    assert "2661e14b526a4e5098f5085d59ca664222edfdafafdb8ea15ca2c09dfb04f847" in text
    assert "PUSHT_MATCHED_ONE_PASS_COMPLETE" in text
    assert "PI and vanilla common initialization differs" in text
    assert "PUSHT_MATCHED_TWO_PASS_COMPLETE" in text
