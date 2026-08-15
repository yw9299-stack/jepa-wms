from pathlib import Path


SCRIPT = Path("scripts/pusht_one_pass_relay_autodl.sh")


def test_relay_is_one_pass_and_variant_specific():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "pusht_jepa_wm_pi_ltc_step111464_v4_seed3072" in text
    assert "pusht_jepa_wm_vanilla_step111464_v4_seed3072" in text
    assert "EXPECTED_OPTIMIZER_STEPS_PER_PASS=13923" in text
    assert "EXPECTED_MICROBATCHES_PER_PASS=111384" in text
    assert 'PI_PASS1="$PI_DIR/jepa-e0.pth.tar"' in text
    assert 'VANILLA_PASS1="$VANILLA_DIR/jepa-e0.pth.tar"' in text


def test_relay_validates_before_stopping_or_launching():
    text = SCRIPT.read_text(encoding="utf-8")

    checkpoint_validation = text.index("checkpoint_metadata()")
    pi_wait = text.index('wait_for_pass_checkpoint "$PI_PASS1"')
    pi_stop = text.index('stop_at_boundary "$PI_PID" pi')
    vanilla_launch = text.index("[launch vanilla]")
    assert checkpoint_validation < pi_wait < pi_stop < vanilla_launch
    assert 'torch.load(path, map_location="cpu", weights_only=False)' in text
    assert '"training_complete": False' in text
    assert "os.replace(temporary, path)" in text


def test_relay_enforces_matched_schedule_and_provenance():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "PI and vanilla configs differ outside the intended PI fields" in text
    assert "PI and vanilla repository commits differ" in text
    assert "PI and vanilla common initialization differs" in text
    assert "it no longer owns $config" in text
    assert '"configured_scheduler_horizon_optimizer_steps": 111464' in text
    assert '"intentional_boundary_stop": True' in text
    assert "tmux" not in text.lower()
