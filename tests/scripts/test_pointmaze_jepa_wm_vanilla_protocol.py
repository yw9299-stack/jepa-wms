import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from scripts.generate_pointmaze_jepa_wm_vanilla_training_config import (
    EXPECTED_BASE_FOLDER,
    VANILLA_FOLDER,
    derive_vanilla_config,
)


class TestPointMazeJepaWmVanillaProtocol(unittest.TestCase):
    def test_only_intended_training_fields_change(self):
        base_path = Path("configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml")
        base = yaml.safe_load(base_path.read_text())
        vanilla = derive_vanilla_config(base)

        self.assertEqual(base["folder"], EXPECTED_BASE_FOLDER)
        self.assertEqual(vanilla["folder"], VANILLA_FOLDER)
        self.assertTrue(base["model"]["predictor"]["planner_identified_input_scale"])
        self.assertFalse(vanilla["model"]["predictor"]["planner_identified_input_scale"])
        self.assertTrue(base["planner_identified"]["enabled"])
        self.assertEqual(vanilla["planner_identified"], {"enabled": False})

        restored = deepcopy(vanilla)
        restored["folder"] = base["folder"]
        restored["model"]["predictor"]["planner_identified_input_scale"] = True
        restored["planner_identified"] = deepcopy(base["planner_identified"])
        self.assertEqual(restored, base)

    def test_schedule_remains_five_matched_passes(self):
        base_path = Path("configs/pi_ltc_cross_model/pointmaze_jepa_wm_pi_ltc_5pass.yaml")
        vanilla = derive_vanilla_config(yaml.safe_load(base_path.read_text()))
        optimization = vanilla["optimization"]["transition_model"]
        self.assertEqual(optimization["num_epochs"], 5)
        self.assertEqual(optimization["gradient_accumulation_steps"], 8)
        self.assertEqual(vanilla["data"]["loader"]["batch_size"], 16)
        self.assertEqual(vanilla["data"]["loader"]["num_workers"], 12)


if __name__ == "__main__":
    unittest.main()
