import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path("app/plan_common/models/dino.py")


def load_dino_module(hub_dir):
    calls = []

    class FakeHub:
        _validate_not_a_forked_repo = None

        @staticmethod
        def get_dir():
            return str(hub_dir)

        @staticmethod
        def load(*args, **kwargs):
            calls.append((args, kwargs))
            return "loaded"

    fake_torch = types.ModuleType("torch")
    fake_torch.hub = FakeHub()
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = type("Module", (), {})
    fake_torch.nn = fake_nn

    spec = importlib.util.spec_from_file_location("dino_hub_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": fake_torch, "torch.nn": fake_nn}):
        spec.loader.exec_module(module)
    return module, calls


class TestDinoHubLoading(unittest.TestCase):
    def test_existing_main_checkout_is_loaded_without_github_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub_dir = Path(temporary)
            repository = hub_dir / "facebookresearch_dinov2_main"
            repository.mkdir()
            (repository / "hubconf.py").write_text("", encoding="utf-8")
            module, calls = load_dino_module(hub_dir)

            with patch.dict(
                module.os.environ,
                {"JEPAWM_DINOV2_REPO": ""},
            ):
                self.assertEqual(
                    module._load_dinov2_from_cache_or_github("dinov2_vits14"),
                    "loaded",
                )
            self.assertEqual(calls[0][0], (str(repository), "dinov2_vits14"))
            self.assertEqual(calls[0][1], {"source": "local"})

    def test_missing_checkout_uses_pinned_main_without_validation_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, calls = load_dino_module(Path(temporary))

            with patch.dict(
                module.os.environ,
                {"JEPAWM_DINOV2_REPO": ""},
            ):
                self.assertEqual(
                    module._load_dinov2_from_cache_or_github("dinov2_vits14"),
                    "loaded",
                )
            self.assertEqual(
                calls[0][0],
                ("facebookresearch/dinov2:main", "dinov2_vits14"),
            )
            self.assertEqual(
                calls[0][1],
                {"trust_repo": True, "skip_validation": True},
            )


if __name__ == "__main__":
    unittest.main()
