import importlib.util
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


MODEL_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_path = Path(self.directory.name) / "model-a"
        self.seed_model(self.model_path)
        self.prediction = [{"label": "sfw", "score": 1.0}]
        self.predict = mock.Mock(side_effect=lambda image: self.prediction)
        self.factory = mock.Mock(return_value=self.predict)
        transformers = types.ModuleType("transformers")
        transformers.pipeline = self.factory
        comfy = types.ModuleType("comfy")
        management = types.ModuleType("comfy.model_management")
        self.device = mock.Mock(return_value="cpu")
        management.get_torch_device = self.device
        comfy.model_management = management
        utils = types.ModuleType("reactor_utils")
        utils.download = mock.Mock()
        logger = types.ModuleType("scripts.reactor_logger")
        logger.logger = mock.Mock()
        modules = mock.patch.dict("sys.modules", {
            "transformers": transformers,
            "comfy": comfy,
            "comfy.model_management": management,
            "reactor_utils": utils,
            "scripts.reactor_logger": logger,
        })
        modules.start()
        self.addCleanup(modules.stop)
        path = Path(__file__).resolve().parents[1] / "scripts" / "reactor_sfw.py"
        spec = importlib.util.spec_from_file_location("classifier_under_test", path)
        self.sfw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.sfw)
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "green").save(image, format="PNG")
        self.image = image.getvalue()

    def seed_model(self, directory):
        directory.mkdir(parents=True)
        for name in MODEL_FILES:
            (directory / name).write_bytes(b"fixture")

    def classify(self, path=None):
        return self.sfw.nsfw_image(self.image, str(path or self.model_path))

    def test_repeated_images_reuse_classifier(self):
        self.assertFalse(self.classify())
        self.assertFalse(self.classify())
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(self.predict.call_count, 2)

    def test_model_path_change_reloads_classifier(self):
        other_model = Path(self.directory.name) / "model-b"
        self.seed_model(other_model)
        self.assertFalse(self.classify())
        self.assertFalse(self.classify(other_model))
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual(self.factory.call_args.kwargs["model"], str(other_model))

    def test_device_change_reloads_on_the_requested_device(self):
        self.assertFalse(self.classify())
        self.device.return_value = "mps"
        self.assertFalse(self.classify())
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual(self.factory.call_args.kwargs["device"], "mps")

    def test_unindexed_cuda_device_is_supported(self):
        self.device.return_value = "cuda"
        self.assertFalse(self.classify())
        self.assertEqual(self.factory.call_args.kwargs["device"], "cuda")

    def test_high_confidence_nsfw_is_still_rejected(self):
        self.prediction = [{"label": "nsfw", "score": 0.99}]
        self.assertTrue(self.classify())

    def test_threshold_boundary_is_unchanged(self):
        self.prediction = [{"label": "nsfw", "score": 0.979}]
        self.assertFalse(self.classify())

    def test_zero_byte_model_member_is_downloaded_again(self):
        missing = self.model_path / "model.safetensors"
        missing.write_bytes(b"")
        with mock.patch.object(
            self.sfw, "download", side_effect=lambda url, path, name: Path(path).write_bytes(b"repaired")
        ):
            self.assertTrue(self.sfw.ensure_nsfw_model(str(self.model_path)))
        self.assertEqual(missing.read_bytes(), b"repaired")

    def test_unload_releases_and_recreates_classifier(self):
        self.assertFalse(self.classify())
        self.sfw.clear_model_cache()
        self.assertFalse(self.classify())
        self.assertEqual(self.factory.call_count, 2)

    def test_failed_initialization_can_be_retried(self):
        self.factory.side_effect = [RuntimeError("load failed"), self.predict]
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.classify()
        self.assertFalse(self.classify())

    def test_import_does_not_require_transformers(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "reactor_sfw.py"
        spec = importlib.util.spec_from_file_location("lazy_classifier_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict("sys.modules", {"transformers": None}):
            spec.loader.exec_module(module)
