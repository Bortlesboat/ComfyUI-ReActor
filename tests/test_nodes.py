import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


@unittest.skipUnless(os.environ.get("COMFYUI_ROOT"), "Set COMFYUI_ROOT for node integration tests")
class NodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.append(os.environ["COMFYUI_ROOT"])
        import folder_paths
        import torch

        cls.torch = torch
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        path = Path(__file__).resolve().parents[1] / "nodes.py"
        spec = importlib.util.spec_from_file_location("reactor_nodes_under_test", path)
        cls.nodes = importlib.util.module_from_spec(spec)
        with mock.patch.object(folder_paths, "models_dir", cls.directory.name):
            spec.loader.exec_module(cls.nodes)

    def setUp(self):
        self.image = self.torch.ones((1, 16, 24, 3))
        self.node = self.nodes.reactor()
        device = mock.patch.object(
            self.nodes.model_management, "get_torch_device", return_value=self.torch.device("cpu")
        )
        device.start()
        self.addCleanup(device.stop)
        interrupted = mock.patch.object(self.nodes.state, "interrupted", False)
        interrupted.start()
        self.addCleanup(interrupted.stop)
        cancellation = mock.patch.object(
            self.nodes.model_management, "processing_interrupted", return_value=False
        )
        cancellation.start()
        self.addCleanup(cancellation.stop)

    def execute(self, **changes):
        inputs = dict(
            enabled=True, input_image=self.image, swap_model="inswapper_128.onnx",
            detect_gender_source="no", detect_gender_input="no", source_faces_index="0",
            input_faces_index="0", console_log_level=0, face_restore_model="none",
            face_restore_visibility=1.0, codeformer_weight=0.5,
            facedetection="retinaface_resnet50", source_image=self.image,
        )
        inputs.update(changes)
        return self.node.execute(**inputs)

    def assert_passthrough_outputs(self, result):
        self.assertEqual(len(result), 3)
        self.assertTrue(self.torch.equal(result[0], self.image))
        self.assertTrue(self.torch.equal(result[2], self.image))

    def test_disabled_node_returns_all_declared_outputs(self):
        self.assert_passthrough_outputs(self.execute(enabled=False))

    def test_missing_source_returns_all_declared_outputs(self):
        self.assert_passthrough_outputs(self.execute(source_image=None))

    def test_none_face_model_is_a_missing_source(self):
        with mock.patch.object(self.nodes.sfw, "nsfw_image", side_effect=AssertionError("No source")):
            self.assert_passthrough_outputs(self.execute(source_image=None, face_model="none"))

    def test_rejected_image_reports_a_node_error(self):
        with mock.patch.object(self.nodes.sfw, "nsfw_image", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "content check rejected input image 1"):
                self.execute()

    def test_rejected_batch_never_swaps_a_shortened_batch(self):
        images = self.image.repeat(2, 1, 1, 1)
        with mock.patch.object(self.nodes.sfw, "nsfw_image", side_effect=[False, True]):
            with mock.patch.object(self.nodes.FaceSwapScript, "process") as process:
                with self.assertRaisesRegex(RuntimeError, "content check rejected input image 2"):
                    self.execute(input_image=images)
                process.assert_not_called()

    def test_interruption_uses_comfyui_cancellation(self):
        with mock.patch.object(self.nodes.state, "interrupted", True):
            with self.assertRaises(self.nodes.model_management.InterruptProcessingException):
                self.execute()

    def test_empty_batch_reports_a_node_error(self):
        with self.assertRaisesRegex(RuntimeError, "input image batch is empty"):
            self.execute(input_image=self.image[:0], nsfw_detection=False)

    def test_unload_node_clears_classifier_cache(self):
        with mock.patch.object(self.nodes.sfw, "clear_model_cache") as clear_cache:
            result = self.nodes.ReActorUnload().execute(self.image)
        clear_cache.assert_called_once_with()
        self.assertTrue(self.torch.equal(result[0], self.image))

    def test_detector_off_skips_classification_and_preserves_batch(self):
        images = self.image.repeat(2, 1, 1, 1)
        with mock.patch.object(self.nodes.sfw, "nsfw_image", side_effect=AssertionError("Detector used")):
            with mock.patch.object(self.nodes.FaceSwapScript, "process"):
                with mock.patch.object(self.nodes.sfw, "clear_model_cache") as clear_cache:
                    result = self.execute(input_image=images, nsfw_detection=False)
        self.assertEqual(len(result), 3)
        self.assertTrue(self.torch.equal(result[0], images))
        self.assertTrue(self.torch.equal(result[2], images))
        clear_cache.assert_called_once_with()

    def test_fast_node_forwards_detector_switch(self):
        with mock.patch.object(self.nodes.reactor, "execute", return_value=(self.image, None, self.image)) as execute:
            result = self.nodes.ReActorPlusOpt().execute(
                enabled=True, input_image=self.image, swap_model="inswapper_128.onnx",
                facedetection="retinaface_resnet50", face_restore_model="none",
                face_restore_visibility=1.0, codeformer_weight=0.5,
                source_image=self.image, nsfw_detection=False,
            )
        self.assertEqual(len(result), 3)
        self.assertIs(execute.call_args.kwargs["nsfw_detection"], False)

    def test_both_nodes_expose_an_enabled_default_switch(self):
        with mock.patch.object(self.nodes, "get_restorers", return_value=[]):
            for node_type in (self.nodes.reactor, self.nodes.ReActorPlusOpt):
                with self.subTest(node=node_type.__name__):
                    control = node_type.INPUT_TYPES()["optional"]["nsfw_detection"]
                    self.assertEqual(control[0], "BOOLEAN")
                    self.assertIs(control[1]["default"], True)
