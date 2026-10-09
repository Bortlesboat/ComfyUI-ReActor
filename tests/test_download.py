import http.client
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from tqdm import tqdm

from reactor_download import download


PAYLOAD = b"model download regression fixture\n" * 1000


class DownloadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests += 1
        if self.path == "/error":
            self.send_error(404)
            return
        payload = b"" if self.path == "/empty" else PAYLOAD
        self.send_response(200)
        if self.path != "/unknown-length":
            length = len(payload) + (1024 if self.path == "/truncated" else 0)
            self.send_header("Content-Length", str(length))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.destination = self.root / "model.onnx"
        self.server.requests = 0
        quiet_progress = mock.patch(
            "reactor_download.tqdm", side_effect=lambda **kw: tqdm(**kw, disable=True)
        )
        quiet_progress.start()
        self.addCleanup(quiet_progress.stop)

    def fetch(self, route, destination=None):
        download(self.base_url + route, destination or self.destination, "model.onnx")

    def test_complete_download_uses_one_request(self):
        self.fetch("/ok")
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(self.server.requests, 1)
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    def test_creates_missing_parent_directories(self):
        destination = self.root / "nested" / "models" / "model.onnx"
        self.fetch("/ok", destination)
        self.assertEqual(destination.read_bytes(), PAYLOAD)

    def test_download_without_content_length(self):
        self.fetch("/unknown-length")
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_truncated_download_leaves_no_model_or_partial_file(self):
        with self.assertRaises((OSError, http.client.IncompleteRead)):
            self.fetch("/truncated")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_truncated_download_preserves_existing_model(self):
        self.destination.write_bytes(b"existing complete model")
        with self.assertRaises((OSError, http.client.IncompleteRead)):
            self.fetch("/truncated")
        self.assertEqual(self.destination.read_bytes(), b"existing complete model")
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    def test_empty_download_is_not_published(self):
        with self.assertRaises(OSError):
            self.fetch("/empty")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_http_failure_preserves_existing_model(self):
        self.destination.write_bytes(b"existing complete model")
        with self.assertRaises(OSError):
            self.fetch("/error")
        self.assertEqual(self.destination.read_bytes(), b"existing complete model")
        self.assertEqual(list(self.root.iterdir()), [self.destination])

    def test_replace_failure_removes_temporary_file(self):
        self.destination.write_bytes(b"existing complete model")
        with mock.patch("reactor_download.os.replace", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                self.fetch("/ok")
        self.assertEqual(self.destination.read_bytes(), b"existing complete model")
        self.assertEqual(list(self.root.iterdir()), [self.destination])
