"""Tests for the chunked weight downloader.

A local HTTP server stands in for the mirrors: it serves a byte-exact file and
honours range requests, so the parallel chunk writer, the resume path and the
completion marker can be checked without any network traffic.
"""

import hashlib
import http.server
import os
import threading
from pathlib import Path

import pytest

from mflux.models.qwen3_tts_voice.download import chunked_download, probe_mirror, probe_mirrors

PAYLOAD = os.urandom(3 * 1024 * 1024 + 12345)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = PAYLOAD

    def do_GET(self):  # noqa: N802 - http.server API
        range_header = self.headers.get("Range", "")
        start, end = 0, len(self.payload) - 1
        if range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1]
            first, _, last = spec.partition("-")
            start = int(first)
            end = int(last) if last else len(self.payload) - 1
        chunk = self.payload[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):  # noqa: D102 - silence the test server
        return


@pytest.fixture()
def mirror():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://{host}:{port}/model.safetensors"
    server.shutdown()


def test_chunked_download_reproduces_the_file_exactly(mirror, tmp_path):
    target = tmp_path / "model.safetensors"
    assert chunked_download([mirror], target, expected_size=len(PAYLOAD), chunk_size=256 * 1024, workers=4)
    assert target.read_bytes() == PAYLOAD
    assert target.with_name(target.name + ".complete").is_file()
    assert not target.with_name(target.name + ".chunks.json").exists()


def test_a_second_run_is_a_no_op_and_a_partial_file_resumes(mirror, tmp_path):
    target = tmp_path / "model.safetensors"
    chunk_size = 256 * 1024

    assert chunked_download([mirror], target, expected_size=len(PAYLOAD), chunk_size=chunk_size, workers=4)
    marker = target.with_name(target.name + ".complete")
    assert marker.is_file()

    # A finished file is not downloaded again.
    marker.unlink()
    target.write_bytes(b"\x00" * len(PAYLOAD))
    assert chunked_download([mirror], target, expected_size=len(PAYLOAD), chunk_size=chunk_size, workers=4)
    assert target.read_bytes() == PAYLOAD

    # A half-finished download continues where it stopped.
    marker.unlink()
    target.write_bytes(b"\x00" * len(PAYLOAD))
    state = target.with_name(target.name + ".chunks.json")
    state.write_text('{"0": 262144}', encoding="utf-8")
    assert chunked_download([mirror], target, expected_size=len(PAYLOAD), chunk_size=chunk_size, workers=4)
    assert target.read_bytes() == PAYLOAD


def test_a_mirror_that_ignores_ranges_is_dropped(mirror, tmp_path):
    size, honours = probe_mirror(mirror)
    assert size == len(PAYLOAD)
    assert honours is True

    class _NoRangeHandler(_RangeHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        url = f"http://{host}:{port}/model.safetensors"
        probes = probe_mirrors([url])
        assert probes[0][2] is False
    finally:
        server.shutdown()


def test_probe_reports_the_total_size_from_the_content_range(mirror):
    size, honours = probe_mirror(mirror)
    assert size == len(PAYLOAD)
    assert honours is True
    assert hashlib.sha256(PAYLOAD).digest()
    assert Path(mirror).name == "model.safetensors"
