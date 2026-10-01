from seizure_v2.data.download import download, safe_relative
import hashlib
import pytest


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status_code, self.headers = data, status, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def close(self):
        pass
    def raise_for_status(self):
        assert self.status_code < 400
    def iter_content(self, size):
        yield self.data


class Client:
    def __init__(self, response):
        self.response, self.calls = response, []
    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.mark.parametrize("range_supported", [True, False])
def test_resume_and_server_ignoring_range(tmp_path, range_supported):
    body = b"abcdef"
    target = tmp_path / "sample.edf"
    target.with_suffix(".edf.part").write_bytes(body[:3])
    client = Client(Response(body[3:] if range_supported else body, 206 if range_supported else 200, {"Content-Range": "bytes 3-5/6"}))
    download("https://example.invalid/data", target, hashlib.sha256(body).hexdigest(), client)
    assert target.read_bytes() == body
    assert client.calls[0]["headers"] == {"Range": "bytes=3-"}
    download("https://example.invalid/data", target, hashlib.sha256(body).hexdigest(), client)
    assert len(client.calls) == 1


def test_invalid_hash_and_path_rejected(tmp_path):
    target = tmp_path / "bad"
    with pytest.raises(ValueError, match="Checksum"):
        download("https://example.invalid/data", target, "0" * 64, Client(Response(b"wrong")))
    assert not target.exists()
    with pytest.raises(ValueError):
        safe_relative("../secret")
