"""Verified, resumable downloads into an explicitly owned cache."""
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from seizure_v2.common import sha256

BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"


def session():
    client = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    client.mount("https://", HTTPAdapter(max_retries=retry))
    return client


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe source path: {name}")
    return name


def recording_url(base, recording):
    # S3 interprets a literal '+' as a space; preserve the inventory's exact key.
    return base + quote(safe_relative(recording), safe="/")


def download(url, destination, expected_sha=None, client=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_sha is None or sha256(destination) == expected_sha):
        return destination
    part = destination.with_name(destination.name + ".part")
    if part.exists() and expected_sha and sha256(part) == expected_sha:
        part.replace(destination)
        return destination
    own_client = client is None
    client = client or session()
    try:
        offset = part.stat().st_size if part.exists() else 0
        response = client.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, stream=True, timeout=(20, 120))
        if response.status_code == 416:
            response.close()
            part.unlink(missing_ok=True)
            response = client.get(url, stream=True, timeout=(20, 120))
            offset = 0
        with response:
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned an unexpected byte range")
            with part.open("ab" if append else "wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        if expected_sha and sha256(part) != expected_sha:
            part.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch: {destination.name}; retry to download again")
        part.replace(destination)
        return destination
    finally:
        if own_client:
            client.close()


def inventory(root):
    root = Path(root) / "source_metadata"
    with session() as client:
        checksums = download(BASE_URL + "SHA256SUMS.txt", root / "SHA256SUMS.txt", client=client)
        sums = {}
        for line in checksums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            sums[safe_relative(name.lstrip("*"))] = digest
        for name in ["RECORDS", "RECORDS-WITH-SEIZURES"] + [f"chb{i:02d}/chb{i:02d}-summary.txt" for i in range(1, 25)]:
            download(BASE_URL + name, root / name, sums[name], client)
    records = [safe_relative(line.strip()) for line in (root / "RECORDS").read_text().splitlines() if line.strip()]
    seizures = set((root / "RECORDS-WITH-SEIZURES").read_text().split())
    if len(records) != len(set(records)) or not seizures <= set(records):
        raise ValueError("Invalid source inventory")
    return root, records, seizures, sums
