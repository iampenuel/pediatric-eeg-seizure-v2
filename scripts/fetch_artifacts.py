"""Build-time artifact retrieval; never put credentials or unverified models in Git."""
import os
import tarfile
from pathlib import Path
from seizure_v2.data.download import download


def main():
    url, digest = os.environ.get("SEIZURE_ARTIFACT_URL"), os.environ.get("SEIZURE_ARTIFACT_SHA256")
    if not url and not digest:
        print("No artifact configured: the app will explicitly show unavailable inference.")
        return
    if not url or not digest or not url.startswith("https://") or len(digest) != 64:
        raise ValueError("Both HTTPS artifact URL and SHA-256 are required")
    archive = download(url, Path("artifacts/release.tar.gz"), digest)
    target = Path("artifacts/demo").resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not (target / member.name).resolve().is_relative_to(target) or member.issym() or member.islnk():
                raise ValueError("Unsafe artifact archive")
        tar.extractall(target, filter="data")
    print("Verified artifact installed.")


if __name__ == "__main__":
    main()
