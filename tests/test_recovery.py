import io
import json
from pathlib import Path
import tarfile
import pytest
from seizure_v2.common import sha256
from seizure_v2.recovery import snapshot, inspect_archive, restore, checkpoint, PREFIX


def test_compact_roundtrip_and_tampering(tmp_path):
    run = tmp_path / "run"
    (run / "baseline").mkdir(parents=True)
    (run / "baseline/last.pt").write_bytes(b"optimizer-and-rng-state")
    (run / "baseline/scaler.json").write_text('{"fit_partition":"train"}')
    (run / "cache").mkdir()
    (run / "cache/large.npy").write_bytes(b"excluded")
    (run / "raw.edf").write_bytes(b"excluded")
    (run / "demo.tar.gz").write_bytes(b"separately-distributed")
    (run / "manifests/recording-metadata/chb02").mkdir(parents=True)
    (run / "manifests/recording-metadata/chb02/chb02_01.json").write_text('{"calibration":[],"source_channel_indices":[]}')
    (run / "manifests/source-metadata/chb02").mkdir(parents=True)
    (run / "manifests/source-metadata/chb02/chb02-summary.txt").write_text("verified annotations")
    event = snapshot(run, tmp_path / "snapshot.tar.gz", "baseline-epoch-01")
    manifest = inspect_archive(event["path"], event["sha256"])
    assert set(manifest["files"]) == {"baseline/last.pt", "baseline/scaler.json",
        "manifests/recording-metadata/chb02/chb02_01.json",
        "manifests/source-metadata/chb02/chb02-summary.txt"}
    target = tmp_path / "restored"
    restore(event["path"], event["sha256"], target)
    assert (target / "baseline/last.pt").read_bytes() == b"optimizer-and-rng-state"
    with pytest.raises(ValueError, match="never overwritten"):
        restore(event["path"], event["sha256"], target)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inspect_archive(event["path"], "0" * 64)


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "reports/../../escape.json"])
def test_recovery_rejects_traversal(tmp_path, name):
    archive = tmp_path / "bad.tar.gz"
    manifest = {"schema": 1, "files": {name: {"bytes": 0, "sha256": ""}}}
    with tarfile.open(archive, "w:gz") as target:
        payload = json.dumps(manifest).encode()
        member = tarfile.TarInfo("recovery.json")
        member.size = len(payload)
        target.addfile(member, io.BytesIO(payload))
        target.addfile(tarfile.TarInfo("run/" + name), io.BytesIO())
    with pytest.raises(ValueError, match="Unsafe recovery path"):
        restore(archive, sha256(archive), tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_checkpoint_requires_matching_local_ack(tmp_path, monkeypatch, capsys):
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setenv("EEG_RECOVERY_ROOT", str(root))
    monkeypatch.setenv("EEG_RECOVERY_DIR", str(tmp_path / "backups"))
    monkeypatch.setattr("sys.stdin", io.StringIO("incorrect\n"))
    with pytest.raises(RuntimeError, match="not acknowledged"):
        checkpoint("prepared")
    output = capsys.readouterr().out
    event = json.loads(output.removeprefix(PREFIX))
    assert inspect_archive(event["path"], event["sha256"])["stage"] == "prepared"


def test_recovery_rejects_symlink(tmp_path):
    root = tmp_path / "run"
    (root / "baseline").mkdir(parents=True)
    target = tmp_path / "outside"
    target.write_text("unrelated")
    (root / "baseline/last.pt").symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        snapshot(root, tmp_path / "snapshot.tar.gz", "test")


def test_local_receiver_rotates_only_verified_snapshots(tmp_path):
    from scripts.receive_recovery import receive
    root = tmp_path / "run"
    root.mkdir()
    destination = tmp_path / "local-recovery"
    digests = []
    for epoch in range(3):
        (root / "study.json").write_text(json.dumps({"epoch": epoch}))
        event = snapshot(root, tmp_path / "download.tar.gz", f"epoch-{epoch}")
        receive(event["path"], event["sha256"], destination)
        digests.append(event["sha256"])
    assert sha256(destination / "latest.tar.gz") == digests[-1]
    assert sha256(destination / "previous.tar.gz") == digests[-2]
    assert len(json.loads((destination / "receipts.json").read_text())) == 3
    (destination / "previous.tar.gz").write_bytes(b"unknown file")
    with pytest.raises(ValueError, match="unrecorded"):
        receive(event["path"], event["sha256"], destination)


def test_checkpoint_accepts_only_verified_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("EEG_RECOVERY_ROOT", str(tmp_path))
    monkeypatch.setenv("EEG_RECOVERY_DIR", str(tmp_path / "staging"))
    monkeypatch.setattr("seizure_v2.recovery.snapshot", lambda *args: {"sha256": "abc", "stage": "test"})
    monkeypatch.setattr("sys.stdin", io.StringIO("abc\n"))
    checkpoint("test")
