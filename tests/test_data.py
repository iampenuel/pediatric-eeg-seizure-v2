import copy
import numpy as np
import pytest
from seizure_v2.data.annotations import parse_summary, annotations_for
from seizure_v2.data.channels import CHANNELS, select_channels
from seizure_v2.data.splits import SPLITS, individual_id, validate_splits
from seizure_v2.data.windows import window_rows, seizure_samples
from seizure_v2.data.edf import convert_edf, read_window, verified_cache
from seizure_v2.common import sha256
from seizure_v2.data.annotations import reconcile_inventory


def test_frozen_partition_has_23_independent_people():
    assert validate_splits(SPLITS)
    groups = [{individual_id(case) for case in cases} for cases in SPLITS.values()]
    assert [len(g) for g in groups] == [15, 4, 4]
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            assert not left & right


def test_pair_cannot_be_separated():
    split = copy.deepcopy(SPLITS)
    split["test"].remove("chb21")
    split["train"].append("chb21")
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(split)


def test_known_source_inventory_correction_requires_evidence():
    corrected, log = reconcile_inventory({"chb07/chb07_18.edf"}, {"chb07/chb07_19.edf.seizures"})
    assert corrected == {"chb07/chb07_19.edf"} and len(log) == 1
    with pytest.raises(ValueError):
        reconcile_inventory({"chb07/chb07_18.edf"}, set())


def test_parser_variants_and_missing_chb24():
    text = """File Name: a.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 8 seconds
Seizure 1 End Time: 16 seconds
Seizure Start Time: 30 seconds
Seizure End Time: 32 seconds
File Name: b.edf
Number of Seizures in File: 0
"""
    assert parse_summary(text) == {"a.edf": [[8., 16.], [30., 32.]], "b.edf": []}
    assert annotations_for("chb24/x.edf", {}, set()) == []
    with pytest.raises(ValueError, match="Missing"):
        annotations_for("chb02/x.edf", {}, set())
    with pytest.raises(ValueError, match="disagreement"):
        annotations_for("chb02/b.edf", {"b.edf": []}, {"chb02/b.edf"})
    with pytest.raises(ValueError):
        parse_summary(text.replace("File: 2", "File: 3"))


def metadata(n=256 * 3600, intervals=None):
    return {"case_id": "chb02", "recording": "chb02/chb02_16.edf", "source_sha256": "fixture", "n_samples": n, "seizures": intervals or []}


def test_half_open_boundaries_and_partial_tail():
    rows = list(window_rows(metadata(2048 * 4 + 12, [[8, 16], [23.5, 24.5]])))
    assert [r["label"] for r in rows] == [0, 1, 1, 1]
    assert [r["seizure_overlap_samples"] for r in rows] == [0, 2048, 128, 128]
    assert all(r["end_sample"] - r["start_sample"] == 2048 for r in rows)
    assert rows[0]["partition"] == "train"
    with pytest.raises(ValueError):
        seizure_samples([[0, 12]], 2048)


def test_chb02_known_seizure_has_eleven_positive_windows():
    rows = list(window_rows(metadata(256 * 959, [[130, 212]])))
    positives = [r for r in rows if r["label"]]
    assert len(positives) == 11
    assert positives[0]["start_seconds"] == 128
    assert positives[-1]["end_seconds"] == 216


def test_channels_preserve_order_and_first_duplicate():
    labels = ["-", *CHANNELS, "T8-P8", "ECG"]
    assert select_channels(labels) == list(range(1, 19))
    with pytest.raises(ValueError, match="Unsupported montage"):
        select_channels(CHANNELS[:-1])


def test_real_edf_roundtrip_scaling_and_recovery(tmp_path):
    import pyedflib
    from pyedflib import highlevel
    source = tmp_path / "input.edf"
    values = np.tile(np.arange(4096, dtype=np.int32) - 2000, (18, 1))
    headers = highlevel.make_signal_headers(CHANNELS, sample_frequency=256, physical_min=-1000, physical_max=1000, digital_min=-32768, digital_max=32767)
    highlevel.write_edf(str(source), values, headers, digital=True)
    digest = sha256(source)
    meta = convert_edf(source, tmp_path, "chb02/test.edf", digest, [[1, 2]])
    cached = np.load(tmp_path / "cache/chb02/test.npy", mmap_mode="r")
    assert cached.shape == (4096, 18) and cached.dtype == np.int16
    np.testing.assert_array_equal(cached, values.T)
    assert meta["cache_layout"] == "samples_channels_v1"
    signal = read_window(tmp_path, "chb02/test.edf", 0, 2048)
    with pyedflib.EdfReader(str(source)) as reader:
        expected = np.vstack([reader.readSignal(i, 0, 2048) for i in range(18)])
    assert signal.shape == (18, 2048)
    np.testing.assert_allclose(signal, expected, atol=1e-5)
    assert verified_cache(tmp_path, "chb02/test.edf", digest, [[1, 2]]) == meta
    cache = tmp_path / "cache/chb02/test.npy"
    cache.write_bytes(b"interrupted cache")
    assert verified_cache(tmp_path, "chb02/test.edf", digest, [[1, 2]]) is None
    convert_edf(source, tmp_path, "chb02/test.edf", digest, [[1, 2]])
    assert verified_cache(tmp_path, "chb02/test.edf", digest, [[1, 2]])


def test_cache_layouts_preserve_calibration_and_window_boundaries(tmp_path):
    from seizure_v2.data.edf import cached_samples, to_microvolts
    values = np.random.default_rng(42).integers(-32768, 32768, (18, 6145), dtype=np.int16)
    meta = {"calibration": [{"scale_uv": .017 * (i + 1), "offset_uv": i - 9.25} for i in range(18)]}
    time_major = {**meta, "cache_layout": "samples_channels_v1"}
    for start, end in [(0, 2048), (2048, 4096), (4096, 6144), (6144, 6145)]:
        old = to_microvolts(cached_samples(values, meta, start, end), meta)
        new = to_microvolts(cached_samples(values.T.copy(), time_major, start, end), time_major)
        np.testing.assert_array_equal(old, new)
    with pytest.raises(ValueError, match="layout"):
        cached_samples(values, {"cache_layout": "unknown"}, 0, 2048)
