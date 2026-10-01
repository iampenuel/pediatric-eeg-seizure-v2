from pathlib import Path
import numpy as np
from seizure_v2.common import sha256, write_json, read_json, object_hash
from seizure_v2.data.channels import CHANNELS, select_channels
from seizure_v2.data.windows import FS, PREPROCESSING_VERSION, seizure_samples
from seizure_v2.data.download import BASE_URL, recording_url


def cache_paths(root, recording):
    base = Path(root) / "cache" / recording.removesuffix(".edf")
    return base.with_suffix(".npy"), base.with_suffix(".json")


def fingerprint(recording, source_sha, intervals):
    return object_hash({"recording": recording, "source_sha256": source_sha, "seizures": intervals, "version": PREPROCESSING_VERSION, "channels": CHANNELS})


def verified_cache(root, recording, source_sha, intervals):
    array_path, metadata_path = cache_paths(root, recording)
    if not metadata_path.exists():
        return None
    meta = read_json(metadata_path)
    if meta.get("input_fingerprint") != fingerprint(recording, source_sha, intervals):
        return None
    if meta["status"] == "excluded":
        return meta
    if not array_path.exists() or sha256(array_path) != meta.get("cache_sha256"):
        return None
    return meta


def convert_edf(source, root, recording, source_sha, intervals):
    import pyedflib
    if sha256(source) != source_sha:
        raise ValueError("Unverified EDF source")
    array_path, metadata_path = cache_paths(root, recording)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    with pyedflib.EdfReader(str(source)) as reader:
        labels = reader.getSignalLabels()
        meta = {"recording": recording, "case_id": recording.split("/")[0],
                "source_sha256": source_sha, "seizures": intervals,
                "source_url": recording_url(BASE_URL, recording),
                "duration_seconds": float(reader.getFileDuration()),
                "input_fingerprint": fingerprint(recording, source_sha, intervals),
                "preprocessing_version": PREPROCESSING_VERSION, "source_labels": labels}
        try:
            indices = select_channels(labels)
        except ValueError as error:
            meta.update(status="excluded", reason=str(error))
            write_json(metadata_path, meta)
            return meta
        rates = [reader.getSampleFrequency(index) for index in indices]
        lengths = [int(reader.getNSamples()[index]) for index in indices]
        if any(rate != FS for rate in rates) or len(set(lengths)) != 1:
            meta.update(status="excluded", reason="Unsupported sampling rate or unequal channel lengths")
            write_json(metadata_path, meta)
            return meta
        n_samples = lengths[0]
        seizure_samples(intervals, n_samples)
        calibration = []
        for index in indices:
            header = reader.getSignalHeader(index)
            unit = header["dimension"].strip().lower().replace("µ", "u").replace("μ", "u")
            if unit not in {"uv", "mv", "v"}:
                raise ValueError(f"Unrecognized physical unit: {unit}")
            multiplier = {"uv": 1., "mv": 1000., "v": 1e6}[unit]
            dmin, dmax = header["digital_min"], header["digital_max"]
            if dmin < -32768 or dmax > 32767 or dmax <= dmin:
                raise ValueError("EDF calibration cannot be represented as int16")
            scale = (header["physical_max"] - header["physical_min"]) / (dmax - dmin) * multiplier
            offset = header["physical_min"] * multiplier - dmin * scale
            if not np.isfinite([scale, offset]).all() or scale <= 0:
                raise ValueError("Invalid channel calibration")
            calibration.append({"scale_uv": scale, "offset_uv": offset, "edf_header": header})
        temp = array_path.with_name(array_path.name + ".tmp")
        array = np.lib.format.open_memmap(temp, mode="w+", dtype=np.int16, shape=(18, n_samples))
        for target, source_index in enumerate(indices):
            for start in range(0, n_samples, 262144):
                size = min(262144, n_samples - start)
                values = reader.readSignal(source_index, start, size, digital=True)
                if len(values) != size or values.min() < -32768 or values.max() > 32767:
                    raise ValueError("Invalid digital samples")
                array[target, start:start + size] = values
        array.flush()
        del array
        temp.replace(array_path)
        meta.update(status="prepared", n_samples=n_samples, channels=CHANNELS,
                    source_channel_indices=indices, calibration=calibration,
                    sampling_rate=FS, cache_sha256=sha256(array_path),
                    discarded_tail_samples=n_samples % 2048)
        write_json(metadata_path, meta)
        return meta


def to_microvolts(samples, meta):
    scale = np.asarray([c["scale_uv"] for c in meta["calibration"]], dtype=np.float64)
    offset = np.asarray([c["offset_uv"] for c in meta["calibration"]], dtype=np.float64)
    return (samples.astype(np.float64) * scale[:, None] + offset[:, None]).astype(np.float32)


def read_window(root, recording, start, end):
    array_path, metadata_path = cache_paths(root, recording)
    meta = read_json(metadata_path)
    if not 0 <= start < end <= meta["n_samples"]:
        raise ValueError("Window outside recording")
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    return to_microvolts(array[:, start:end], meta)
