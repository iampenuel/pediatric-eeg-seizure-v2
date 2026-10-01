import numpy as np
from seizure_v2.common import read_json, object_hash
from seizure_v2.data.channels import CHANNELS


def fit_scaler(root, rows, dataset_hash):
    from seizure_v2.data.edf import cache_paths, to_microvolts
    if not rows or any(row["partition"] != "train" for row in rows):
        raise ValueError("Scaler fitting accepts training rows only")
    recordings = {}
    for row in rows:
        recordings[row["recording"]] = max(recordings.get(row["recording"], 0), row["end_sample"])
    count = 0
    mean = np.zeros(18, dtype=np.float64)
    m2 = np.zeros(18, dtype=np.float64)
    for recording, end in sorted(recordings.items()):
        array_path, meta_path = cache_paths(root, recording)
        array, meta = np.load(array_path, mmap_mode="r", allow_pickle=False), read_json(meta_path)
        for start in range(0, end, 65536):
            block = to_microvolts(array[:, start:min(start + 65536, end)], meta).astype(np.float64)
            n = block.shape[1]
            block_mean = block.mean(axis=1)
            delta = block_mean - mean
            m2 += ((block - block_mean[:, None]) ** 2).sum(axis=1) + delta ** 2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
    std = np.sqrt(m2 / count)
    scaler = {"mean_uv": mean.tolist(), "std_uv": np.maximum(std, 1e-6).tolist(),
              "channels": CHANNELS, "fit_partition": "train", "sample_count_per_channel": count,
              "recordings": sorted(recordings), "dataset_hash": dataset_hash,
              "constant_channels": [CHANNELS[i] for i in np.flatnonzero(std < 1e-6)]}
    scaler["sha256"] = object_hash(scaler)
    return scaler


def normalize(signal, scaler):
    if scaler["fit_partition"] != "train" or scaler["channels"] != CHANNELS:
        raise ValueError("Incompatible scaler")
    mean = np.asarray(scaler["mean_uv"], dtype=np.float32)[:, None]
    std = np.asarray(scaler["std_uv"], dtype=np.float32)[:, None]
    value = np.asarray(signal, dtype=np.float32)
    if value.shape[-2:] != (18, 2048) or not np.isfinite(value).all():
        raise ValueError("Expected finite EEG windows of shape (...,18,2048)")
    return (value - mean) / std
