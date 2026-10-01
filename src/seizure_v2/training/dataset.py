from collections import OrderedDict
import mmap
import numpy as np
from torch.utils.data import Dataset
from seizure_v2.common import read_json
from seizure_v2.data.edf import cache_paths, to_microvolts, cached_samples
from seizure_v2.data.normalization import normalize


def open_window_cache(path, random_access=False):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    mapping = getattr(array, "_mmap", None)
    if random_access and hasattr(mapping, "madvise") and hasattr(mmap, "MADV_RANDOM"):
        # Avoid fetching unused sequential regions for shuffled windows.
        # This also limits read-ahead for legacy channel-major caches.
        # This is an optional OS I/O hint; it does not modify samples or ordering.
        try:
            mapping.madvise(mmap.MADV_RANDOM)
        except OSError:
            pass  # Filesystems/platforms may decline an advisory optimization.
    return array


class WindowDataset(Dataset):
    def __init__(self, root, rows, scaler, random_access=False):
        self.root, self.rows, self.scaler = str(root), rows, scaler
        self.random_access = random_access
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        recording = row["recording"]
        if recording not in self.cache:
            array, meta = cache_paths(self.root, recording)
            self.cache[recording] = (open_window_cache(array, self.random_access), read_json(meta))
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(recording)
        array, meta = self.cache[recording]
        values = to_microvolts(cached_samples(array, meta, row["start_sample"], row["end_sample"]), meta)
        return normalize(values, self.scaler), np.float32(row["label"])
