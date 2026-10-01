from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset
from seizure_v2.common import read_json
from seizure_v2.data.edf import cache_paths, to_microvolts
from seizure_v2.data.normalization import normalize


class WindowDataset(Dataset):
    def __init__(self, root, rows, scaler):
        self.root, self.rows, self.scaler = str(root), rows, scaler
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        recording = row["recording"]
        if recording not in self.cache:
            array, meta = cache_paths(self.root, recording)
            self.cache[recording] = (np.load(array, mmap_mode="r", allow_pickle=False), read_json(meta))
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(recording)
        array, meta = self.cache[recording]
        values = to_microvolts(array[:, row["start_sample"]:row["end_sample"]], meta)
        return normalize(values, self.scaler), np.float32(row["label"])
