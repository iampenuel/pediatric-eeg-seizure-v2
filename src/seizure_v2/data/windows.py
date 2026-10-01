from seizure_v2.data.splits import individual_id, partition_for

FS = 256
WINDOW_SAMPLES = 2048
PREPROCESSING_VERSION = "bipolar18-int16-any-overlap-8s-v1"


def seizure_samples(intervals, n_samples):
    result = []
    for start, end in intervals:
        a, b = round(start * FS), round(end * FS)
        if abs(start * FS - a) > 1e-6 or abs(end * FS - b) > 1e-6:
            raise ValueError("Seizure boundary not aligned to sampling grid")
        if a < 0 or b <= a or b > n_samples:
            raise ValueError("Seizure interval outside recording")
        if result and a < result[-1][1]:
            raise ValueError("Seizure intervals overlap or are unsorted")
        result.append((a, b))
    return result


def window_rows(meta):
    case = meta["case_id"]
    intervals = seizure_samples(meta["seizures"], meta["n_samples"])
    for start in range(0, meta["n_samples"] - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        end = start + WINDOW_SAMPLES
        overlap = sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)
        yield {
            "window_id": f"{meta['recording']}:{start}",
            "individual_id": individual_id(case), "case_id": case,
            "recording": meta["recording"], "partition": partition_for(case),
            "start_sample": start, "end_sample": end,
            "start_seconds": start / FS, "end_seconds": end / FS,
            "label": int(overlap > 0), "seizure_overlap_samples": overlap,
            "seizure_overlap_seconds": overlap / FS,
            "source_sha256": meta["source_sha256"],
            "preprocessing_version": PREPROCESSING_VERSION,
        }
