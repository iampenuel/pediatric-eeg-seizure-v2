import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from seizure_v2.common import write_json, read_json, sha256, object_hash
from seizure_v2.data.download import inventory, download, BASE_URL
from seizure_v2.data.annotations import parse_summary, annotations_for, reconcile_inventory
from seizure_v2.data.edf import verified_cache, convert_edf, cache_paths
from seizure_v2.data.splits import split_manifest, individual_id, partition_for
from seizure_v2.data.windows import window_rows, PREPROCESSING_VERSION


def prepare(root, **kwargs):
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare(root, **kwargs)


def _prepare(root, patients=None, recordings=None, full=False, evict_raw=False, mirror="physionet", workers=1):
    root = Path(root)
    if sum([bool(patients), bool(recordings), bool(full)]) != 1:
        raise ValueError("Choose exactly one of patients, recordings, or full")
    source_dir, all_records, seizures, sums = inventory(root)
    seizures, corrections = reconcile_inventory(seizures, sums)
    summaries = {f"chb{i:02d}": parse_summary((source_dir / f"chb{i:02d}/chb{i:02d}-summary.txt").read_text()) for i in range(1, 25)}
    annotations = {record: annotations_for(record, summaries[record.split('/')[0]], seizures) for record in all_records}
    if patients and not set(patients) <= {record.split('/')[0] for record in all_records}:
        raise ValueError("Unknown patient selection")
    selected = list(recordings) if recordings else [r for r in all_records if full or r.split('/')[0] in patients]
    if not selected or not set(selected) <= set(all_records):
        raise ValueError("Unknown/empty recording selection")
    if mirror not in {"physionet", "s3"} or not 1 <= workers <= 4:
        raise ValueError("Supported mirrors: physionet, s3; workers: 1–4")
    base = BASE_URL if mirror == "physionet" else "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/"
    def prepare_one(recording):
        intervals = annotations[recording]
        meta = verified_cache(root, recording, sums[recording], intervals)
        reused = meta is not None
        if meta is None:
            source = download(base + recording, root / "raw" / recording, sums[recording])
            meta = convert_edf(source, root, recording, sums[recording], intervals)
            if evict_raw:
                # Only this downloader's verified staging file, never arbitrary input files.
                source.unlink()
        return recording, "reused" if reused else meta["status"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_one, recording) for recording in selected]
        for number, future in enumerate(as_completed(futures), 1):
            recording, status = future.result()
            print(f"[{number}/{len(selected)}] {recording}: {status}", flush=True)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    write_json(manifests / "split.json", split_manifest())
    prepared, exclusions, cases, counts, labels = [], [], set(), Counter(), Counter()
    accounting = []
    temp = manifests / "windows.csv.tmp"
    writer = None
    with temp.open("w", newline="") as stream:
        for recording in all_records:
            meta = verified_cache(root, recording, sums[recording], annotations[recording])
            case = recording.split("/")[0]
            accounting.append({"recording": recording, "case_id": case, "individual_id": individual_id(case),
                               "partition": partition_for(case), "status": meta["status"] if meta else "not_prepared",
                               "duration_seconds": meta["duration_seconds"] if meta else None,
                               "seizure_count": len(annotations[recording]),
                               "seizure_seconds": sum(end - start for start, end in annotations[recording]),
                               "reason": meta.get("reason", "") if meta else "", "source_sha256": sums[recording]})
            if meta is None:
                continue
            if meta["status"] == "excluded":
                exclusions.append(meta)
                continue
            prepared.append({"recording": recording, "cache_sha256": meta["cache_sha256"], "input_fingerprint": meta["input_fingerprint"], "metadata_sha256": sha256(cache_paths(root, recording)[1])})
            cases.add(meta["case_id"])
            for row in window_rows(meta):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                counts[row["partition"]] += 1
                labels[f"{row['partition']}_{row['label']}"] += 1
    if writer is None:
        temp.unlink(missing_ok=True)
        raise ValueError("No eligible windows were prepared")
    temp.replace(manifests / "windows.csv")
    for name, entries in [("recordings.csv", accounting), ("exclusions.csv", [entry for entry in accounting if entry["status"] == "excluded"])]:
        temporary = manifests / (name + ".tmp")
        with temporary.open("w", newline="") as stream:
            report_writer = csv.DictWriter(stream, fieldnames=list(accounting[0]))
            report_writer.writeheader()
            report_writer.writerows(entries)
        temporary.replace(manifests / name)
    complete = len(prepared) + len(exclusions) == len(all_records) and len({individual_id(c) for c in cases}) == 23
    report = {"source_dataset": "CHB-MIT 1.0.0", "source_recordings": len(all_records),
              "source_seizure_recordings": len(seizures), "prepared_recordings": len(prepared),
              "excluded_recordings": len(exclusions), "individuals": sorted({individual_id(c) for c in cases}),
              "excluded_duration_seconds": sum(meta["duration_seconds"] for meta in exclusions),
              "excluded_seizure_count": sum(len(meta["seizures"]) for meta in exclusions),
              "cases": sorted(cases), "window_counts": dict(counts), "label_counts": dict(labels),
              "full_cohort_complete": complete, "preprocessing_version": PREPROCESSING_VERSION,
              "windows_sha256": sha256(manifests / "windows.csv"), "split_sha256": split_manifest()["sha256"],
              "source_inventory_sha256": sha256(source_dir / "RECORDS"),
              "source_checksums_sha256": sha256(source_dir / "SHA256SUMS.txt"),
              "prepared": prepared, "exclusions": exclusions, "annotation_corrections": corrections}
    report["dataset_hash"] = object_hash(report)
    write_json(manifests / "dataset.json", report)
    if full and not complete:
        raise ValueError("Full preparation did not retain all 23 individuals")
    return report


def load_rows(root, partition=None):
    with (Path(root) / "manifests/windows.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ["start_sample", "end_sample", "label", "seizure_overlap_samples"]:
            row[key] = int(row[key])
        for key in ["start_seconds", "end_seconds", "seizure_overlap_seconds"]:
            row[key] = float(row[key])
    return [row for row in rows if partition is None or row["partition"] == partition]


def audit_dataset(root, require_full=False):
    root = Path(root)
    report = read_json(root / "manifests/dataset.json")
    expected_hash = report.pop("dataset_hash")
    if object_hash(report) != expected_hash:
        raise ValueError("Dataset report was modified")
    report["dataset_hash"] = expected_hash
    if sha256(root / "manifests/windows.csv") != report["windows_sha256"]:
        raise ValueError("Window manifest was modified")
    if read_json(root / "manifests/split.json") != split_manifest():
        raise ValueError("Frozen split was modified")
    if require_full and not report["full_cohort_complete"]:
        raise ValueError("Final experiment requires the complete 23-individual cohort")
    for entry in report["prepared"]:
        array, metadata = cache_paths(root, entry["recording"])
        meta = read_json(metadata)
        if sha256(array) != entry["cache_sha256"] or sha256(metadata) != entry["metadata_sha256"] or meta["input_fingerprint"] != entry["input_fingerprint"]:
            raise ValueError(f"Cache modified: {entry['recording']}")
    return report
