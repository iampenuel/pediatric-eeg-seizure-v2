"""Read file-relative seizure times, never surrogate wall-clock times."""
import re


def reconcile_inventory(seizure_records, checksum_names):
    """Resolve the one source error verified against summary and sidecar inventory."""
    corrected = set(seizure_records)
    wrong, right = "chb07/chb07_18.edf", "chb07/chb07_19.edf"
    corrections = []
    if wrong in corrected and right not in corrected:
        if right + ".seizures" not in checksum_names or wrong + ".seizures" in checksum_names:
            raise ValueError("Cannot verify the known chb07 inventory correction")
        corrected.remove(wrong)
        corrected.add(right)
        corrections.append({"removed": wrong, "added": right, "reason": "chb07 summary and SHA256SUMS sidecar inventory agree on chb07_19; RECORDS-WITH-SEIZURES incorrectly lists chb07_18"})
    return corrected, corrections


def parse_summary(text):
    records = {}
    current = None
    pending = None
    count = None

    def finish():
        if current is not None:
            if pending is not None or count is None or count != len(records[current]):
                raise ValueError(f"Incomplete/inconsistent annotations: {current}")
            intervals = records[current]
            for i, (start, end) in enumerate(intervals):
                if start < 0 or end <= start or (i and start < intervals[i - 1][1]):
                    raise ValueError(f"Invalid/overlapping seizure intervals: {current}")

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            finish()
            current = line.split(":", 1)[1].strip()
            if current in records:
                raise ValueError(f"Duplicate recording: {current}")
            records[current] = []
            pending, count = None, None
        elif line.startswith("Number of Seizures in File:"):
            if current is None or count is not None:
                raise ValueError("Unexpected seizure count")
            count = int(line.rsplit(":", 1)[1])
        elif re.match(r"Seizure(?:\s+\d+)?\s+(Start|End) Time:", line):
            match = re.fullmatch(r"Seizure(?:\s+\d+)?\s+(Start|End) Time:\s*(\d+(?:\.\d+)?)\s+seconds", line)
            if not match or current is None:
                raise ValueError(f"Malformed annotation: {line}")
            kind, value = match.group(1), float(match.group(2))
            if kind == "Start":
                if pending is not None:
                    raise ValueError("Seizure start without preceding end")
                pending = value
            else:
                if pending is None:
                    raise ValueError("Seizure end without start")
                records[current].append([pending, value])
                pending = None
    finish()
    if not records:
        raise ValueError("Summary contains no recordings")
    return records


def annotations_for(recording, summaries, seizure_records):
    case, filename = recording.split("/")
    intervals = summaries.get(filename)
    if intervals is None:
        if case == "chb24" and recording not in seizure_records:
            return []
        raise ValueError(f"Missing summary entry: {recording}")
    if bool(intervals) != (recording in seizure_records):
        raise ValueError(f"Summary/inventory disagreement: {recording}")
    return intervals
