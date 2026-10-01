from seizure_v2.common import object_hash

SPLITS = {
    "train": ["chb02", "chb03", "chb06", "chb07", "chb10", "chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb20", "chb24"],
    "validation": ["chb05", "chb08", "chb19", "chb23"],
    "test": ["chb01", "chb04", "chb09", "chb21", "chb22"],
}


def individual_id(case_id):
    if case_id not in {f"chb{i:02d}" for i in range(1, 25)}:
        raise ValueError(f"Unknown case: {case_id}")
    return "chb01" if case_id == "chb21" else case_id


def validate_splits(splits, require_full=True):
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, test partitions")
    seen_cases, seen_people = set(), set()
    for partition, cases in splits.items():
        if not cases or len(cases) != len(set(cases)):
            raise ValueError(f"Empty/duplicate cases in {partition}")
        people = {individual_id(case) for case in cases}
        if seen_people & people or seen_cases & set(cases):
            raise ValueError("Patient leakage across partitions")
        seen_cases.update(cases)
        seen_people.update(people)
    if require_full and seen_cases != {f"chb{i:02d}" for i in range(1, 25)}:
        raise ValueError("Final split must contain all 24 cases / 23 individuals")
    return True


def partition_for(case):
    validate_splits(SPLITS)
    return next(partition for partition, cases in SPLITS.items() if case in cases)


def split_manifest():
    return {"seed": 42, "cases": SPLITS, "canonical_map": {case: individual_id(case) for cases in SPLITS.values() for case in cases}, "sha256": object_hash(SPLITS)}
