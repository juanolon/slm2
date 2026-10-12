from __future__ import annotations

import csv
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .data import data_fingerprint, pack_rows

REVISION = "49a5f5effbc997b86769e4ef335ba308144561b1"
BASE_URL = (
    f"https://raw.githubusercontent.com/Black-Phoenix/4x4-Sudoku-Dataset/{REVISION}"
)
PUZZLES_FILE = "4x4_sudoku_unique_puzzles.csv"
SOLUTIONS_FILE = "4x4_sudoku_unique_solution.csv"


def fetch_csv(data_dir="data", filename=PUZZLES_FILE):
    if filename not in (PUZZLES_FILE, SOLUTIONS_FILE):
        raise ValueError("Unknown 4x4 dataset file")
    path = Path(data_dir) / "sudoku4x4" / filename
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            with (
                urlopen(f"{BASE_URL}/{filename}", timeout=60) as source,
                temporary.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def read_rows(path):
    # The 288-row file is tab-separated despite its .csv extension.
    with Path(path).open(newline="") as handle:
        delimiter = "\t" if "\t" in handle.readline() else ","
        handle.seek(0)
        for row in csv.DictReader(handle, delimiter=delimiter):
            question, answer = row["Puzzle"], row["Solution"]
            if (
                len(question) != 16
                or len(answer) != 16
                or set(question) - set("01234")
                or set(answer) - set("1234")
            ):
                raise ValueError("Invalid 4x4 CSV board")
            yield question, answer, 0


def load_sudoku4x4(config, data_dir="data"):
    path = fetch_csv(data_dir)
    # Deduplicate by puzzle BEFORE splitting, never by reference solution:
    # only 288 completed boards exist, so repeated answers are expected.
    unique = {}
    for question, answer, rating in read_rows(path):
        unique.setdefault(question, (question, answer, rating))
    rows = list(unique.values())
    sizes = [config.max_train_split, config.max_validation_split, config.max_test_split]
    if sum(sizes) > len(rows):
        raise ValueError(
            f"Requested {sum(sizes)} puzzles, but only {len(rows)} are available"
        )
    indices = np.random.default_rng(config.seed).permutation(len(rows))
    selected = {}
    start = 0
    for split, size in zip(("train", "validation", "test"), sizes):
        selected[split] = [rows[i] for i in indices[start : start + size]]
        start += size
    data = {split: pack_rows(records) for split, records in selected.items()}
    return data, {
        "config": vars(config),
        "source": f"{BASE_URL}/{PUZZLES_FILE}",
        "unique_puzzles": len(rows),
        "selected": {split: len(records) for split, records in selected.items()},
        "fingerprint": data_fingerprint(data),
    }
