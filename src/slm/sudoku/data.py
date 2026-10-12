from __future__ import annotations

import csv
import hashlib
from itertools import islice
from pathlib import Path

import numpy as np
import torch


def encode_board(text: str) -> torch.Tensor:
    """Encode CSV boards: blanks -1, digits zero-indexed."""
    ids = np.frombuffer(text.replace(".", "0").encode("ascii"), dtype=np.uint8)
    return torch.from_numpy(ids.astype(np.int64) - ord("1")).reshape(len(text))


def decode_board(tokens: torch.Tensor) -> str:
    if (
        tokens.shape not in ((81,), (16,))
        or tokens.dtype != torch.long
        or ((tokens < -1) | (tokens > (3 if tokens.numel() == 16 else 8))).any()
    ):
        raise ValueError("Expected [16] or [81] long Sudoku IDs")
    return "".join(
        "." if value == -1 else str(value + 1) for value in tokens.cpu().tolist()
    )


def sudoku_units(boards: torch.Tensor) -> torch.Tensor:
    assert boards.ndim == 2 and boards.shape[1] in (16, 81)
    side = 4 if boards.shape[1] == 16 else 9
    box = 2 if side == 4 else 3
    grid = boards.reshape(-1, side, side)
    columns = grid.transpose(1, 2)
    boxes = grid.reshape(-1, box, box, box, box).permute(0, 1, 3, 2, 4).reshape(-1, side, side)
    return torch.cat((grid, columns, boxes), dim=1)


def valid_units(boards: torch.Tensor) -> torch.Tensor:
    sorted_units = torch.sort(sudoku_units(boards), dim=-1)[0]
    return (sorted_units == torch.arange(sorted_units.shape[-1], device=boards.device)).all(-1)


def fetch_csvs(data_dir="data"):
    root = Path(data_dir)
    paths = [root / f"{split}.csv" for split in ("train", "test")]
    for path in paths:
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id="sapientinc/sudoku-extreme",
                filename=path.name,
                repo_type="dataset",
                revision="main",
                local_dir=root,
            )
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def eligible_rows(path, max_rating):
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            rating = int(row["rating"])
            if rating <= max_rating:
                yield row["question"], row["answer"], rating


def pack_rows(rows):
    """Convert selected rows to tensors without revalidating the dataset."""
    if not rows:
        raise ValueError("Empty Sudoku split; check rating and split sizes")
    return {
        "questions": torch.stack([encode_board(question) for question, _, _ in rows]),
        "answers": torch.stack([encode_board(answer) for _, answer, _ in rows]),
        "ratings": torch.tensor([rating for _, _, rating in rows], dtype=torch.long),
    }


def data_fingerprint(data):
    """Identify selected tensors for checkpoints; never used to assign splits."""
    digest = hashlib.sha256()
    for split in sorted(data):
        for name in ("questions", "answers", "ratings"):
            digest.update(f"{split}/{name}".encode())
            digest.update(data[split][name].contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_sudoku(config, data_dir="data"):
    if config.dataset == "sudoku9x9":
        return load_sudoku9x9(config, data_dir)
    if config.dataset == "sudoku4x4":
        from .data4x4 import load_sudoku4x4

        return load_sudoku4x4(config, data_dir)
    raise ValueError(f"Unknown sudoku dataset: {config.dataset}")


def load_sudoku9x9(config, data_dir="data"):
    train_path, test_path = fetch_csvs(data_dir)
    # Preserve CSV order after rating filtering, just like a fixed stream split.
    rows = list(
        islice(
            eligible_rows(train_path, config.max_rating),
            config.max_train_split + config.max_validation_split,
        )
    )
    selected = {
        "train": rows[: config.max_train_split],
        "validation": rows[config.max_train_split :],
        "test": list(
            islice(eligible_rows(test_path, config.max_rating), config.max_test_split)
        ),
    }
    data = {name: pack_rows(records) for name, records in selected.items()}
    metadata = {
        "config": vars(config),
        "selected": {name: len(records) for name, records in selected.items()},
        "fingerprint": data_fingerprint(data),
    }
    return data, metadata


class RandomBoardSource:
    def __init__(self, split, batch_size):
        questions, answers = split["questions"], split["answers"]
        if (
            questions.device.type != "cpu"
            or answers.device.type != "cpu"
            or questions.ndim != 2
            or questions.shape[1] not in (16, 81)
            or questions.shape != answers.shape
            or len(questions) < 1
            or batch_size < 1
        ):
            raise ValueError(
                "Expected nonempty paired CPU boards and a positive batch size"
            )
        self.questions, self.answers, self.batch_size = questions, answers, batch_size

    def next_batch(self):
        indices = torch.randint(
            low=0, high=len(self.questions), size=(self.batch_size,)
        )
        return self.questions[indices], self.answers[indices]
