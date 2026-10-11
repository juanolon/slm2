from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

ALPHABET = " abcdefghijklmnopqrstuvwxyz"


def encode_ascii_bytes(raw: bytes, alphabet: str = ALPHABET) -> torch.Tensor:
    lookup = np.full(256, 255, dtype=np.uint8)
    for index, char in enumerate(alphabet):
        lookup[ord(char)] = index
    data = lookup[np.frombuffer(raw, dtype=np.uint8)]
    assert not (data == 255).any(), "unexpected character in corpus"
    return torch.from_numpy(data.copy())


def split_stream(ids, train_end, valid_end):
    assert 0 < train_end < valid_end < ids.numel()
    return ids[:train_end], ids[train_end:valid_end], ids[valid_end:]


def gather_windows(ids, starts, length):
    assert length > 0 and ids.numel() >= length
    assert ((starts >= 0) & (starts <= ids.numel() - length)).all()

    offsets = torch.arange(length, device=ids.device)

    # build matrix: [B, 1] + [1, L] = [B, L]
    #             offsets
    #              0  1  2
    # starts  0   [0, 1, 2]
    #         2   [2, 3, 4]
    #         3   [3, 4, 5]
    indices = starts[:, None] + offsets[None, :]
    return ids[indices].long()


def load_text8(directory="data"):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "text8.zip"
    if not archive.exists():
        temporary = root / "text8.zip.part"
        urllib.request.urlretrieve("https://mattmahoney.net/dc/text8.zip", temporary)
        os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as handle:
        assert "text8" in handle.namelist()
        raw = handle.read("text8")
    assert len(raw) == 100000000, "expected original 100M-character text8"
    fingerprint = json.dumps(
        {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "alphabet": ALPHABET,
            "cuts": [90000000, 95000000],
        },
        sort_keys=True,
    )
    train, valid, test = split_stream(encode_ascii_bytes(raw), 90000000, 95000000)
    (root / "fingerprint.json").write_text(fingerprint)
    return train, valid, test, fingerprint


class RandomWindowSource:
    def __init__(self, ids, batch_size, length):
        if (
            ids.device.type != "cpu"
            or ids.ndim != 1
            or len(ids) < length
            or batch_size < 1
        ):
            raise ValueError(
                    "Expected 1D tensor on CPU with length >= window length and batch_size >= 1"
            )
        self.ids, self.batch_size, self.length = ids, batch_size, length

    def next_batch(self):
        starts = torch.randint(
            low=0, high=len(self.ids) - self.length + 1, size=(self.batch_size,)
        )
        return gather_windows(self.ids, starts, self.length)
