"""
dataset.py — CausalDataset for mini-gpt.

Identical structure to mini-transformer but accepts pre-split
train/val encoded lists so the two sets stay cleanly separated.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class CausalDataset(Dataset):
    """
    Wraps encoded sentences as (input[:-1], target[1:]) pairs.

    Args:
        encoded_sentences: list of token-index lists (already BOS/EOS-wrapped)
        min_len: skip sequences shorter than this
    """

    def __init__(
        self,
        encoded_sentences : List[List[int]],
        min_len           : int = 3,
    ) -> None:
        self.pairs: List[Tuple[List[int], List[int]]] = []
        for seq in encoded_sentences:
            if len(seq) < min_len + 1:
                continue
            self.pairs.append((seq[:-1], seq[1:]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.pairs[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )

    def __repr__(self) -> str:
        return f"CausalDataset(sentences={len(self.pairs)})"


def collate_fn(
    batch   : List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx : int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads a batch of variable-length sequences."""
    srcs, tgts = zip(*batch)
    max_len    = max(s.size(0) for s in srcs)

    src_pad = torch.full((len(srcs), max_len), pad_idx, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), max_len), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :s.size(0)] = s
        tgt_pad[i, :t.size(0)] = t

    return src_pad, tgt_pad


def make_collate(pad_idx: int):
    def _fn(batch):
        return collate_fn(batch, pad_idx)
    return _fn
