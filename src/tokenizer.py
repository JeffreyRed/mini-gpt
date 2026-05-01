"""
tokenizer.py — Word-level tokenizer with train/val split.

Upgrades from mini-transformer:
  - Corpus is split into train (80%) and validation (20%) sets.
    The vocabulary is built from the FULL corpus so both sets share
    the same token space, but the model only trains on the train split.
  - Frequency-filtered vocabulary: words appearing fewer than min_freq
    times are replaced with <UNK>. On a 50-sentence corpus every word
    appears at least once, so this has little effect here — but it is
    the correct practice for any real dataset.

Special tokens:
    <PAD>  0  — padding
    <BOS>  1  — beginning of sentence
    <EOS>  2  — end of sentence
    <UNK>  3  — unknown / rare word
"""

import random
from collections import Counter
from typing import List, Dict, Tuple


class Tokenizer:

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    SPECIAL   = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(
        self,
        path     : str,
        val_split: float = 0.2,
        min_freq : int   = 1,
        seed     : int   = 42,
    ) -> None:
        random.seed(seed)
        with open(path) as f:
            all_sentences = [l.strip().split() for l in f if l.strip()]

        # Shuffle then split
        random.shuffle(all_sentences)
        n_val = max(1, int(len(all_sentences) * val_split))
        self.val_sentences   : List[List[str]] = all_sentences[:n_val]
        self.train_sentences : List[List[str]] = all_sentences[n_val:]
        self.all_sentences   : List[List[str]] = all_sentences

        self._build_vocab(all_sentences, min_freq)

    # ------------------------------------------------------------------

    def _build_vocab(self, sentences: List[List[str]], min_freq: int) -> None:
        counts = Counter(w for s in sentences for w in s)
        words  = sorted(w for w, c in counts.items() if c >= min_freq)
        all_tokens = self.SPECIAL + words
        self.word2idx : Dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.idx2word : Dict[int, str] = {i: t for t, i in self.word2idx.items()}
        self.vocab_size : int  = len(self.word2idx)
        self.pad_idx = self.word2idx[self.PAD_TOKEN]
        self.bos_idx = self.word2idx[self.BOS_TOKEN]
        self.eos_idx = self.word2idx[self.EOS_TOKEN]
        self.unk_idx = self.word2idx[self.UNK_TOKEN]

    # ------------------------------------------------------------------

    def encode(self, sentence: List[str], add_special: bool = True) -> List[int]:
        ids = [self.word2idx.get(w, self.unk_idx) for w in sentence]
        if add_special:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, indices: List[int], strip_special: bool = True) -> List[str]:
        words = [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]
        if strip_special:
            words = [w for w in words if w not in self.SPECIAL]
        return words

    def encode_split(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Returns (train_encoded, val_encoded)."""
        train = [self.encode(s) for s in self.train_sentences]
        val   = [self.encode(s) for s in self.val_sentences]
        return train, val

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Tokenizer(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  train_sentences={len(self.train_sentences)},\n"
            f"  val_sentences={len(self.val_sentences)},\n"
            f"  special={self.SPECIAL}\n"
            f")"
        )
