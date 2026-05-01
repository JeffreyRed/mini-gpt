"""
train.py — Training loop with cosine LR decay and validation perplexity.

Upgrades over mini-transformer:
  1. Cosine LR decay after warmup — the learning rate follows a cosine curve
     from peak LR down to min_lr over the training run. This gives larger
     updates early when the model is far from optimum, and smaller updates
     later when fine adjustments matter. GPT-3 uses cosine decay.

  2. Validation perplexity evaluated every epoch on the held-out split.
     This is what exposes overfitting: train perplexity keeps falling
     but val perplexity stops falling (or rises). The gap between them
     is the overfitting signal.

  3. Best-model checkpointing: the model weights at the epoch with the
     lowest validation perplexity are saved separately. This is the
     standard practice for any real training run.

Cosine LR schedule:
    warmup:  lr rises linearly from 0 to peak over warmup_steps
    decay:   lr follows cosine from peak down to min_lr over remaining steps

    lr(t) = min_lr + 0.5 * (peak - min_lr) * (1 + cos(π * progress))
"""

import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

from src.model   import MiniGPT
from src.dataset import CausalDataset, make_collate


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(
    step         : int,
    warmup_steps : int,
    total_steps  : int,
    peak_lr      : float,
    min_lr       : float = 1e-5,
) -> float:
    """Cosine LR with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model      : MiniGPT,
    dataset    : CausalDataset,
    pad_idx    : int,
    batch_size : int = 8,
) -> Tuple[float, float]:
    """
    Evaluates model on a dataset.

    Returns:
        (mean_loss, perplexity)
    """
    model.eval()
    loader  = DataLoader(dataset, batch_size=batch_size,
                         collate_fn=make_collate(pad_idx))
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total   = 0.0
    n       = 0

    for src, tgt in loader:
        logits, _ = model(src)
        loss = loss_fn(logits.transpose(1, 2), tgt)
        total += loss.item()
        n     += 1

    mean_loss  = total / max(n, 1)
    perplexity = math.exp(min(mean_loss, 30))
    return mean_loss, perplexity


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model         : MiniGPT,
    train_dataset : CausalDataset,
    val_dataset   : CausalDataset,
    pad_idx       : int,
    epochs        : int   = 300,
    lr            : float = 3e-3,
    min_lr        : float = 1e-5,
    batch_size    : int   = 8,
    warmup_steps  : int   = 100,
    verbose       : bool  = True,
    snapshots     : list  = None,
    snapshot_every: int   = 10,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], MiniGPT]:
    """
    Trains MiniGPT and tracks both train and validation perplexity.

    Args:
        model:          MiniGPT instance
        train_dataset:  training split
        val_dataset:    validation split  (never trained on)
        pad_idx:        padding token index
        epochs:         total training epochs
        lr:             peak learning rate
        min_lr:         minimum LR at end of cosine decay
        batch_size:     sequences per gradient step
        warmup_steps:   linear LR warmup duration
        verbose:        print progress every 10 epochs
        snapshots:      list to append (epoch, attn_weights) tuples
        snapshot_every: snapshot frequency in epochs

    Returns:
        train_history:  list of (loss, perplexity) per epoch
        val_history:    list of (loss, perplexity) per epoch
        best_model:     deep copy of model at best validation perplexity
    """
    loader    = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=make_collate(pad_idx),
    )
    loss_fn   = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    total_steps   = epochs * len(loader)
    step          = 0
    best_val_ppl  = float("inf")
    best_model    = None
    train_history : List[Tuple[float, float]] = []
    val_history   : List[Tuple[float, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt in loader:
            step += 1
            current_lr = cosine_lr(step, warmup_steps, total_steps, lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.zero_grad()
            logits, _ = model(src)
            loss = loss_fn(logits.transpose(1, 2), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(loader)
        train_ppl  = math.exp(min(train_loss, 30))
        train_history.append((train_loss, train_ppl))

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_ppl = evaluate(model, val_dataset, pad_idx, batch_size)
        val_history.append((val_loss, val_ppl))

        # Best model checkpoint
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_model   = copy.deepcopy(model)

        # Snapshot for animation
        if snapshots is not None and (epoch % snapshot_every == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                src0, _ = next(iter(loader))
                _, all_w = model(src0[:1])
                snapshots.append((epoch, [w.detach().clone() for w in all_w]))

        if verbose and (epoch % 10 == 0 or epoch == 1):
            gap = val_ppl - train_ppl
            overfit = "  ⚠ overfitting" if gap > 5 and epoch > 50 else ""
            print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"train_ppl={train_ppl:>7.2f}  "
                f"val_ppl={val_ppl:>7.2f}  "
                f"lr={current_lr:.5f}"
                f"{overfit}"
            )

    return train_history, val_history, best_model
