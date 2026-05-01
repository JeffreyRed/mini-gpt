"""
visualize.py — Training curves, overfitting visualisation, attention heatmaps.

Key new plot: plot_overfitting()
  Shows train and validation perplexity on the same axes.
  The gap between them tells you whether the model is memorising
  the training data (overfitting) or generalising.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

PALETTE = {
    "bg":        "#0d1117",
    "grid":      "#21262d",
    "text":      "#e6edf3",
    "accent":    "#58a6ff",
    "highlight": "#f78166",
    "muted":     "#8b949e",
    "green":     "#3fb950",
    "purple":    "#bc8cff",
    "yellow":    "#e3b341",
}


# ── Train vs Val perplexity — the overfitting plot ────────────────────────────

def plot_overfitting(
    train_history : list,
    val_history   : list,
    save_path     : str = None,
) -> None:
    """
    Plots train and validation perplexity over training.

    The overfitting pattern is clearly visible when:
      - Train perplexity keeps falling
      - Val perplexity flattens or rises
      - The shaded gap between them grows

    Args:
        train_history: list of (loss, perplexity) tuples from train()
        val_history:   list of (loss, perplexity) tuples from train()
        save_path:     optional path to save the figure
    """
    train_ppl = [h[1] for h in train_history]
    val_ppl   = [h[1] for h in val_history]
    epochs    = list(range(1, len(train_ppl) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    # ── Left: perplexity curves ───────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, zorder=0)

    ax.plot(epochs, train_ppl, color=PALETTE["accent"],
            linewidth=2, label="Train perplexity", zorder=3)
    ax.plot(epochs, val_ppl,   color=PALETTE["highlight"],
            linewidth=2, label="Val perplexity",   zorder=3)

    # Shade the overfitting gap
    ax.fill_between(epochs, train_ppl, val_ppl,
                    where=[v > t for t, v in zip(train_ppl, val_ppl)],
                    alpha=0.15, color=PALETTE["highlight"],
                    label="Overfitting gap")

    ax.set_title("Train vs Validation Perplexity",
                 color=PALETTE["text"], fontsize=12, pad=10, loc="left")
    ax.set_xlabel("Epoch", color=PALETTE["muted"])
    ax.set_ylabel("Perplexity  (lower = better)", color=PALETTE["muted"])
    ax.tick_params(colors=PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    legend = ax.legend(facecolor=PALETTE["grid"],
                       labelcolor=PALETTE["text"], fontsize=9)

    # ── Right: loss curves ────────────────────────────────────────────────────
    train_loss = [h[0] for h in train_history]
    val_loss   = [h[0] for h in val_history]

    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"])
    ax2.grid(color=PALETTE["grid"], linewidth=0.5, zorder=0)

    ax2.plot(epochs, train_loss, color=PALETTE["accent"],
             linewidth=2, label="Train loss")
    ax2.plot(epochs, val_loss,   color=PALETTE["highlight"],
             linewidth=2, label="Val loss")
    ax2.fill_between(epochs, train_loss, val_loss,
                     where=[v > t for t, v in zip(train_loss, val_loss)],
                     alpha=0.15, color=PALETTE["highlight"])

    ax2.set_title("Train vs Validation Loss",
                  color=PALETTE["text"], fontsize=12, pad=10, loc="left")
    ax2.set_xlabel("Epoch", color=PALETTE["muted"])
    ax2.set_ylabel("Cross-Entropy Loss", color=PALETTE["muted"])
    ax2.tick_params(colors=PALETTE["muted"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax2.legend(facecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Overfitting plot saved → {save_path}")
    plt.show()


# ── Attention heatmaps ────────────────────────────────────────────────────────

def plot_attention(
    all_weights    : List["torch.Tensor"],
    words          : List[str],
    sentence_label : str = "",
    save_path      : str = None,
) -> None:
    """
    Plots all heads × all layers for one sentence.

    Args:
        all_weights: list[n_layers] of (n_heads, T, T)
        words:       token strings
        sentence_label: shown in title
        save_path:   optional save path
    """
    n_layers = len(all_weights)
    n_heads  = all_weights[0].shape[0]

    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers)
    )
    fig.patch.set_facecolor(PALETTE["bg"])

    if n_layers == 1: axes = [axes] if n_heads > 1 else [[axes]]
    elif n_heads == 1: axes = [[ax] for ax in axes]

    for li, weights in enumerate(all_weights):
        for hi in range(n_heads):
            ax = axes[li][hi]
            w  = weights[hi].numpy()
            ax.set_facecolor(PALETTE["bg"])
            ax.imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha="right",
                               fontsize=7, color=PALETTE["text"],
                               fontfamily="monospace")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=7, color=PALETTE["text"],
                               fontfamily="monospace")
            for i in range(len(words)):
                for j in range(len(words)):
                    v = w[i, j]
                    c = "white" if v > 0.5 else PALETTE["muted"]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5, color=c)
            for sp in ax.spines.values():
                sp.set_edgecolor(PALETTE["grid"])
            ax.set_title(f"L{li+1} H{hi}", color=PALETTE["text"],
                         fontsize=9, pad=5)

    fig.suptitle(f"Attention  ·  \"{sentence_label}\"",
                 color=PALETTE["text"], fontsize=11, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Attention plot saved → {save_path}")
    plt.show()


# ── LR schedule visualisation ─────────────────────────────────────────────────

def plot_lr_schedule(
    total_steps  : int,
    warmup_steps : int,
    peak_lr      : float,
    min_lr       : float = 1e-5,
    save_path    : str   = None,
) -> None:
    """Plots the cosine LR schedule so you can see the warmup + decay shape."""
    import math
    steps = list(range(total_steps))
    lrs   = []
    for s in steps:
        if s < warmup_steps:
            lrs.append(peak_lr * s / max(warmup_steps, 1))
        else:
            progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
            lrs.append(min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress)))

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5)

    ax.plot(steps, lrs, color=PALETTE["green"], linewidth=2)
    ax.axvline(warmup_steps, color=PALETTE["yellow"], linewidth=1,
               linestyle="--", label=f"End of warmup (step {warmup_steps})")
    ax.set_title("Cosine LR Schedule with Linear Warmup",
                 color=PALETTE["text"], fontsize=11, pad=8, loc="left")
    ax.set_xlabel("Optimizer step", color=PALETTE["muted"])
    ax.set_ylabel("Learning rate", color=PALETTE["muted"])
    ax.tick_params(colors=PALETTE["muted"])
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["grid"])
    ax.legend(facecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"LR schedule saved → {save_path}")
    plt.show()


# ── Animated attention ────────────────────────────────────────────────────────

def animate_attention(
    snapshots : list,
    words     : List[str],
    layer     : int = 0,
    head      : int = 0,
    save_path : str = None,
    interval  : int = 200,
) -> None:
    """Animates one head's attention pattern evolving during training."""
    if not snapshots:
        return

    seq_len = len(words)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    im = ax.imshow(np.zeros((seq_len, seq_len)), cmap="Blues",
                   vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(words, rotation=45, ha="right",
                       fontsize=8, color=PALETTE["text"],
                       fontfamily="monospace")
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(words, fontsize=8, color=PALETTE["text"],
                       fontfamily="monospace")
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["grid"])

    title = ax.set_title("", color=PALETTE["text"], fontsize=9, pad=6)

    def update(fi):
        epoch, all_w = snapshots[fi]
        w = all_w[layer]
        if w.dim() == 4: w = w[0]
        data = w[head, :seq_len, :seq_len].numpy()
        im.set_data(data)
        title.set_text(
            f"L{layer+1} H{head}  epoch {epoch}  [{fi+1}/{len(snapshots)}]"
        )
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots),
                                   interval=interval, blit=False)
    if save_path:
        anim.save(save_path, writer="pillow", dpi=100)
        print(f"Animation saved → {save_path}")
    plt.tight_layout()
    plt.show()
