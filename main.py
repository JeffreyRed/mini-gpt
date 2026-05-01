"""
main.py — End-to-end pipeline for mini-gpt.

Usage:
    python main.py

Pipeline:
    1.  Load Einstein corpus, build tokenizer, show train/val split
    2.  Plot cosine LR schedule
    3.  Build MiniGPT with learned positional embeddings
    4.  Train with cosine LR decay, tracking train + val perplexity
    5.  Plot overfitting curves  (the key new visualisation)
    6.  Generate text from the best checkpoint (lowest val perplexity)
    7.  Interactive generation: greedy / temperature / top-k / top-p / beam
    8.  Attention heatmap for one Einstein sentence
    9.  Save best model
"""

import torch
from pathlib import Path

from src.tokenizer  import Tokenizer
from src.dataset    import CausalDataset
from src.model      import MiniGPT
from src.train      import train, evaluate
from src.utils      import (
    generate_text, beam_search,
    interactive_generate, get_attention_weights,
)
from src.visualize  import (
    plot_overfitting, plot_attention,
    plot_lr_schedule, animate_attention,
)

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH    = "data/einstein.txt"
VAL_SPLIT      = 0.2           # 20% of sentences held out for validation
EMB_DIM        = 64            # larger than mini-transformer
N_HEADS        = 4             # 4 heads, head_dim = 16
N_LAYERS       = 4             # 4 stacked blocks
FF_DIM         = 128           # 2 × EMB_DIM
MAX_LEN        = 64            # max sequence length
EPOCHS         = 400
LR             = 3e-3
MIN_LR         = 1e-5
BATCH_SIZE     = 8
WARMUP_STEPS   = 100
OUTPUTS_DIR    = Path("outputs")
INSPECT_PROMPT = "imagination is"
# ──────────────────────────────────────────────────────────────────────────────


def show_split(tok: Tokenizer) -> None:
    """Prints the train/val split so you can see what the model trains on."""
    print("── Train / Val split ───────────────────────────────")
    print(f"  Total sentences : {len(tok.all_sentences)}")
    print(f"  Train           : {len(tok.train_sentences)}  (used for weight updates)")
    print(f"  Val             : {len(tok.val_sentences)}   (never trained on — only evaluated)\n")
    print("  First 3 val sentences (the model will never see these during training):")
    for s in tok.val_sentences[:3]:
        print(f"    {' '.join(s)}")
    print()


def demo_generation(model: MiniGPT, tok: Tokenizer) -> None:
    """Shows a few generation examples before the interactive loop."""
    print("── Generation examples (best checkpoint) ───────────")
    prompts = [
        ("imagination is",         dict(greedy=True)),
        ("life is",                 dict(temperature=0.8, top_k=10)),
        ("knowledge is",            dict(top_p=0.9, temperature=0.9)),
        ("the secret to",           dict(beam_width=3)),
    ]
    for prompt, kwargs in prompts:
        if "beam_width" in kwargs:
            result = beam_search(model, tok, prompt, **kwargs)
            mode   = f"beam={kwargs['beam_width']}"
        else:
            result = generate_text(model, tok, prompt, max_new=12, **kwargs)
            mode   = str(kwargs)
        print(f"  [{mode}]\n  prompt: \"{prompt}\"\n  output: \"{result}\"\n")


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── 1. Tokeniser + split ──────────────────────────────────────────────────
    print("\n── Tokeniser ───────────────────────────────────────")
    tok = Tokenizer(CORPUS_PATH, val_split=VAL_SPLIT)
    print(tok)
    show_split(tok)

    # ── 2. Datasets ───────────────────────────────────────────────────────────
    train_enc, val_enc = tok.encode_split()
    train_ds = CausalDataset(train_enc)
    val_ds   = CausalDataset(val_enc)
    print(f"  {train_ds}  (train)")
    print(f"  {val_ds}    (val)\n")

    # ── 3. LR schedule preview ────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_ds) // BATCH_SIZE)
    total_steps     = EPOCHS * steps_per_epoch
    plot_lr_schedule(
        total_steps, WARMUP_STEPS, LR, MIN_LR,
        save_path=str(OUTPUTS_DIR / "lr_schedule.png"),
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print("── Model ───────────────────────────────────────────")
    model = MiniGPT(
        vocab_size = tok.vocab_size,
        emb_dim    = EMB_DIM,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        max_len    = MAX_LEN,
        ff_dim     = FF_DIM,
        pad_idx    = tok.pad_idx,
    )
    print(model, "\n")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("── Training ────────────────────────────────────────")
    print("  Columns:  train_ppl = training perplexity")
    print("            val_ppl   = validation perplexity  (overfitting shows here)")
    print("            ⚠         = val_ppl > train_ppl + 5  (overfitting detected)\n")

    snapshots: list = []
    train_hist, val_hist, best_model = train(
        model, train_ds, val_ds,
        pad_idx       = tok.pad_idx,
        epochs        = EPOCHS,
        lr            = LR,
        min_lr        = MIN_LR,
        batch_size    = BATCH_SIZE,
        warmup_steps  = WARMUP_STEPS,
        snapshots     = snapshots,
        snapshot_every= max(1, EPOCHS // 30),
    )

    final_train_ppl = train_hist[-1][1]
    final_val_ppl   = val_hist[-1][1]
    best_val_ppl    = min(h[1] for h in val_hist)
    best_epoch      = val_hist.index(min(val_hist, key=lambda x: x[1])) + 1

    print(f"\n  Final  → train_ppl={final_train_ppl:.2f}  val_ppl={final_val_ppl:.2f}")
    print(f"  Best val_ppl={best_val_ppl:.2f} at epoch {best_epoch}")
    if final_val_ppl > final_train_ppl * 1.5:
        print("  ⚠  Overfitting detected: val_ppl significantly exceeds train_ppl")
        print("     This is expected on a 50-sentence corpus — the model memorises training data.")
        print("     On a larger dataset (mini-chat), this gap will be much smaller.\n")
    else:
        print("  ✓  Train and val perplexity are close — reasonable generalisation.\n")

    # ── 6. Overfitting plot ───────────────────────────────────────────────────
    plot_overfitting(
        train_hist, val_hist,
        save_path=str(OUTPUTS_DIR / "overfitting.png"),
    )

    # ── 7. Generation from best checkpoint ───────────────────────────────────
    print("── Generation (best checkpoint) ────────────────────")
    demo_generation(best_model, tok)

    # ── 8. Interactive generation ─────────────────────────────────────────────
    interactive_generate(best_model, tok)

    # ── 9. Attention heatmap ──────────────────────────────────────────────────
    inspect_words   = tok.decode(
        tok.encode(INSPECT_PROMPT.split(), add_special=True),
        strip_special=False,
    )
    inspect_encoded = tok.encode(INSPECT_PROMPT.split(), add_special=True)

    all_w = get_attention_weights(best_model, inspect_encoded)
    plot_attention(
        all_w, inspect_words,
        sentence_label = INSPECT_PROMPT,
        save_path      = str(OUTPUTS_DIR / "attention.png"),
    )

    # ── 10. Save best model ───────────────────────────────────────────────────
    ckpt_path = OUTPUTS_DIR / "minigpt_best.pt"
    torch.save({
        "model_state": best_model.state_dict(),
        "config": {
            "vocab_size": tok.vocab_size,
            "emb_dim":    EMB_DIM,
            "n_heads":    N_HEADS,
            "n_layers":   N_LAYERS,
            "max_len":    MAX_LEN,
            "ff_dim":     FF_DIM,
        },
        "best_val_ppl": best_val_ppl,
        "best_epoch":   best_epoch,
    }, ckpt_path)
    print(f"Best model saved → {ckpt_path}")

    if snapshots:
        animate_attention(
            snapshots, inspect_words,
            layer=0, head=0,
            save_path=str(OUTPUTS_DIR / "attention_animation.gif"),
        )


if __name__ == "__main__":
    main()
