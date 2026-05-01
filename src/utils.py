"""
utils.py — Text generation interface and inspection helpers.

New vs mini-transformer:
  - top-p (nucleus) sampling exposed in the interactive loop
  - beam search (simple, width=3) — produces more coherent output
    by keeping the N best partial sequences at each step instead of
    just one. This is what most production LLMs use for deterministic
    high-quality output.
"""

import torch
import torch.nn.functional as F
from typing import List

from src.model     import MiniGPT
from src.tokenizer import Tokenizer


# ── Text generation ───────────────────────────────────────────────────────────

def generate_text(
    model       : MiniGPT,
    tokenizer   : Tokenizer,
    prompt      : str,
    max_new     : int   = 20,
    temperature : float = 1.0,
    top_k       : int   = 0,
    top_p       : float = 0.0,
    greedy      : bool  = False,
) -> str:
    """Generates text from a string prompt."""
    words      = prompt.strip().split()
    prompt_ids = tokenizer.encode(words, add_special=True)
    src        = torch.tensor([prompt_ids], dtype=torch.long)

    out = model.generate(
        src,
        max_new_tokens = max_new,
        temperature    = temperature,
        top_k          = top_k,
        top_p          = top_p,
        greedy         = greedy,
        eos_idx        = tokenizer.eos_idx,
    )
    return " ".join(tokenizer.decode(out[0].tolist(), strip_special=True))


def beam_search(
    model      : MiniGPT,
    tokenizer  : Tokenizer,
    prompt     : str,
    beam_width : int = 3,
    max_new    : int = 15,
) -> str:
    """
    Beam search generation.

    Keeps `beam_width` candidate sequences at each step, always expanding
    the most probable ones. Returns the single highest-scoring sequence.

    Unlike greedy (beam_width=1), beam search can recover from locally
    suboptimal choices — it explores multiple paths simultaneously.

    Args:
        model:      trained MiniGPT
        tokenizer:  Tokenizer used at training
        prompt:     seed text
        beam_width: number of beams (3 is a good default)
        max_new:    maximum new tokens

    Returns:
        Best generated string.
    """
    model.eval()
    words      = prompt.strip().split()
    prompt_ids = tokenizer.encode(words, add_special=True)

    # Each beam: (log_prob, token_ids_list)
    beams = [(0.0, prompt_ids)]

    for _ in range(max_new):
        candidates = []

        for log_prob, ids in beams:
            if ids[-1] == tokenizer.eos_idx:
                candidates.append((log_prob, ids))
                continue

            src    = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model(src)
            next_log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

            # Expand: take top beam_width continuations
            top_lp, top_ids = torch.topk(next_log_probs, beam_width)
            for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                candidates.append((log_prob + lp, ids + [tid]))

        # Keep best beam_width candidates (normalise by length to avoid length bias)
        candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        beams = candidates[:beam_width]

    best_ids = beams[0][1]
    return " ".join(tokenizer.decode(best_ids, strip_special=True))


# ── Interactive generation loop ───────────────────────────────────────────────

def interactive_generate(model: MiniGPT, tokenizer: Tokenizer) -> None:
    """
    Interactive text generation.

    Flags:
        --greedy          deterministic argmax
        --temp=<float>    sampling temperature  (default 1.0)
        --topk=<int>      top-k sampling
        --topp=<float>    nucleus sampling  (default 0.0 = off)
        --beam=<int>      beam search width  (default 0 = off)
        --n=<int>         tokens to generate  (default 20)
    """
    vocab          = [w for w in tokenizer.word2idx if not w.startswith("<")]
    lower_to_vocab = {w.lower(): w for w in tokenizer.word2idx}

    print("── Text generation ─────────────────────────────────")
    print(f"  Corpus vocabulary ({len(vocab)} words): {vocab[:15]} ...")
    print("  Flags: --greedy  --temp=0.8  --topk=5  --topp=0.9  --beam=3  --n=20")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            raw = input("  Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        parts  = raw.split()
        flags  = {p for p in parts if p.startswith("--")}
        words  = [p for p in parts if not p.startswith("--")]

        greedy = "--greedy" in flags
        temp   = 1.0
        top_k  = 0
        top_p  = 0.0
        beam   = 0
        n_new  = 20

        for f in flags:
            if f.startswith("--temp="): temp  = float(f.split("=")[1])
            if f.startswith("--topk="): top_k = int(f.split("=")[1])
            if f.startswith("--topp="): top_p = float(f.split("=")[1])
            if f.startswith("--beam="): beam  = int(f.split("=")[1])
            if f.startswith("--n="):    n_new = int(f.split("=")[1])

        # Normalise case
        resolved = [lower_to_vocab.get(w.lower(), w) for w in words]
        unknown  = [w for w in resolved if w not in tokenizer.word2idx]
        if unknown:
            print(f"  ✗ Unknown: {unknown}. Try: {vocab[:10]} ...\n")
            continue

        prompt_str = " ".join(resolved)

        if beam > 0:
            result = beam_search(model, tokenizer, prompt_str,
                                 beam_width=beam, max_new=n_new)
            mode = f"beam={beam}"
        else:
            result = generate_text(model, tokenizer, prompt_str,
                                   max_new=n_new, temperature=temp,
                                   top_k=top_k, top_p=top_p, greedy=greedy)
            mode = "greedy" if greedy else f"temp={temp} topk={top_k} topp={top_p}"

        print(f"\n  [{mode}]\n  → {result}\n")

    print("────────────────────────────────────────────────────\n")


# ── Attention inspection ──────────────────────────────────────────────────────

def get_attention_weights(
    model    : MiniGPT,
    sentence : List[int],
) -> List[torch.Tensor]:
    """Returns (n_heads, T, T) attention weights for every layer."""
    model.eval()
    x = torch.tensor(sentence).unsqueeze(0)
    with torch.no_grad():
        _, all_w = model(x)
    return [w.squeeze(0) for w in all_w]
