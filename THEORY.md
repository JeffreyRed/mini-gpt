# Theory & Code Walkthrough — mini-gpt

> Step 4 of the mini-LLM series. Prerequisite: [mini-transformer](../mini-transformer).

---

## Table of Contents

1. [What this step adds](#1-what-this-step-adds)
2. [Train / val split — why it matters](#2-train--val-split--why-it-matters)
3. [Perplexity — what it measures](#3-perplexity--what-it-measures)
4. [Reading the overfitting plot](#4-reading-the-overfitting-plot)
5. [Learned positional embeddings](#5-learned-positional-embeddings)
6. [Cosine LR decay](#6-cosine-lr-decay)
7. [AdamW — weight decay](#7-adamw--weight-decay)
8. [Top-p (nucleus) sampling](#8-top-p-nucleus-sampling)
9. [Beam search](#9-beam-search)
10. [Code walkthrough](#10-code-walkthrough)
11. [Full data flow](#11-full-data-flow)
12. [What mini-chat will add](#12-what-mini-chat-will-add)

---

## 1. What this step adds

`mini-transformer` was structurally a complete GPT-style model but trained
on 10 toy sentences. `mini-gpt` keeps the exact same architecture and
adds the practices that turn a toy into a real training run:

| Addition | Where | Why |
|---|---|---|
| Real corpus (Einstein quotes) | `data/einstein.txt` | Actual language patterns |
| Train / val split | `tokenizer.py` | Measure generalisation |
| Validation perplexity | `train.py` | Expose overfitting |
| Cosine LR decay | `train.py` | Better convergence |
| AdamW | `train.py` | Regularisation via weight decay |
| Learned PE | `model.py` | GPT-2 style, tunable |
| Top-p sampling | `model.py` | Better generation quality |
| Beam search | `utils.py` | Deterministic quality generation |
| Best-model checkpointing | `train.py` | Save the generalising model, not the overfit one |

---

## 2. Train / val split — why it matters

Every machine learning model is evaluated on data it was **never trained on**.
This is the only honest way to know whether the model learned something
general or simply memorised the training examples.

```
All data  →  shuffle  →  80% train  +  20% val

Training loop:       uses train split ONLY
  weights updated    ←  loss on train batches

Evaluation (every epoch):
  weights NOT updated  ←  loss measured on val split
```

The vocabulary is built from the full corpus so both splits share the same
token space. Only the weight updates are restricted to the training split.

**Why 80/20?**
A common default. On 50 sentences: 40 train, 10 val. With more data you
can use 90/10 or 95/5. The principle stays the same.

---

## 3. Perplexity — what it measures

```
Perplexity = exp( cross-entropy loss )
```

Cross-entropy loss at position `i` is `-log P(correct token)`.
If the model assigns probability 1.0 to the correct token, loss = 0,
perplexity = 1. Perfect.
If the model assigns equal probability to all `V` tokens (random),
loss = `log V`, perplexity = `V`.

So perplexity measures: **"how many words was the model effectively
choosing between at each step?"**

| Perplexity | Interpretation |
|---|---|
| = vocab_size (~150 here) | Random — model learned nothing |
| = 50 | Model has ruled out two-thirds of the vocabulary at each step |
| = 10 | Model is quite confident, narrows to ~10 candidates |
| = 1 | Perfect — model knows exactly what comes next |

On a 50-sentence corpus the model will overfit toward perplexity ~1 on
train data. Validation perplexity will be higher — and the gap between
them is what the overfitting plot shows.

---

## 4. Reading the overfitting plot

`outputs/overfitting.png` shows four curves on two axes:

![overfitting](outputs/overfitting.png)

```
Left panel — perplexity:

  train_ppl  ──────────────────────────╲  (keeps falling)
                                        ╲
  val_ppl    ─────────────╲____________/ (falls then flattens/rises)
                           ↑
                    overfitting begins here
                    (shaded gap grows)

Right panel — same pattern in cross-entropy loss
```

**Three phases you will see:**

1. **Early training** — both curves fall together. The model is learning
   genuine patterns that generalise.

2. **Inflection point** — val perplexity stops improving even as train
   perplexity continues to fall. The model starts memorising training
   sentences rather than learning language structure.

3. **Overfitting** — train perplexity approaches 1. Val perplexity
   plateaus or slowly rises. The shaded gap between them is the
   overfitting gap.

**Why does this happen on Einstein quotes?**
50 sentences, ~150 unique words, ~40K parameters. The model has more
parameters than it needs to memorise every training sentence. With a
dataset 100× larger (mini-chat), the model cannot memorise everything
and is forced to generalise.

**The best checkpoint** is saved at the epoch where val perplexity was
lowest — this is the most generalisable version of the model.

---

## 5. Learned positional embeddings

`mini-transformer` used fixed sinusoidal positional encoding:

```python
# sinusoidal (fixed, never updated)
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

`mini-gpt` uses a learned table:

```python
# learned (updated by backprop like any other embedding)
self.pos_emb = nn.Embedding(max_len, emb_dim)
```

**In the forward pass:**
```python
positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
h = self.tok_emb(x) + self.pos_emb(positions)              # sum, not concat
```

**Trade-offs:**

| | Sinusoidal | Learned |
|---|---|---|
| Parameters | 0 extra | max_len × emb_dim extra |
| Generalises to longer sequences | ✓ | ✗ (only up to max_len) |
| Can adapt to data | ✗ | ✓ |
| Used in | original transformer | GPT-2, GPT-3, LLaMA |

For a fixed-length corpus like Einstein quotes, learned PE is strictly
better — the model tunes position representations to the actual sentence
length distribution in the data.

---

## 6. Cosine LR decay

`mini-transformer` used a constant LR after warmup.
`mini-gpt` decays the LR following a cosine curve:

```
warmup phase:  lr = peak × (step / warmup_steps)      (linear rise)
cosine phase:  lr = min_lr + 0.5 × (peak - min_lr) × (1 + cos(π × progress))
```

where `progress = (step - warmup_steps) / (total_steps - warmup_steps)`.

```
  peak ──╮
         │╲
         │  ╲
         │    ╲_____
  min_lr │          ─────
         └──────────────────→ steps
         warmup   cosine decay
```

**Why this works better than constant LR:**
- Early in training, large LR steps help escape bad initialisations
- Late in training, small LR steps allow fine-grained convergence
- The cosine shape is smooth — no abrupt LR changes that can destabilise
- GPT-3, LLaMA, and most modern LMs use cosine decay

The LR schedule is plotted in `outputs/lr_schedule.png` so you can see
the warmup + decay shape before training starts.

![lr_schedule.png](outputs/lr_schedule.png)

---

## 7. AdamW — weight decay

`mini-transformer` used `Adam`. `mini-gpt` uses `AdamW`:

```python
optimizer = optim.AdamW(model.parameters(), lr=lr,
                        betas=(0.9, 0.95), weight_decay=0.1)
```

**The difference: weight decay.**

Adam updates each parameter as:
```
θ ← θ - lr × (gradient_term)
```

AdamW adds an explicit L2 regularisation term:
```
θ ← θ - lr × (gradient_term) - lr × weight_decay × θ
```

This shrinks every weight slightly toward zero at each step, regardless
of the gradient. This acts as a regulariser — it prevents any individual
weight from growing very large, which is a common cause of overfitting.

`weight_decay=0.1` is the value used by GPT-3 and most modern LMs.
`betas=(0.9, 0.95)` is also the GPT-3 default (vs Adam's (0.9, 0.999)).

---

## 8. Top-p (nucleus) sampling

`mini-transformer` offered top-k sampling. `mini-gpt` adds top-p:

**Top-k problem:** the cutoff is fixed regardless of the distribution.
If the model is very confident (one token has 0.99 probability), top-k=10
still allows 9 unlikely alternatives. If the model is very uncertain
(many tokens have 0.01 probability), top-k=10 may cut off good options.

**Top-p solution:** keep the smallest set of tokens whose cumulative
probability exceeds `p`, then sample from that set.

```
Sorted probabilities:  [0.60, 0.20, 0.10, 0.05, 0.03, 0.02, ...]
Cumulative:            [0.60, 0.80, 0.90, 0.95, ...]

With p=0.9: keep first 3 tokens (cumulative = 0.90)
            → nucleus = {token_0, token_1, token_2}
            → renormalise and sample
```

When the model is confident, the nucleus is small (maybe 1–2 tokens).
When the model is uncertain, the nucleus is larger. This adapts the
sampling to the model's current state.

**In practice:** `top_p=0.9` with `temperature=0.9` is a common
starting point for high-quality language model output.

---

## 9. Beam search

All previous generation was **greedy** or **sampled**: at each step,
pick one token and move on. Early mistakes compound — a bad first token
locks the generation into a bad path.

**Beam search** keeps `k` candidate sequences in parallel:

```
Step 0  — start with prompt
  beam 1: [BOS, imagination]
  beam 2: [BOS, imagination]      (same start)
  beam 3: [BOS, imagination]

Step 1  — expand each beam, keep top-k by log-probability
  beam 1: [BOS, imagination, is]        log_p = -0.3
  beam 2: [BOS, imagination, was]       log_p = -1.1
  beam 3: [BOS, imagination, and]       log_p = -1.4

Step 2  — expand again
  candidates from beam 1: [... is more], [... is the], [... is not]
  candidates from beam 2: [... was more], ...
  keep top-3 across all candidates

... continue until EOS or max_new_tokens ...

Return the sequence with highest total log_p / length
```

**Length normalisation** (dividing by sequence length) prevents beam
search from always preferring shorter sequences, which have naturally
higher joint probabilities.

**When to use what:**
- Greedy: fastest, deterministic, tends to repeat
- Temperature/top-k/top-p: creative, variable, can be incoherent
- Beam search: best single output, coherent, no randomness

---

## 10. Code walkthrough

### `tokenizer.py`

**Train/val split:**
```python
random.shuffle(all_sentences)
n_val = max(1, int(len(all_sentences) * val_split))
self.val_sentences   = all_sentences[:n_val]
self.train_sentences = all_sentences[n_val:]
```

Shuffle first so the split is random, not the last 20% of the file.
`max(1, ...)` ensures at least one validation sentence even on tiny corpora.

**`<UNK>` token:**
```python
counts = Counter(w for s in sentences for w in s)
words  = sorted(w for w, c in counts.items() if c >= min_freq)
```
Words below `min_freq` are excluded from the vocabulary. At inference,
unknown words map to `unk_idx`. On the Einstein corpus every word appears
at least once so `<UNK>` is rarely used, but it is essential for any
larger dataset.

---

### `model.py`

**Learned positional embedding:**
```python
self.pos_emb = nn.Embedding(max_len, emb_dim)
# ...
positions = torch.arange(T, device=x.device).unsqueeze(0)
h = self.drop(self.tok_emb(x) + self.pos_emb(positions))
```

`torch.arange(T)` produces `[0, 1, 2, ..., T-1]` — the position indices.
The `unsqueeze(0)` adds a batch dimension so it broadcasts over all
sequences in the batch.

**Top-p implementation:**
```python
if top_p > 0.0:
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_p, dim=-1)
    remove = cumulative - sorted_p > top_p   # shift by one to include the boundary token
    sorted_p[remove] = 0.0
    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
    next_id = sorted_idx.gather(-1, torch.multinomial(sorted_p, 1))
```

The `cumulative - sorted_p > top_p` shift is important: it keeps the
token that pushes cumulative over `p` (the boundary token), otherwise
the nucleus could sum to slightly less than `p`.

---

### `train.py`

**Cosine LR:**
```python
def cosine_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

Applied per optimizer step (not per epoch) for smooth decay:
```python
for pg in optimizer.param_groups:
    pg["lr"] = cosine_lr(step, ...)
```

**Best-model checkpointing:**
```python
if val_ppl < best_val_ppl:
    best_val_ppl = val_ppl
    best_model   = copy.deepcopy(model)
```

`copy.deepcopy` creates a completely independent copy of the model.
Without `deepcopy`, `best_model` would be a reference to the same object
and would keep changing as training continues.

**Overfitting warning:**
```python
gap = val_ppl - train_ppl
overfit = "  ⚠ overfitting" if gap > 5 and epoch > 50 else ""
```

Printed live during training so you see it as it happens.

---

### `utils.py`

**Beam search length normalisation:**
```python
candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
```

`x[0]` is the total log-probability of the sequence.
`x[1]` is the token list (length = number of tokens).
Dividing by length prevents the search from preferring short sequences.

---

## 11. Full data flow

```
einstein.txt  (50 quotes)
      │
      ▼  tokenizer.py  (shuffle + split)
  train: 40 sentences   val: 10 sentences
      │
      ▼  encode with BOS/EOS
  e.g. "imagination is more important than knowledge"
  →  [1, 47, 55, 74, 55, 42, 78, 2]
  input:  [1, 47, 55, 74, 55, 42, 78]
  target: [47, 55, 74, 55, 42, 78,  2]
      │
      ▼  MiniGPT.forward()
  tok_emb(input)  +  pos_emb([0,1,2,3,4,5,6])   →  (1, 7, 64)
  4 × TransformerBlock  (causal mask)             →  (1, 7, 64)
  LayerNorm  →  head  →  logits                   →  (1, 7, vocab_size)
      │
      ▼  CrossEntropyLoss(logits, target)
  loss averaged over 7 positions
      │
      ▼  backward  +  clip  +  AdamW step
  all weights updated (including pos_emb, which sinusoidal PE never was)
      │
      ▼  every epoch: evaluate(model, val_dataset)
  val_loss, val_ppl  ←  no gradients, no weight updates
      │
      ▼  if val_ppl < best → copy.deepcopy(model)
```

---

## 12. What mini-chat will add

`mini-gpt` is a **language model**: it continues text.
`mini-chat` will be an **instruction-following model**: it responds to prompts.

The key additions:

| | mini-gpt | mini-chat |
|---|---|---|
| Data format | plain sentences | `[HUMAN]: ... [AI]: ...` pairs |
| Training signal | next-token on quotes | next-token on AI responses only |
| Corpus | 50 Einstein quotes | larger dialogue dataset |
| Generation | continuation | response to a question |
| Evaluation | perplexity | response quality |

The architecture is the same. The data format and training masking change.
This is exactly how GPT-3 was turned into InstructGPT (and then ChatGPT):
the same base model, fine-tuned on instruction-response pairs.

---

*Next: `mini-chat` — prompt-response format, larger dataset, instruction following.*
