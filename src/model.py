"""
model.py — MiniGPT: a small GPT-style causal language model.

Upgrades over mini-transformer:
  1. Learned positional embeddings  (nn.Embedding instead of sinusoidal)
     GPT-2 uses learned PE. On a longer corpus they outperform sinusoidal
     because the model can tune position representations to the data.
  2. Attention dropout added inside MultiHeadAttention (regularisation).
  3. The same generate() interface as mini-transformer — no changes needed.

Architecture is otherwise identical to mini-transformer:
    Embedding + LearnedPE → N × TransformerBlock → LayerNorm → Linear head
    with causal mask, pre-norm, GELU, weight tying, gradient clipping.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, ff_dim: int = None, dropout: float = 0.1) -> None:
        super().__init__()
        ff_dim = ff_dim or 4 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (same as mini-transformer)."""

    def __init__(
        self,
        emb_dim : int,
        n_heads : int,
        ff_dim  : int   = None,
        dropout : float = 0.1,
    ) -> None:
        super().__init__()
        self.attn  = MultiHeadAttention(emb_dim, n_heads, dropout)
        self.ff    = FeedForward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x    : torch.Tensor,
        mask : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, w = self.attn(self.norm1(x), mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, w


class MiniGPT(nn.Module):
    """
    MiniGPT — small causal language model with learned positional embeddings.

    Args:
        vocab_size (int):   vocabulary size
        emb_dim    (int):   embedding / model dimension
        n_heads    (int):   attention heads per block
        n_layers   (int):   number of stacked TransformerBlocks
        max_len    (int):   maximum sequence length (sets size of learned PE table)
        ff_dim     (int):   feedforward inner dimension  (default 4 × emb_dim)
        dropout    (float): dropout probability
        pad_idx    (int):   padding token index (excluded from loss)
    """

    def __init__(
        self,
        vocab_size : int,
        emb_dim    : int,
        n_heads    : int,
        n_layers   : int   = 4,
        max_len    : int   = 128,
        ff_dim     : int   = None,
        dropout    : float = 0.1,
        pad_idx    : int   = 0,
    ) -> None:
        super().__init__()
        self.pad_idx  = pad_idx
        self.n_layers = n_layers
        self.emb_dim  = emb_dim
        self.max_len  = max_len

        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        # ── Learned positional embeddings (upgrade from sinusoidal) ───────────
        # Each position 0..max_len-1 gets its own trainable vector.
        # The model adjusts these during training to encode whatever positional
        # signal is most useful for predicting the next token.
        self.pos_emb = nn.Embedding(max_len, emb_dim)

        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask: (1, 1, T, T), True = masked."""
        return torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------------------

    def forward(
        self,
        x : torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len) token indices

        Returns:
            logits:      (batch, seq_len, vocab_size)
            all_weights: list[n_layers] of (batch, n_heads, T, T)
        """
        B, T = x.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"

        # ── Token + positional embeddings ─────────────────────────────────────
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.drop(self.tok_emb(x) + self.pos_emb(positions))   # (B, T, D)

        mask        = self._causal_mask(T, x.device)
        all_weights = []

        for block in self.blocks:
            h, w = block(h, mask)
            all_weights.append(w)

        h      = self.norm(h)
        logits = self.head(h)   # (B, T, vocab_size)

        return logits, all_weights

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids     : torch.Tensor,
        max_new_tokens : int   = 20,
        temperature    : float = 1.0,
        top_k          : int   = 0,
        top_p          : float = 0.0,
        greedy         : bool  = False,
        eos_idx        : int   = None,
    ) -> torch.Tensor:
        """
        Autoregressively generates tokens after a prompt.

        New vs mini-transformer: top-p (nucleus) sampling.

        Top-p sampling keeps the smallest set of tokens whose cumulative
        probability exceeds p, then samples from that set.
        More adaptive than top-k: when the model is confident (one token
        dominates), the nucleus is small; when uncertain, it is larger.

        Args:
            prompt_ids:     (1, prompt_len)
            max_new_tokens: tokens to generate
            temperature:    sampling temperature
            top_k:          if >0, restrict to top-k tokens
            top_p:          if >0, nucleus sampling threshold (e.g. 0.9)
            greedy:         always pick argmax
            eos_idx:        stop if this token is generated

        Returns:
            (1, prompt_len + generated_len)
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate context to max_len
            context = ids[:, -self.max_len:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if greedy:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                # Top-k
                if top_k > 0:
                    vals, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < vals[:, -1:]] = float("-inf")

                probs = torch.softmax(next_logits, dim=-1)

                # Top-p (nucleus)
                if top_p > 0.0:
                    sorted_p, sorted_idx = torch.sort(probs, descending=True)
                    cumulative            = torch.cumsum(sorted_p, dim=-1)
                    # Remove tokens beyond the nucleus
                    remove = cumulative - sorted_p > top_p
                    sorted_p[remove] = 0.0
                    sorted_p         = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                    next_id = sorted_idx.gather(
                        -1, torch.multinomial(sorted_p, 1)
                    )
                else:
                    next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)

            if eos_idx is not None and next_id.item() == eos_idx:
                break

        return ids

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params   = sum(p.numel() for p in self.parameters())
        v, d     = self.tok_emb.weight.shape
        return (
            f"MiniGPT(\n"
            f"  vocab={v}, emb_dim={d}, n_layers={self.n_layers},\n"
            f"  {self.blocks[0].attn},\n"
            f"  ff_dim={self.blocks[0].ff.net[0].out_features},\n"
            f"  max_len={self.max_len},\n"
            f"  parameters={params:,}\n"
            f")"
        )
