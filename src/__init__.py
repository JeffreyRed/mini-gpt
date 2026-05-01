"""mini-gpt — source package."""
from src.tokenizer  import Tokenizer
from src.attention  import MultiHeadAttention
from src.model      import FeedForward, TransformerBlock, MiniGPT
from src.dataset    import CausalDataset, collate_fn, make_collate
from src.train      import train, evaluate, cosine_lr
from src.utils      import (
    generate_text, beam_search,
    interactive_generate, get_attention_weights,
)
from src.visualize  import (
    plot_overfitting, plot_attention,
    plot_lr_schedule, animate_attention,
)

__all__ = [
    "Tokenizer",
    "MultiHeadAttention",
    "FeedForward", "TransformerBlock", "MiniGPT",
    "CausalDataset", "collate_fn", "make_collate",
    "train", "evaluate", "cosine_lr",
    "generate_text", "beam_search",
    "interactive_generate", "get_attention_weights",
    "plot_overfitting", "plot_attention",
    "plot_lr_schedule", "animate_attention",
]
