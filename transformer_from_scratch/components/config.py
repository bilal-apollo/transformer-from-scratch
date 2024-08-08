from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Transformer Config

    Defaults to GPT2"""

    d_model: int = 1536
    """Number of residual stream features per token."""

    d_vocab: int = 50257
    """Number of tokens in the vocabulary."""

    n_ctx: int = 1024
    """Maximum number of tokens in a sequence."""

    d_head: int = 64
    """Number of head features per token."""

    d_mlp: int = 3072
    """Number of MLP layer features per token."""

    n_heads: int = 12
    """Number of attention heads."""

    n_layers: int = 12
    """Number of layers (attention + mlp = 1 layer) in the architecture."""
    
    output_stream: bool = False
    """Whether to use the output stream"""
    
    d_output_stream: int = 0
    """Dimension of the output stream"""
    
    def __post_init__(self):
        self.d_resid = self.d_model - self.d_output_stream 

