"""Transformer."""
from torch import nn
import torch
from transformer_from_scratch.components.config import TransformerConfig

from transformer_from_scratch.components.embed_unembed import Embed, Unembed
from transformer_from_scratch.components.layer import Layer
from transformer_from_scratch.components.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from transformer_from_scratch.types import (
    BatchLogits,
    BatchResidualStream,
    BatchTokenIndices,
)


class Transformer(nn.Module):
    """Encoder Only Transformer."""

    def __init__(self, config: TransformerConfig = TransformerConfig()) -> None:
        """Initialise the Transformer."""
        super().__init__()

        # Embedding and unembedding
        self.embed = Embed(config)
        self.unembed = Unembed(config)

        # Positional encoding
        self.positional_encoding = LearnedPositionalEncoding(config)

        # Layers
        self.layers = nn.ModuleList([])
        for _layer_idx in range(config.n_layers):
            self.layers.append(Layer(config))

        # Expose config as public attribute
        self.config = config

    def forward(self, tokens: BatchTokenIndices) -> BatchLogits:
        """Forward pass.

        Args:
            tokens (BatchTokenIndices): Input tokens (indices rather than one-hot)

        Returns:
            BatchLogits: Logits representing log probabilities for the tokens
        """
        
        batch, seq = tokens.shape
        
        output = torch.empty(self.config.n_layers+1, batch, seq, self.config.d_vocab).to(tokens.device)
        
        # Embed + positional encoding
        residual_stream: BatchResidualStream = self.embed(tokens)
        residual_stream += self.positional_encoding(residual_stream)
        output[0] = self.unembed(residual_stream)

        # Loop through layers
        for i, layer in enumerate(self.layers):
            residual_stream = layer(residual_stream)
            output[i+1] = self.unembed(residual_stream)
        
        # Unembed and return
        return output
