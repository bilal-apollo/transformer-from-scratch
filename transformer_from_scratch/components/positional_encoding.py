"""Positional Encoding Module."""

import einops
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_from_scratch.components.config import TransformerConfig

from transformer_from_scratch.types import BatchResidualStream, ResidualStream
from transformer_from_scratch.types import TensorShapeLabels as D

PosUnsqueeze = Float[Tensor, f"{D.POSITION} 1"]
DModelHalf = Float[Tensor, f" {D.RESIDUAL_FEATURE_HALF}"]
PosDModelHalf = Float[Tensor, f"{D.POSITION} {D.RESIDUAL_FEATURE_HALF}"]


class SinusoidalPositionalEncoding(torch.nn.Module):
    """Sinusoidal Positional Encoding Module.

    The purpose of positional encoding is to add information about the relative positions of tokens
    within a sequence since the self-attention mechanism does not have any inherent notion of order.
    This is important as a transformer doesn't otherwise inherently have any way of easily way of
    obtaining the order of tokens.

    Reference: https://arxiv.org/pdf/1706.03762.pdf (p6)

    ## Interweaved Sinusoidal Positional Encoding Formula

    The positional encoding is calculated as follows (and then added to the embedded tokens):

    pos = position of a token in the sequence (0, 1, ..., max_tokens - 1)

    i = index of dimension in the embedding space (0, 1, ..., d_model - 1)

    PE(pos, 2i) = sin(pos / (10000 ^ (2i / d_model)))

    PE(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d_model)))

    The resulting positional encoding matrix has a shape of (max_tokens, d_model), where each row
    represents the positional encoding for a specific position in the sequence, and each column
    represents a dimension in the embedding space. The embedded tokens have a shape of (tokens,
    d_model), and the positional encoding matrix is simply added elementwise to get the beginning of
    the residual stream.

    ## Why Sinusoidal?

    The formulas above create a unique positional encoding for each token position by combining
    sinusoidal functions (sine and cosine) with different wavelengths. This works because there is a
    different wavelength for each dimension in the embedding space (d_model), so each embedded token
    will contain the values of its position on lots of different waves. The frequency is controlled
    by the (10000 ^ (2i / d_model) wavelength term.

    ### It's Scalable

    One advantage of this approach is that it's scalable. This is best illustrated with the trivial
    alternative of linear encoding (i.e. PE(pos, i) = pos / max_tokens). This wouldn't scale as well
    to longer sequences because the differences between positions would be smaller.

    ### It's Fixed (Doesn't Require Learning)

    Another advantage is that it's fixed, which is both computationally more efficient than learnt
    positional encodings (assuming both result in equally accurate models, which seems to be the
    case in practice). This is because we don't need to update gradients on each backwards pass.

    ## Why Interweaved?

    ### Relative Positions Can Be Learnt With Tensor Multiplication

    A common thing a model might want to do is have a specific attention head always attend to the
    previous token (or more generally k positions earlier). To do this, it would be nice if PE(pos +
    k) is just a simple linear function of PE(pos).

    We can prove this using the trigonometric angle sum identities:

    sin(A + B) = sin(A)cos(B) + cos(A)sin(B)

    cos(A + B) = cos(A)cos(B) - sin(A)sin(B)

    Let A = pos / 10000^(2i/d_model) and B = k / 10000^(2i/d_model):

    PE(pos + k, 2i) = PE(pos, 2i)cos(B) + PE(pos, 2i + 1)sin(B)

    PE(pos + k, 2i + 1) = PE(pos, 2i + 1)cos(B) - PE(pos, 2i)sin(B)

    Now, looking at the full embeddings for 2 specific tokens (at pos and pos + k), we can represent
    these as vectors:

    v_pos = [PE(pos, 0), PE(pos, 1), PE(pos, 2), ..., PE(pos, d - 1)]^T

    v_pos_k = [PE(pos + k, 0), PE(pos + k, 1), PE(pos + k, 2), ..., PE(pos + k,
    d - 1)]^T

    We can then show that v_pos_k is just a linear multiplication of M and v_pos:

    v_pos_k = M * v_pos

    M = | cos(B_0) sin(B_0) 0 0 0 ... 0 |
        | -sin(B_0) cos(B_0) 0 0 0 ... 0 |
        | 0 0 cos(B_1) sin(B_1) 0 ... 0 |
        | 0 0 -sin(B_1) cos(B_1) 0 ... 0 |
        | ... ... ... ... ... ... ... |
        | 0 0 ... 0 0 cos(B_n) sin(B_n) |
        | 0 0 ... 0 0 -sin(B_n) cos(B_n) |

    This result implies that if an attention head in the first layer multiplies M by the embedding
    of the token at position pos + k, it would obtain a keys vector that closely resembles the
    embedding of the token at position pos, with some "noise" from the actual tokens. If, on the
    query side, it employs an identity matrix, it would obtain a vector that closely matches the one
    from the token at position pos, while producing different vectors for other positions.
    Consequently, the dot product q . k, which forms the attention matrix, would likely have the
    largest value for elements at positions pos and pos + k.

    This is because the dot product of a vector with itself tends to be larger than the dot product
    of two different random vectors of the same size. For instance, when taking the dot product of
    vector with itself, each element is effectively squared before being summed. In this case, a
    square is always non-negative, unlike when considering random vectors where one element might be
    positive and the corresponding one could be negative.

    This example shows how the interweaving of sinusoidal positional encodings enables attention
    heads to learn fixed relative attention patterns.

    Note that this example is a simplified illustration, as attention heads work on linearly
    projected subsets of the embedding space (with dimensions d_head, which is smaller than d_model)
    rather than the entire space. However, the same idea applies, and multiple heads can collaborate
    to achieve the desired behavior across the full embedding space.
    """

    pos_encoding: ResidualStream

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the positional encoding matrix."""
        super().__init__()

        # Create everything inside the parentheses
        # inner = pos/(10000^(2i/d_model) = pos/wavelength
        positions: PosUnsqueeze = torch.arange(0, config.n_ctx).unsqueeze(1).float()
        dimensions_2: DModelHalf = torch.arange(0, config.d_model, 2).float()
        inner: PosDModelHalf = positions / (10000 ** (dimensions_2 / config.d_model))

        # Create interweaved positional encoding
        pos_encoding = torch.zeros(config.n_ctx, config.d_model)
        pos_encoding[:, 0::2] = torch.sin(inner)
        pos_encoding[:, 1::2] = torch.cos(inner)

        # Register as a non-persistent buffer so that it isn't stored in the state dict. This is
        # important as it allows the transformer to be instantiated with a different `max_tokens`
        # value, whilst still re-using the same state dict.
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)

    def forward(self, embedding: BatchResidualStream) -> BatchResidualStream:
        """Apply the positional encoding to the given input embedding.

        Args:
            embedding (ResidualStream): The input embedding with shape (batch_size, tokens,
                d_model).

        Returns:
            ResidualStream: The output embedding with positional encoding applied, having the same
                shape as the input embedding (batch_size, tokens, d_model).
        """
        num_tokens_in_embedding: int = embedding.shape[-2]
        trimmed_pos_encoding: ResidualStream = self.pos_encoding[
            :num_tokens_in_embedding,
            :,
        ]
        return trimmed_pos_encoding + embedding


import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned Absolute Positional Encoding Module.

    This module implements a learned positional encoding, where the positional embeddings
    are parameters that are updated during training, as opposed to being fixed like in
    the sinusoidal positional encoding.

    The learned positional encoding allows the model to potentially learn more flexible
    position representations, which might be beneficial for certain tasks or datasets.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the learned positional encoding."""
        super().__init__()

        # Create a learnable parameter for positional encodings
        self.pos_embedding = nn.Parameter(torch.empty(config.n_ctx, config.d_model))
        nn.init.xavier_normal_(self.pos_embedding)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply the learned positional encoding to the given input embedding.

        Args:
            embedding (torch.Tensor): The input embedding with shape (batch_size, tokens, d_model).

        Returns:
            torch.Tensor: The output embedding with positional encoding applied, having the same
                shape as the input embedding (batch_size, tokens, d_model).
        """
        batch, seq = embedding.shape[:2]
        return einops.repeat(
            self.pos_embedding[:seq, :], "seq d_model -> batch seq d_model", batch=batch
        )
