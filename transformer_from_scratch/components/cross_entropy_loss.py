"""Cross Entropy Loss."""

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from transformer_from_scratch.types import BatchLogits, BatchTokenIndices
from transformer_from_scratch.types import TensorShapeLabels as D

BatchTargetIndices = Int[Tensor, f"{D.BATCH} {D.POSITION_MINUS_1}"]
BatchTargetIndicesUnsqueeze = Int[Tensor, f"{D.BATCH} {D.POSITION_MINUS_1} ONE"]
BatchLogitsExceptLast = Float[
    Tensor,
    f"{D.BATCH} {D.POSITION_MINUS_1} {D.VOCAB}",
]


def layerwise_cross_entropies(
    inputs: Int[Tensor, "batch seq"], logits: Float[Tensor, "layer batch seq d_vocab"]
):
    """Language Model Cross Entropy Loss

    Loss is calculated as the average negative log probs of the correct tokens.

    https://arxiv.org/pdf/1706.03762.pdf (p8)

    Params:
        Input: Input tokens
        logits: Logits from the forward pass

    Returns:
        Log loss
    """

    layer, batch, seq, d_vocab = logits.shape
    # Targets are inputs except for the first one (which we aren't predicting)
    # Logits except last exclude the last one (which we don't have a target for)
    target: BatchTargetIndices = inputs[:, 1:]
    logits_except_last: BatchLogitsExceptLast = logits[:, :, :-1, :].float()

    logits_except_last = einops.rearrange(
        logits_except_last, "layer batch seq d_vocab -> batch d_vocab layer seq"
    )
    target = einops.repeat(target, "batch seq -> batch layer seq", layer=layer)

    cross_entropies: Float[Tensor, "batch layer seq"] = F.cross_entropy(
        logits_except_last, target, reduction = 'none'
    )
    
    cross_entropies = einops.rearrange(cross_entropies, "batch layer seq -> layer batch seq")
    
    return cross_entropies

    # log_probs: Float[Tensor, "layer batch seq d_vocab"] = F.log_softmax(
    #     logits_except_last,
    #     dim=-1,
    # )

    # # Predicted log probs are the log probs of the correct tokens
    # index: Int[Tensor, "1 batch seq 1"] = target.unsqueeze(-1).unsqueeze(0)
    # predicted_log_probs: Float[Tensor, ""] = log_probs.gather(-1, index)

    # # Cross entropy loss
    # return -predicted_log_probs.mean()
