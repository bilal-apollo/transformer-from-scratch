"""Training Utilities."""
import math
from pathlib import Path
from typing import Optional

import einops
import torch
from jaxtyping import Int, Float
from torch import Tensor, optim
from torch import save as torch_save
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from transformer_from_scratch.components.cross_entropy_loss import layerwise_cross_entropies
from transformer_from_scratch.transformer import Transformer
from transformer_from_scratch.types import BatchLogits, BatchTokenIndices
from transformer_from_scratch.types import TensorShapeLabels as D

TargetIndices = Int[Tensor, f" {D.POSITION_MINUS_1}"]
BatchTargetIndices = Int[Tensor, f"{D.BATCH} {D.POSITION_MINUS_1}"]


def get_default_device() -> torch.device:
    """Get the default device to use.

    Returns:
        torch.device: Device to use.
    """
    if torch.backends.mps.is_built():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def evaluate(
    model: Module,
    test_dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate the model on a test dataloader

    Args:
        model (Transformer): Transformer model
        test_dataloader (DataLoader): Test dataloader
        device (torch.device): Pytorch device

    Returns:
        float: Accuracy (portion of tokens that are correctly predicted)
    """
    total, correct = 0, 0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs: BatchTokenIndices = batch["input_ids"].to(device)
            inputs_except_last: BatchTargetIndices = inputs[:, :-1]
            labels: BatchTargetIndices = inputs[:, 1:]
            outputs = model(inputs_except_last)
            _, predicted = torch.max(outputs.data, -1)
            total += labels.numel()
            correct += (predicted == labels).sum().item()

    return correct / total

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
learning_rate = 6e-4

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def train_loop(
    model: Transformer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int = 1,
    checkpoint_dir: Path = Path(".checkpoints"),
    max_batches: Optional[int] = None,
    ce_weights: Optional[Tensor] = None,
    use_wandb: bool = False,
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
) -> None:
    """Train loop

    Args:
        model: The Transformer model to train.
        train_dataloader: Dataloader for training data.
        test_dataloader: Dataloader for test data.
        epochs (int): Number of epochs to train for. Defaults to 1.
        checkpoint_dir (Path): Directory to save model parameters. Defaults to ".checkpoints".
        device (torch.device): Device to use for training. Defaults to GPU if available, else CPU.
        max_batches (Optional[int]): Maximum number of batches to process. Defaults to None, which
            processes all batches in the dataloader.
    """
    # default weights
    if ce_weights is None:
        ce_weights = torch.ones(model.config.n_layers + 1) / (model.config.n_layers + 1)
    ce_weights = ce_weights.to(device)

    assert torch.allclose(ce_weights.sum(), torch.tensor(1.0))

    if use_wandb:
        wandb.init(project="output_stream")

    # Note that the paper also uses a warmup period of 4000 steps (which has not
    # been done here)
    # , betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # Loop over epochs
    for epoch in range(epochs):
        # Set to training mode
        model.train()
        # Loop over batches
        for batch_index, batch in tqdm(
            enumerate(train_dataloader),
            desc="Batches",
            total=len(train_dataloader),
            position=1,
        ):

            # Check not over max_batches
            if max_batches and batch_index >= max_batches:
                break

            lr = get_lr(batch_index)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Move inputs to the device
            inputs: BatchTokenIndices = batch["input_ids"].to(device)

            # Forward pass
            seq = inputs.shape[1]
            optimizer.zero_grad()
            outputs: Float[Tensor, "layer batch seq d_vocab"] = model(inputs)
            losses = layerwise_cross_entropies(inputs, outputs)
            layerwise_losses = einops.reduce(losses, "layer batch seq -> layer", "mean")
            loss = einops.einsum(ce_weights, layerwise_losses, "layer, layer ->") 

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log to Wandb
            if batch_index % 1 == 0 and use_wandb:
                wandb.log(
                    {
                        "batch": batch_index,
                        "loss": loss.item(),
                        "layerwise_losses": layerwise_losses.tolist(),
                    },
                )

        # Evaluate & log this (accuracy)
        model.eval()
        test_accuracy = evaluate(model, test_dataloader, device)
        if use_wandb:
            wandb.log(
                {"epoch": epoch, "test_accuracy": test_accuracy},
            )

        # Save model parameters
        torch_save(model.state_dict(), checkpoint_dir / f"model_{epoch}.pt")
        torch_save(model.state_dict(), latest_checkpoint)
    wandb.finish()
