# %%
from transformer_from_scratch.examples.gpt2_code_comparison import get_tokenized_datasets
from transformer_from_scratch.train import train_loop
from transformer_from_scratch.components.config import TransformerConfig
from transformer_from_scratch.transformer import Transformer
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% model
config = TransformerConfig(output_stream=True, d_output_stream=768)
model  = Transformer(config) 
model = model.to(device)

# %% data
# Load the dataset
ds = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2")
ds = ds.with_format("torch")
ds = ds["train"]
# %%
# Split the dataset into train and test sets
train_size = int(0.8 * len(ds))
train_indices = list(range(train_size))
test_indices = list(range(train_size, len(ds)))

# Create the train and test dataloaders
train_dataloader = DataLoader(Subset(ds, train_indices), batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(Subset(ds, test_indices), batch_size=4, shuffle=False, num_workers=4)

# %%
train_loop(model, train_dataloader, test_dataloader, device=device, epochs=1, use_wandb=True)


# %%
