"""
Data loading for training.

This module creates training batches from tokenized data.

Key concept:
- We have one long sequence of tokens: [a, b, c, d, e, f, g, h, ...]
- We create pairs:
  - Input (X): [a, b, c, d]
  - Target (Y): [b, c, d, e]
- The model learns: "given [a,b,c,d], predict [b,c,d,e]"

This is called "autoregressive" training.

See docs/training.md Part 3 for explanation.
"""

import torch

# Load pre-tokenized training data (created by data/prepare_data.py)
data = torch.load("data/train.pt")


def get_batch(batch_size, block_size, device):
    """
    Create a training batch.

    Args:
        batch_size: How many sequences in this batch (e.g., 4)
        block_size: Length of each sequence (e.g., 128 tokens)
        device: "cuda" or "cpu"

    Returns:
        x: Input sequences, shape (batch_size, block_size)
        y: Target sequences, shape (batch_size, block_size)

    Example:
        If data = [10, 20, 30, 40, 50, 60, 70, 80]
        and we get positions [0, 4]:
            x = [[10, 20, 30, 40],     (position 0 to 3)
                 [50, 60, 70, 80]]     (position 4 to 7)
            y = [[20, 30, 40, 50],     (position 1 to 4, shifted by 1)
                 [60, 70, 80, ...]]    (position 5 to 8, shifted by 1)
    """

    # Step 1: Choose random starting positions
    # Pick batch_size random positions in the dataset
    # Make sure we don't go past the end (need at least block_size+1 tokens)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Step 2: Extract input sequences
    # For each starting position, take the next block_size tokens
    x = torch.stack([data[i : i + block_size] for i in ix])

    # Step 3: Extract target sequences (shifted by 1)
    # Model learns to predict token i+1 given tokens up to i
    # This is how we turn one sequence into input-target pairs
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    # Step 4: Move to specified device (GPU or CPU)
    return x.to(device), y.to(device)
