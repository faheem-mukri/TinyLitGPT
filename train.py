"""
Training loop for TinyLitGPT.

This is where the model learns to predict the next token.

The loop does:
1. Get a batch of data
2. Forward pass (predict next tokens)
3. Compute loss (how wrong are predictions?)
4. Backward pass (compute gradients)
5. Update weights (move in direction that reduces loss)

See docs/training.md for detailed explanation.
"""

import torch
from model.gpt import GPT
from data.dataloader import get_batch
import os
import sentencepiece as spm

# ========== Configuration ==========
# These control how training works. Adjust if needed.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

batch_size = 4          # How many sequences per batch (larger = smoother gradients but needs more memory)
block_size = 128        # Context length (how many previous tokens to consider)

# Load pre-trained tokenizer
sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")
vocab_size = sp.get_piece_size()  # 10,000
print(f"Vocabulary size: {vocab_size}")

# Model hyperparameters
d_model = 512           # Embedding dimension (size of vectors)
n_heads = 8             # Number of attention heads
n_layers = 8            # Number of transformer blocks
learning_rate = 3e-4    # How big are the training steps? (smaller = slower but more stable)
max_iters = 5000        # How many training steps?
eval_interval = 100     # Print loss every X steps

# ========== Initialize Model ==========
# Create a fresh model with random weights
model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
model = model.to(device)
model.train()  # Set to training mode (enables dropout, etc.)

# ========== Setup Optimizer ==========
# AdamW: adaptive learning rate optimization
# It automatically adjusts learning rates per parameter
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training on:", device)
print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# ========== Training Loop ==========
# The actual learning happens here

running_loss = 0
saving_interval = 1000  # Save checkpoint every X steps

for step in range(max_iters):

    # Step 1: Get a batch of training data
    # x: previous tokens (context)
    # y: next tokens (targets)
    # Example: if data is [a,b,c,d,e]
    #   x = [a,b,c,d]
    #   y = [b,c,d,e]
    x, y = get_batch(batch_size, block_size, device)

    # Step 2: Forward pass - model makes predictions
    # logits: unnormalized probabilities for each word in vocab
    # loss: how wrong are the predictions?
    logits, loss = model(x, y)

    # Step 3: Zero gradients from previous step
    # (Otherwise gradients accumulate, which we don't want)
    optimizer.zero_grad()

    # Step 4: Backward pass - compute gradients
    # This tells us how to adjust each parameter to reduce loss
    loss.backward()

    # Step 5: Gradient clipping - prevent exploding gradients
    # If gradients are huge (norm > 1.0), scale them down
    # This prevents training instability
    # See docs/training.md Part 5 for explanation
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Step 6: Optimizer step - update weights
    # Move each parameter in the direction that reduces loss
    optimizer.step()

    # Track running loss for reporting
    running_loss += loss.item()

    # ========== Progress Reporting ==========
    if step % eval_interval == 0 and step > 0:

        avg_loss = running_loss / eval_interval
        print(f"Step {step:5d} | Loss: {avg_loss:.4f}")
        running_loss = 0

        # Save checkpoint periodically
        # This lets you resume training if interrupted
        if step % saving_interval == 0 and step > 0:
            torch.save(model.state_dict(), f"checkpoint_{step}.pth")
            print(f"  → Checkpoint saved")

# ========== Save Final Model ==========
# After training, save the weights
torch.save(model.state_dict(), "model.pth")
print("Training complete! Model saved to model.pth")
print("\nNext step: python generate.py")
