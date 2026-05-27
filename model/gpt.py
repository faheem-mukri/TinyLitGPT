"""
TinyLitGPT Model Architecture

This file implements a GPT-style transformer from scratch.
For detailed explanation, see: docs/attention.md and docs/theory.md

Key components:
- Head: Single attention head (one perspective on what to focus on)
- MultiHeadAttention: Multiple heads in parallel
- FeedForward: Neural network for non-linear transformations
- Block: Attention + FeedForward combined
- GPT: Full model (embeddings + blocks + output layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    Single self-attention head.

    How it works:
    1. Each token asks: "what other tokens should I look at?"
    2. Compute similarity (Query @ Key) for all pairs
    3. Take weighted sum of Values based on similarities

    See docs/attention.md Part 2 for the math.
    """

    def __init__(self, head_size, d_model, block_size):
        super().__init__()
        # Three linear layers: Query, Key, Value
        # These learn to extract relevant info from each token
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        # Causal mask: prevent looking at future tokens (cheating!)
        # Lower triangle = can look, upper triangle = can't look
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        B, T, C = x.shape

        # Step 1: Compute Query and Key
        k = self.key(x)      # Shape: (B, T, head_size)
        q = self.query(x)    # Shape: (B, T, head_size)

        # Step 2: Compute attention weights (similarities)
        # q @ k.T gives a matrix where entry (i,j) = "how much does position i want to look at j?"
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)  # Divide by sqrt(head_size) for stability

        # Step 3: Apply causal mask (no peeking at future!)
        # Set attention to -inf where we can't look, then softmax will make it 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Step 4: Convert to probabilities (which tokens to focus on)
        wei = F.softmax(wei, dim=-1)

        # Step 5: Weighted sum of values (combine information from relevant tokens)
        v = self.value(x)    # Shape: (B, T, head_size)
        out = wei @ v        # Shape: (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel.

    Why multiple heads?
    - Each head learns different patterns (grammar, semantics, long-range, etc.)
    - Combining them gives richer understanding
    - Like having 8 "experts" each looking at different aspects

    See docs/attention.md Part 4 for explanation.
    """

    def __init__(self, num_heads, head_size, d_model, block_size):
        super().__init__()
        # Create num_heads independent attention heads
        # Each processes head_size dimensions
        self.heads = nn.ModuleList(
            [Head(head_size, d_model, block_size) for _ in range(num_heads)]
        )
        # After concatenating all heads, project back to d_model dimensions
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Run each head independently
        head_outputs = [h(x) for h in self.heads]
        # Concatenate results: (B, T, head_size) * 8 heads -> (B, T, d_model)
        out = torch.cat(head_outputs, dim=-1)
        # Project back (mixes information from all heads)
        return self.proj(out)


class FeedForward(nn.Module):
    """
    Simple feed-forward neural network.

    Why do we need this after attention?
    - Attention tells us WHICH tokens to look at
    - FeedForward tells us HOW to transform the combined information
    - Adds non-linearity (model can learn complex patterns)

    Architecture: Linear(512 -> 2048) -> ReLU -> Linear(2048 -> 512)
    Expands then contracts (common pattern in transformers)
    """

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # Expand 4x
            nn.ReLU(),                         # Non-linearity
            nn.Linear(4 * d_model, d_model),   # Contract back
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    One transformer block = Attention + FeedForward.

    Flow:
    1. Self-Attention: Learn which tokens to focus on
    2. Add residual connection: Keep original information
    3. Layer norm: Stabilize training
    4. FeedForward: Transform the combined information
    5. Add residual connection: Again keep original
    6. Layer norm: Again stabilize

    Residual connections (x + something) help gradients flow during training.
    Layer norm (normalize mean and variance) helps training stability.

    See docs/theory.md Part 7-8 for explanation.
    """

    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        head_size = d_model // n_heads
        # Multi-head attention: learn what to look at
        self.sa = MultiHeadAttention(n_heads, head_size, d_model, block_size)
        # Feed-forward: learn how to transform
        self.ff = FeedForward(d_model)
        # Layer normalization: stabilize training
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention branch with residual connection
        # ln1(x) -> attention -> add back original x
        x = x + self.sa(self.ln1(x))
        # Feed-forward branch with residual connection
        # ln2(x) -> feed-forward -> add back original x
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Full GPT model.

    Architecture pipeline:
    1. Tokenize text -> token IDs
    2. token_embedding: Convert IDs to vectors with meaning
    3. position_embedding: Add "you are at position X" info
    4. 8 blocks: Process through attention + feedforward
    5. layer_norm: Stabilize before output
    6. lm_head: Linear layer that outputs vocab probabilities
    7. Loss: Compare to ground truth

    See docs/theory.md Part 3 for full explanation.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()

        # Step 1: Convert token IDs to dense vectors
        # Token 25 -> [0.2, -0.5, 0.1, ...] (512-dimensional)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Step 2: Add position information
        # Position 0 -> [0.8, 0.1, -0.3, ...]
        # Position 1 -> [0.7, 0.2, -0.2, ...]
        # This tells the model WHERE each token is
        self.position_embedding = nn.Embedding(block_size, d_model)

        # Step 3: Stack multiple transformer blocks
        # Each block does: attention (what to look at) + feedforward (how to transform)
        self.blocks = nn.Sequential(
            *[Block(d_model, n_heads, block_size) for _ in range(n_layers)]
        )

        # Step 4: Final normalization before output
        self.ln_f = nn.LayerNorm(d_model)

        # Step 5: Output layer (language modeling head)
        # Takes 512-dimensional vector -> outputs logits for all 10k vocab words
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        """
        Forward pass through the model.

        Args:
            x: Input token IDs, shape (batch_size, seq_len)
            targets: Target token IDs (for training), shape (batch_size, seq_len)

        Returns:
            logits: Predicted logits for next tokens, shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (only if targets provided)
        """
        B, T = x.shape

        # Step 1: Get embeddings (add position info to token info)
        tok_emb = self.token_embedding(x)  # (B, T) -> (B, T, d_model)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos)  # (T,) -> (T, d_model)

        # Combine: each token knows its meaning AND position
        x = tok_emb + pos_emb  # Broadcasting adds pos_emb to each batch item

        # Step 2: Run through all transformer blocks
        # Each block processes the tokens (attention + feedforward)
        x = self.blocks(x)  # (B, T, d_model) -> (B, T, d_model)

        # Step 3: Final layer norm
        x = self.ln_f(x)

        # Step 4: Convert to vocabulary logits (unnormalized probabilities)
        logits = self.lm_head(x)  # (B, T, d_model) -> (B, T, vocab_size)

        # Step 5: Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for loss computation
            B, T, C = logits.shape
            logits = logits.view(B * T, C)      # (B*T, vocab_size)
            targets = targets.view(B * T)       # (B*T,)

            # Cross-entropy loss: how wrong are our predictions?
            # See docs/training.md Part 2 for explanation
            loss = F.cross_entropy(logits, targets)

        return logits, loss