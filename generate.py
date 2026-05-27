"""
Text generation using trained TinyLitGPT model.

How it works:
1. Start with a seed phrase (e.g., "Once upon a")
2. Model predicts the most likely next token
3. Add that token to the sequence
4. Repeat until we reach max length or end-of-sequence token

See docs/sampling.md for detailed explanation of sampling strategies.
"""

import torch
import torch.nn.functional as F
import sentencepiece as spm
from model.gpt import GPT

# ========== Configuration ==========

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model architecture (must match what we trained)
d_model = 512
n_heads = 8
n_layers = 8
block_size = 128

# ========== Load Tokenizer ==========
# Need this to convert text <-> token IDs

sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")
vocab_size = sp.get_piece_size()
print(f"Vocabulary size: {vocab_size}")

# ========== Load Trained Model ==========

model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
model.load_state_dict(torch.load("model.pth"))
model = model.to(device)
model.eval()  # Set to evaluation mode (disables dropout, etc.)

# ========== Generation Function ==========

def generate(
    model,
    sp,
    start_text,
    max_tokens=50,
    temperature=0.8,
    top_k=40
):
    """
    Generate text starting from a seed.

    Args:
        model: Trained GPT model
        sp: SentencePiece tokenizer
        start_text: Starting phrase (e.g., "Once upon a")
        max_tokens: How many tokens to generate (larger = longer text)
        temperature: Randomness (lower = more deterministic, higher = more creative)
        top_k: Only sample from top-K most likely tokens (prevents nonsense)

    Returns:
        Generated text as a string

    See docs/sampling.md for parameter tuning tips.
    """

    # Step 1: Encode seed text to token IDs
    tokens = sp.encode(start_text)
    # Convert to torch tensor and add batch dimension
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Step 2: Autoregressive generation loop
    # Keep generating one token at a time until we hit max_tokens or end-of-sequence
    with torch.no_grad():  # Don't track gradients during generation

        for _ in range(max_tokens):

            # Step 2a: Keep only the last block_size tokens (context window)
            # Our model can only attend to the last 128 tokens
            tokens_cond = tokens[:, -block_size:]

            # Step 2b: Get model predictions for next token
            logits, _ = model(tokens_cond)  # Logits shape: (1, seq_len, vocab_size)

            # Step 2c: Extract logits for the LAST token (what comes next?)
            logits = logits[:, -1, :]  # Shape: (1, vocab_size)

            # Step 3: Apply temperature (control randomness)
            # temperature < 1: Make confident predictions more confident (deterministic)
            # temperature > 1: Make all predictions more equally likely (random)
            # See docs/sampling.md Part 3 for explanation
            logits = logits / temperature

            # Step 4: Apply repetition penalty (reduce prob of tokens we already generated)
            # This prevents the model from getting stuck repeating the same words
            # See docs/sampling.md Part 6 for explanation
            for token in tokens[0]:
                logits[0, token] /= 1.35  # Penalize all seen tokens

            # Stronger penalty for recent tokens
            recent_tokens = tokens[0][-10:]
            for token in recent_tokens:
                logits[0, token] /= 1.5

            # Step 5: Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # Shape: (1, vocab_size)

            # Step 6: Top-K filtering (only consider top-k most likely tokens)
            # Prevents sampling weird low-probability tokens
            # See docs/sampling.md Part 4 for explanation
            top_k_probs, top_k_indices = torch.topk(probs, top_k)

            # Renormalize so top-k probabilities sum to 1
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            # Step 7: Sample next token from top-k distribution
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            # Convert from sampled index to actual token ID
            next_token = torch.gather(top_k_indices, -1, next_token)

            # Step 8: Append new token and continue
            tokens = torch.cat((tokens, next_token), dim=-1)

            # Step 9: Stop if we hit end-of-sequence token
            if next_token.item() == sp.eos_id():
                break

    # Step 3: Decode tokens back to text
    return sp.decode(tokens[0].tolist())

# ========== Test Generation ==========

print("\n" + "="*50)
print("GENERATING TEXT")
print("="*50 + "\n")

# Try generating with different seeds
prompts = [
    "Once upon a time",
    "The cat",
    "She walked",
]

for prompt in prompts:
    print(f"Seed: \"{prompt}\"")
    output = generate(
        model,
        sp,
        prompt,
        max_tokens=50,
        temperature=0.8,    # Balanced quality and diversity
        top_k=40            # Good safety margin
    )
    print(f"Output: {output}\n")

# ========== Tuning Tips ==========
print("="*50)
print("TUNING PARAMETERS")
print("="*50)
print("""
Temperature (controls randomness):
  0.5  → Very deterministic, high quality
  0.8  → Good balance (default)
  1.5  → Very creative, sometimes nonsense

Top-K (prevents bad tokens):
  10   → Very safe, less diverse
  40   → Balanced (default)
  100  → More diverse, riskier

max_tokens (length of generation):
  20   → Short stories
  50   → Medium stories (default)
  200  → Long stories

Try modifying these to see different outputs!

See docs/sampling.md for detailed explanation.
""")
