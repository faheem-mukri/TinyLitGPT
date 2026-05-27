"""
Prepare training data for the model.

This script:
1. Downloads datasets (TinyStories + WikiText)
2. Extracts and combines all text
3. Tokenizes using SentencePiece
4. Saves as a tensor for fast loading during training

Run this AFTER: tokenizer/train_tokenizer.py

See docs/tokenization.md and docs/theory.md Part 3 for explanation.
"""

import sentencepiece as spm
from datasets import load_dataset
import torch
from tqdm import tqdm
import os

print("Loading tokenizer...")

# Load the pre-trained tokenizer (created by train_tokenizer.py)
sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

print("Loading datasets...")

# Step 1: Download TinyStories dataset
# 20,000 short stories for children
tiny = load_dataset("roneneldan/TinyStories", split="train[:20000]")

# Step 2: Download WikiText dataset
# 10,000 Wikipedia articles for diversity
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10000]")

print("Extracting text...")

# Step 3: Extract text from datasets
tiny_text = [row["text"] for row in tiny]

# Filter out empty entries from WikiText
wiki_text = [t for t in wiki["text"] if len(t.strip()) > 0]

# Step 4: Show dataset stats
print(f"TinyStories samples: {len(tiny_text)}")
print(f"WikiText samples: {len(wiki_text)}")

# Step 5: Combine datasets
all_text = wiki_text + tiny_text
print(f"Total training chunks: {len(all_text)}")

# ========== Tokenization ==========

print("Tokenizing all text...")
print("(This may take a minute or two)")

# Step 6: Convert all text to token IDs
# This is where text becomes numbers that the model understands
all_tokens = []

for text in tqdm(all_text):
    # Tokenize this text using SentencePiece
    # "Hello world" → [25, 102] (two token IDs)
    tokens = sp.encode(text)

    # Add to the big list of all tokens
    all_tokens.extend(tokens)

print(f"Total tokens: {len(all_tokens)}")

# ========== Save ==========

print("Converting to tensor...")

# Step 7: Convert to PyTorch tensor
# This is what the model actually trains on
data = torch.tensor(all_tokens, dtype=torch.long)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

print("Saving tokenized data...")

# Step 8: Save for fast loading during training
torch.save(data, "data/train.pt")

print(f"✓ Saved to data/train.pt")
print(f"  Total tokens: {len(data)}")
print(f"  File size: {len(data) * 8 / 1e6:.1f} MB")
print("\nNext step: python train.py")
