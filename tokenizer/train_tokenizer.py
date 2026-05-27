"""
Train the SentencePiece tokenizer.

The tokenizer converts text to token IDs that the model understands.

This script:
1. Downloads training data
2. Saves raw text
3. Uses SentencePiece BPE to learn vocabulary
4. Creates tiny.model and tiny.vocab

Run this FIRST before any other scripts.

See docs/tokenization.md for detailed explanation.
"""

from datasets import load_dataset
import sentencepiece as spm
import os

print("Loading datasets...")

# Step 1: Download TinyStories dataset (100k for better coverage)
tiny = load_dataset("roneneldan/TinyStories", split="train[:100000]")

# Step 2: Download WikiText dataset (100k for diversity)
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100000]")

# Step 3: Extract text from both datasets
print("Extracting text...")
tiny_text = [row["text"] for row in tiny]
wiki_text = [t for t in wiki["text"] if len(t.strip()) > 0]

# Step 4: Combine all text
all_text = tiny_text + wiki_text
print(f"Total texts: {len(all_text)}")

# ========== Save Raw Text ==========

print("Saving raw text...")

# Step 5: Create tokenizer directory if needed
os.makedirs("tokenizer", exist_ok=True)

# Step 6: Write all text to a file
# SentencePiece needs raw text to learn from
with open("tokenizer/data.txt", "w", encoding="utf-8") as f:
    for text in all_text:
        f.write(text + "\n")

print(f"Saved to tokenizer/data.txt")

# ========== Train Tokenizer ==========

print("Training tokenizer...")
print("(This learns which subword boundaries make sense)")

# Step 7: Train SentencePiece tokenizer using BPE algorithm
spm.SentencePieceTrainer.train(
    input="tokenizer/data.txt",          # Input text file
    model_prefix="tokenizer/tiny",       # Output file prefix
    vocab_size=10000,                    # 10,000 tokens
    model_type="bpe",                    # Byte-Pair Encoding algorithm
    character_coverage=1.0,              # Cover all characters
    split_digits=True,                   # Separate digits from words
    byte_fallback=True,                  # Can handle any byte
    normalization_rule_name="identity"   # Don't normalize (keep original text)
)

print("✓ Tokenizer training complete!")
print(f"✓ Created: tokenizer/tiny.model (the tokenizer)")
print(f"✓ Created: tokenizer/tiny.vocab (vocabulary list)")
print("\nVocabulary info:")
print(f"  Total tokens: 10,000")
print(f"  Common tokens: the, and, to, etc.")
print(f"  Rare tokens: subword pieces for uncommon words")
print("\nNext step: python data/prepare_data.py")
