from datasets import load_dataset
import sentencepiece as spm
import os

print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:200000]")

#save dataset into a text file
print("Saving dataset to text file...")
with open("tokenizer/data.txt", "w", encoding='utf-8') as f:
    for row in dataset:
        f.write(row["text"] + "\n")

print("Training tokenizer...")

spm.SentencePieceTrainer.train(
    input="tokenizer/data.txt",
    model_prefix = "tokenizer/tiny",
    vocab_size=8000,
    model_type = "bpe",
    character_coverage = 1.0
)

print("Tokenizer training complete.")