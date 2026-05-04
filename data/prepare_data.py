import sentencepiece as spm
from datasets import load_dataset
import torch
import os

#loading tokenizer
sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:200000]")

print("Encoding dataset...")
all_tokens = []

for row in dataset:
    tokens = sp.encode(row["text"])
    all_tokens.extend(tokens)

#converting to tensor
data = torch.tensor(all_tokens, dtype=torch.long)

#save encoded data
os.makedirs("data", exist_ok=True)
torch.save(data, "data/train.pt")

print("Saved tokenized data!")
print("Total tokens:", len(data))
