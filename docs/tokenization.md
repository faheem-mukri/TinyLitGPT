# Tokenization: From Text to Numbers

Tokenization is the bridge between human language (text) and machine language (numbers). This guide explains what tokenizers do, why they matter, and how our implementation works.

---

## Part 1: Why Do We Need Tokenization?

### The Problem

Neural networks only understand numbers. They can perform mathematical operations on vectors and matrices, but they can't directly process text.

```
Input:    "Hello"
Needed:   [0.5, -0.2, 0.8, ...]  # Vector that neural network understands

Solution: Tokenizer converts text → numbers
```

### The Process

A tokenizer is a function that converts text into a sequence of integers:

```
Input:    "Hello world"
Output:   [25, 102]
```

Each integer (token ID) represents a unit of text. The tokenizer learns a mapping:
- ID 25 → "Hello"
- ID 102 → "world"

Then, an embedding layer converts these IDs into vectors:
- ID 25 → [0.5, -0.2, 0.8, ...] (512-dimensional vector)
- ID 102 → [-0.1, 0.4, -0.3, ...] (512-dimensional vector)

---

## Part 2: Types of Tokenization

There are several ways to split text into tokens. Each has tradeoffs.

### 1. Character-Level Tokenization

Split text into individual characters.

```
Input:  "Hello"
Output: ['H', 'e', 'l', 'l', 'o']

Vocabulary size: ~100-200 (all characters in the language)
```

**Pros:**
- ✅ Tiny vocabulary (no out-of-vocabulary words)
- ✅ Can represent any text (even typos and nonsense)

**Cons:**
- ❌ Very long sequences (one character per token)
- ❌ Hard to learn since "H", "e", "l" have no semantic connection
- ❌ Context window fills up quickly

### 2. Word-Level Tokenization

Split text into words.

```
Input:  "Hello world"
Output: ["Hello", "world"]

Vocabulary size: ~10,000-100,000+ (all words you might see)
```

**Pros:**
- ✅ Short sequences (words are meaningful units)
- ✅ Natural for humans

**Cons:**
- ❌ Large vocabulary (need tokens for "running," "runs," "ran"—different forms)
- ❌ Out-of-vocabulary problem (how do you handle new words?)
- ❌ Rare words get underrepresented in training

### 3. Subword Tokenization (BPE / SentencePiece)

Split into meaningful subwords—a sweet spot between characters and words.

```
Input:  "unbelievable"
Output: ["un", "believ", "able"]

Input:  "running"
Output: ["running"] (common word)

Vocabulary size: ~1,000-50,000 (good balance)
```

**Pros:**
- ✅ Reasonable sequence length
- ✅ Rare words can be decomposed
- ✅ No true out-of-vocabulary (any word can be built from subwords)
- ✅ Morphology captured (variations like "run," "running," "runs" are similar)

**Cons:**
- ⚠️ Need to learn which subwords to use (training step)
- ⚠️ Semantic boundaries can be weird (e.g., "butter" → ["but", "ter"])

---

## Part 3: Byte Pair Encoding (BPE) Intuition

This is what SentencePiece uses under the hood. The idea is elegant.

### How BPE Works

**Step 1: Start with characters**
```
Text: "hello world"
Tokens: ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
Vocabulary: {'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'}
```

**Step 2: Find the most common pair of adjacent tokens**
```
Pairs in text:
- ('h', 'e'): 1 occurrence
- ('e', 'l'): 1 occurrence
- ('l', 'l'): 1 occurrence ← most common!
- ('l', 'o'): 1 occurrence
...

Most common: ('l', 'l')
```

**Step 3: Merge this pair into a new token**
```
Replace: 'll' → create token 'll'
Text becomes: ['h', 'e', 'll', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
Vocabulary: {'h', 'e', 'll', 'o', ' ', 'w', 'r', 'd', 'l'}
```

**Step 4: Repeat many times**
```
Next iteration: Find most common pair again
- ('h', 'e'): 1
- ('e', 'll'): 1
- ('ll', 'o'): 1 ← most common!
...

Merge to 'llo':
Text: ['h', 'e', 'llo', ' ', 'w', 'o', 'r', 'l', 'd']

Keep going...
```

After thousands of iterations, you end up with a vocabulary of useful subwords.

### Why This Works

Common sequences (like "th", "ing", "er") get merged early and become single tokens. Rare sequences don't get merged and stay as character sequences.

Result:
- Common words: Few tokens (e.g., "hello" → 1 token)
- Rare words: More tokens (e.g., "pneumonoultramicroscopicsilicovolcanoconiosis" → many tokens)
- Unknown words: Can still be represented (just more tokens)

---

## Part 4: SentencePiece

SentencePiece is a practical implementation of BPE (plus extras) that we use in this project.

### Installation

```bash
pip install sentencepiece
```

### Training a SentencePiece Tokenizer

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data.txt",              # Text file to learn from
    model_prefix="tokenizer/tiny", # Output file prefix
    vocab_size=10000,              # How many tokens in vocabulary
    model_type="bpe",              # Use BPE algorithm
    character_coverage=1.0,        # Cover all characters
    split_digits=True,             # Separate digits
    byte_fallback=True,            # Fallback for unknown bytes
    normalization_rule_name="identity"  # Don't normalize text
)
```

**This creates two files:**
- `tokenizer/tiny.model` — Binary model file (used for encoding/decoding)
- `tokenizer/tiny.vocab` — Human-readable vocabulary list

### Using a Trained Tokenizer

```python
sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

# Encoding: text → token IDs
tokens = sp.encode("Hello world")
# Output: [123, 456]

# Decoding: token IDs → text
text = sp.decode([123, 456])
# Output: "Hello world"
```

### How SentencePiece Differs from Standard BPE

1. **Treats spaces as tokens**: Preserves structure without explicit markers
2. **Byte fallback**: Can represent any byte sequence (handles corrupted text)
3. **Normalization options**: Can optionally normalize Unicode (we use identity)
4. **Reversible**: Can perfectly reconstruct original text from tokens

### Important Properties

```python
vocab_size = sp.get_piece_size()  # How many tokens (10,000 in our case)
eos_id = sp.eos_id()               # End-of-sequence token ID
unk_id = sp.unk_id()               # Unknown token ID
```

---

## Part 5: Our Dataset and Vocabulary

### What Text Did We Train On?

We trained our tokenizer on ~200,000 samples from:

1. **TinyStories** (100k samples)
   - Short stories written in simple English
   - Example: "Once upon a time, there was a little girl named Lucy..."
   - Good for learning natural language patterns

2. **WikiText-2** (100k samples)
   - Articles from Wikipedia
   - Example: "Machine learning is a subset of artificial intelligence..."
   - More diverse vocabulary and topics

### Vocabulary Statistics

```
Total tokens in vocabulary: 10,000
Token types:
- Common words: "the" (token 25), "and" (token 38), etc.
- Subwords: "ing" (appears in "running", "walking", etc.)
- Characters: Single characters for rare/unknown text
- Special tokens: <eos> (end of sequence), <unk> (unknown)
```

### Why 10,000 Tokens?

**Trade-off:**
- Too small (1,000): Sequences get longer, hard to learn
- Too large (50,000): Vocabulary huge, each token gets less training
- 10,000: Sweet spot for ~50M parameter model

Larger models (like GPT-3) use 50,000+ tokens; smaller models use fewer.

---

## Part 6: Tokenization in Our Code

### Step 1: Training the Tokenizer

File: `tokenizer/train_tokenizer.py`

```python
# Load datasets
tiny = load_dataset("roneneldan/TinyStories", split="train[:100000]")
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100000]")

# Combine text
all_text = tiny_text + wiki_text

# Save to file
with open("tokenizer/data.txt", "w", encoding='utf-8') as f:
    for text in all_text:
        f.write(text + "\n")

# Train tokenizer
spm.SentencePieceTrainer.train(
    input="tokenizer/data.txt",
    model_prefix="tokenizer/tiny",
    vocab_size=10000,
    model_type="bpe",
    ...
)
```

**What this does:**
1. Downloads training data (one-time)
2. Combines all text
3. Learns which subword boundaries make sense
4. Saves the learned model

### Step 2: Preparing Training Data

File: `data/prepare_data.py`

```python
# Load the trained tokenizer
sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

# Load datasets (same as tokenizer training)
tiny = load_dataset("roneneldan/TinyStories", split="train[:20000]")
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10000]")

# Encode all text to token IDs
all_tokens = []
for text in tqdm(all_text):
    tokens = sp.encode(text)  # "Hello world" → [123, 456]
    all_tokens.extend(tokens)

# Save as a PyTorch tensor
data = torch.tensor(all_tokens, dtype=torch.long)
torch.save(data, "data/train.pt")
```

**What this produces:**
```
data/train.pt — PyTorch tensor with ~10-20 million token IDs
Shape: (num_tokens,)  # 1D tensor of all tokens concatenated
Example: [123, 456, 789, 25, 102, ...]
```

### Step 3: Creating Training Batches

File: `data/dataloader.py`

```python
def get_batch(batch_size, block_size, device):
    # Load the pre-tokenized data
    data = torch.load("data/train.pt")
    
    # Generate random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create input sequences (X) and targets (Y)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)
```

**How this works:**

Imagine we have token sequence: `[a, b, c, d, e, f, g, h]`

With `block_size=4`:
```
Random position: 2

X (input):  [c, d, e, f]  # tokens 2-5
Y (target): [d, e, f, g]  # tokens 3-6 (shifted by 1)
```

The model learns: "Given [c, d, e, f], predict [d, e, f, g]"

### Step 4: Using Tokens in the Model

File: `model/gpt.py`

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, ...):
        super().__init__()
        # vocab_size = 10,000 (number of tokens)
        # d_model = 512 (embedding dimension)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # This converts token IDs to vectors:
        # ID 123 → [0.5, -0.2, 0.8, ...]
        ...
```

**In forward pass:**
```python
def forward(self, x, targets=None):
    # x shape: (batch_size, block_size) — token IDs
    # e.g., [[123, 456, 789, 25], [102, 789, 456, 123], ...]
    
    tok_emb = self.token_embedding(x)
    # Output shape: (batch_size, block_size, d_model)
    # [[embedding(123), embedding(456), embedding(789), embedding(25)],
    #  [embedding(102), embedding(789), embedding(456), embedding(123)],
    #  ...]
```

### Step 5: Generating Text

File: `generate.py`

```python
def generate(model, sp, start_text, max_tokens=50):
    # Step 1: Encode start text to token IDs
    tokens = sp.encode(start_text)  # "Once upon a" → [25, 102, 890]
    tokens = torch.tensor(tokens).unsqueeze(0)  # [[25, 102, 890]]
    
    for _ in range(max_tokens):
        # Step 2: Get model predictions
        logits, _ = model(tokens)  # Output: probabilities for next token
        
        # Step 3: Sample next token
        next_token = sample_token(logits)  # Returns token ID (e.g., 123)
        
        # Step 4: Append and continue
        tokens = torch.cat([tokens, next_token], dim=1)
        # Now: [[25, 102, 890, 123]]
    
    # Step 5: Decode tokens back to text
    generated_text = sp.decode(tokens[0].tolist())
    return generated_text
```

---

## Part 7: Token Distribution

### Common Tokens (High Frequency)

When we decode the vocabulary file, we see:

```
Token 0: </s>         (end-of-sentence, special token)
Token 1: <unk>        (unknown, special token)
Token 2: <s>          (sentence start, special token)
Token 3: ▁the         (the word "the" with space prefix)
Token 4: ▁and         (the word "and" with space prefix)
Token 5: ▁to          (the word "to" with space prefix)
...
Token 25: ▁I          (the word "I" with space prefix)
Token 102: ▁world    (the word "world" with space prefix)
Token 456: ▁said     (the word "said" with space prefix)
...
```

Notice the `▁` symbol—that's how SentencePiece represents spaces (so it can identify word boundaries).

### Rare Tokens (Low Frequency)

Towards token ID 9,999, you see:

```
Token 9,234: ▁pneum
Token 9,567: ▁oncodermatology
Token 9,876: ▁Zwitterionic
```

These are rare subwords learned only because they appeared in the training data.

---

## Part 8: Tokenization Decisions and Trade-offs

### Decision 1: Vocabulary Size (10,000)

**Considered alternatives:**

| Size | Pros | Cons |
|------|------|------|
| 1,000 | Simple, fast | Sequences too long, hard to learn |
| 5,000 | Reasonable | Still long sequences |
| **10,000** | **Good balance** | **Right for our model size** |
| 50,000 | Used by GPT-2 | Overkill for 50M model |

**Our choice:** 10,000 is optimal for ~50M parameters. Larger models can use larger vocabularies.

### Decision 2: Model Type (BPE)

**Alternatives:**
- **Character-level**: Too long sequences
- **Word-level**: OOV problem, large vocabulary
- **BPE**: Best balance ✓ (our choice)
- **WordPiece**: Similar to BPE (used by BERT)

### Decision 3: Normalization

We use `normalization_rule_name="identity"` — no text normalization.

**Alternatives:**
- Lowercase everything: Loses case information
- Remove accents: Loses character variety
- Identity (no normalization): Preserve all information ✓ (our choice)

---

## Part 9: Common Tokenization Problems and Solutions

### Problem 1: Out-of-Vocabulary Words

If tokenizer encounters a word it hasn't seen before.

```
Text: "COVID-19"
Trained on: Text before COVID pandemic
Result: Token not in vocabulary
```

**Solution (SentencePiece):**
Decompose into subwords:
```
"COVID-19" → ["CO", "VI", "D", "-", "19"]
```

Every subword is in vocabulary (worst case: character level).

### Problem 2: Inconsistent Tokenization

```
Same word tokenized differently in different contexts?
```

**Solution:** SentencePiece is deterministic:
```
sp.encode("hello") will ALWAYS return the same token IDs
```

### Problem 3: Tokenization Boundary Artifacts

Sometimes meaningful units get split weirdly:

```
"butter" → ["but", "ter"]  # Weird!
"running" → ["runn", "ing"]  # Or: ["running"] (depends on frequency)
```

This is acceptable—the model learns to handle it. Over time, it learns that "but" + "ter" means "butter."

### Problem 4: Context Window Filling

Subword tokenization can use more tokens than word tokenization.

```
"unbelievable" (1 word):
  - Word tokenizer: 1 token
  - BPE tokenizer: 3 tokens ["un", "believ", "able"]
```

With `block_size=128`, you have 128 tokens of context. That's ~20-40 words on average (English words are ~3-4 subwords on average).

---

## Part 10: How Tokenization Affects Model Training

### Effect 1: Vocabulary Size → Model Size

```
vocab_size = 10,000
d_model = 512

Token embedding layer parameters:
vocab_size × d_model = 10,000 × 512 = 5.12M parameters
```

About 10% of our total 50M parameters!

Larger vocabulary = more parameters needed.

### Effect 2: Sequence Length → Memory Usage

```
batch_size = 4
block_size = 128

Memory for one batch: 4 × 128 × 512 (embedding dim) × 4 bytes = 1 MB per layer

More tokens = more memory = slower training
```

Subword tokenization trades off token count for vocabulary size.

### Effect 3: Training Data Quality

The tokenizer is trained on the same text as the model. If you then train on *different* data:

```
Tokenizer trained on: English novels
Model trained on: French texts
→ Suboptimal (tokenizer doesn't know French well)
```

This is why large language models retrain tokenizers when switching domains.

---

## Part 11: Debugging Tokenization

### Inspect Vocabulary

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

# See what a token looks like
piece = sp.id_to_piece(25)  # Returns "▁I"

# See token for a piece
id = sp.piece_to_id("▁the")  # Returns 3

# Print vocabulary
for i in range(10):
    print(f"{i}: {sp.id_to_piece(i)}")
```

### Test Encoding/Decoding

```python
# Test round-trip: text → tokens → text
original = "Hello world"
tokens = sp.encode(original)  # [123, 456]
decoded = sp.decode(tokens)    # "Hello world"

assert original == decoded, "Round trip failed!"
print(f"{original} → {tokens} → {decoded}")
```

### Check Coverage

```python
# Test weird text to see how it tokenizes
test_strings = [
    "Hello",
    "HELLO",  # Different case
    "hello123",  # Mixed with numbers
    "😀",  # Emoji
    "pneumonoultramicroscopicsilicovolcanoconiosis"  # Long word
]

for s in test_strings:
    tokens = sp.encode(s)
    decoded = sp.decode(tokens)
    print(f"{s:50} → {len(tokens):2} tokens → {decoded}")
```

---

## Part 12: Key Takeaways

### Understanding Tokenization

1. **Tokenization** bridges text and numbers
2. **SentencePiece/BPE** learns meaningful subword boundaries
3. **Vocabulary size** is a design choice (tradeoff: size vs. sequence length)
4. **Round-trip** (encode → decode) should be lossless

### Practical Implications

1. **Choose vocabulary size** based on model size (we chose 10,000 for 50M model)
2. **Train on representative data** so tokenizer learns your domain
3. **Check coverage** on your test/generation data
4. **Monitor token distribution** to spot problems

### Common Mistakes

❌ Training tokenizer on different data than model
❌ Using too-large vocabulary for small models
❌ Not checking round-trip fidelity (text → tokens → text)
❌ Assuming tokenizer is perfect (it's just a heuristic)

### The Philosophy

Tokenization is a learned compression algorithm. By learning which subword boundaries maximize compression, it naturally discovers meaningful linguistic units. Magic!

---

## Next Steps

Now that you understand tokenization:
- Try training a tokenizer with different vocabulary sizes and compare
- Analyze the vocabulary distribution
- See how different text domains tokenize differently
- Experiment with SentencePiece parameters
