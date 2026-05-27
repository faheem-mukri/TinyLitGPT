# Self-Attention: The Core of Transformers

Attention is the most important mechanism in transformers. This document explains it from the ground up, building intuition before diving into math and code.

---

## Part 1: The Intuition Behind Attention

### A Real-World Example

Imagine you're reading this sentence:

> "The dog ran into the street because it was chasing a car."

To understand what "it" refers to, you:
1. Ask: "What does 'it' refer to?"
2. Look back at previous words: "dog," "street," "car"
3. Decide: "it" most likely refers to "dog" (the subject, the one chasing)
4. Use that connection to understand the sentence

**This is attention.** You're selectively focusing on relevant parts of the input.

### What Attention Does in a Transformer

For each token, the model computes:
- **Query**: "What am I looking for?"
- **Keys**: Labels for each previous token ("I'm a noun", "I'm a verb", etc.)
- **Values**: The actual information in each token

Then it:
1. Compares the query to all keys
2. Finds which keys match the query best
3. Takes a weighted sum of the corresponding values

**Result**: A new representation of the current token, informed by all previous tokens (weighted by relevance).

---

## Part 2: Attention Mathematics

### The Attention Equation

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

Let's break this down step by step.

### Setup

Assume we have:
- **Query vector** Q: shape (seq_len, d_model)
- **Key vectors** K: shape (seq_len, d_model)
- **Value vectors** V: shape (seq_len, d_model)
- **d_model**: embedding dimension (512 in our model)

### Step 1: Compare Query to Keys

```
Q @ K^T
```

Matrix multiply Q with transposed K.

**Shape:** (seq_len, d_model) @ (d_model, seq_len) = (seq_len, seq_len)

**What it computes:** For each position i (query), compute similarity to all positions j (keys) using dot product.

**Result matrix:** similarity[i][j] = how much does query i match key j?

```
Example (seq_len=3):
Similarity matrix =
       0    1    2
   0 [ 0.5  0.2 -0.1]  ← query 0 matches: itself (0.5), token 1 (0.2), token 2 (-0.1)
   1 [ 0.3  0.8  0.1]  ← query 1 matches: token 0 (0.3), itself (0.8), token 2 (0.1)
   2 [-0.2  0.4  0.6]  ← query 2 matches: token 0 (-0.2), token 1 (0.4), itself (0.6)
```

### Step 2: Scale by sqrt(d)

```
Q @ K^T / sqrt(d)
```

Why divide by sqrt(d)?

When d_model is large (e.g., 512), the dot products get very large (ranging from -500 to 500). This makes softmax nearly one-hot (very peaked).

Dividing by sqrt(d_model) ≈ sqrt(512) ≈ 23 brings values to a reasonable range (-20 to 20), so softmax produces smoother probabilities.

**Rule of thumb:** Larger vectors = larger dot products = need to scale down.

### Step 3: Apply Causal Mask

```
masked_logits = logits.masked_fill(mask == 0, -inf)
```

In a language model, token i should only attend to tokens 0 to i (not future tokens).

```
Mask matrix (for seq_len=3):
      0    1    2
   0 [ 1    0    0]  ← token 0 can attend to: itself only
   1 [ 1    1    0]  ← token 1 can attend to: itself and token 0
   2 [ 1    1    1]  ← token 2 can attend to: itself, token 0, token 1

Applying mask:
   Before: [[0.5, 0.2, -0.1], [0.3, 0.8, 0.1], [-0.2, 0.4, 0.6]]
   After:  [[0.5, -∞, -∞], [0.3, 0.8, -∞], [-0.2, 0.4, 0.6]]
```

When we apply softmax to `-∞`, it becomes 0 (no attention to that position).

### Step 4: Softmax to Get Attention Weights

```
attention_weights = softmax(masked_logits, dim=-1)
```

Softmax converts scores to probabilities (sum to 1).

```
Before softmax (raw scores):
[[0.5, -∞, -∞], [0.3, 0.8, -∞], [-0.2, 0.4, 0.6]]

After softmax (probabilities):
[[1.0, 0.0, 0.0], [0.27, 0.73, 0.0], [0.07, 0.37, 0.56]]

Interpretation:
Token 0: attends 100% to itself
Token 1: attends 27% to token 0, 73% to itself
Token 2: attends 7% to token 0, 37% to token 1, 56% to itself
```

### Step 5: Weighted Sum of Values

```
output = attention_weights @ V
```

For each token, combine the value vectors of all previous tokens, weighted by attention.

```
attention_weights @ V
      (seq_len, seq_len)  @  (seq_len, d_model)
                    =  (seq_len, d_model)

Token 0 output: 1.0 * V[0] + 0.0 * V[1] + 0.0 * V[2] = V[0]
Token 1 output: 0.27 * V[0] + 0.73 * V[1] + 0.0 * V[2]
              = weighted combination of tokens 0 and 1
Token 2 output: 0.07 * V[0] + 0.37 * V[1] + 0.56 * V[2]
              = weighted combination of tokens 0, 1, and 2
```

---

## Part 3: Attention in Our Code

Let's map the equation to our implementation.

### File: `model/gpt.py` — Head Class

```python
class Head(nn.Module):
    def __init__(self, head_size, d_model, block_size):
        super().__init__()
        # Parameters for computing Q, K, V
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        
        # Causal mask (lower triangular matrix)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
```

**What's happening:**
- Three linear layers to project input to Q, K, V
- A buffer stores the lower triangular mask for causal masking

**Why `head_size`?** In multi-head attention, we split the embedding into multiple heads:
- d_model = 512
- n_heads = 8
- head_size = 512 / 8 = 64

Each head operates on 64-dimensional vectors independently.

### Forward Pass

```python
def forward(self, x):
    B, T, C = x.shape
    # B = batch size
    # T = sequence length (up to block_size)
    # C = d_model (512)
```

### Computing Q, K, V

```python
k = self.key(x)      # (B, T, d_model) → (B, T, head_size)
q = self.query(x)    # (B, T, d_model) → (B, T, head_size)
```

Each token's embedding is transformed to a query and key vector.

### Computing Similarities

```python
wei = q @ k.transpose(-2, -1) / (C ** 0.5)
```

**Breaking it down:**
```
q:                (B, T, C)
k.transpose:      (B, C, T)
q @ k.transpose:  (B, T, T)
/ (C ** 0.5):     Scale by sqrt(head_size)
```

Result: wei has shape (B, T, T) where wei[b][i][j] = similarity between token i's query and token j's key.

### Applying Causal Mask

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

**What's `self.tril[:T, :T]`?**
- `self.tril` is a (block_size, block_size) lower triangular matrix of 1s
- `self.tril[:T, :T]` extracts the (T, T) submatrix (handles sequences shorter than block_size)
- Where it's 0 (upper triangle), mask to -inf

### Softmax to Attention Weights

```python
wei = F.softmax(wei, dim=-1)
```

Converts (B, T, T) similarity scores to (B, T, T) attention probabilities.

### Weighted Sum of Values

```python
v = self.value(x)     # (B, T, d_model) → (B, T, head_size)
out = wei @ v         # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
```

Result: out has shape (B, T, head_size) where each token's vector is a weighted combination of all previous tokens' value vectors.

---

## Part 4: Multi-Head Attention

One attention head learns one type of relationship. Multiple heads learn different things.

### The Architecture

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_model, block_size):
        super().__init__()
        # Create num_heads independent attention heads
        self.heads = nn.ModuleList(
            [Head(head_size, d_model, block_size) for _ in range(num_heads)]
        )
        # Output projection (combines all heads)
        self.proj = nn.Linear(d_model, d_model)
```

In our model:
- num_heads = 8
- head_size = 512 / 8 = 64

### Forward Pass

```python
def forward(self, x):
    # Run each head independently
    head_outputs = [h(x) for h in self.heads]
    
    # Each output shape: (B, T, head_size)
    # Concatenate along feature dimension
    out = torch.cat(head_outputs, dim=-1)  # (B, T, d_model)
    
    # Project back
    return self.proj(out)
```

### What Each Head Learns

In practice (from what researchers have found):

**Head 1:** Attends to nearby tokens (local context)
```
Token i attends heavily to: i-1, i, i+1
This learns short-range dependencies
```

**Head 2:** Attends to repeated tokens
```
Token i attends to: all occurrences of similar tokens
This learns semantic/syntactic patterns
```

**Head 3:** Attends to long-range dependencies
```
Token i attends to: distant but related tokens
This captures distant context
```

... and so on. Different heads discover different attention patterns through training.

### Why Multi-Head Works Better

Consider: how do we decide what's "relevant" for a prediction?

```
Sentence: "The bank executive sat at the desk"

What's relevant for predicting after "sat"?

- Grammar perspective: "executive" (the subject that sat)
- Semantic perspective: "desk" (where the sitting happens)
- Both are important!
```

One attention head can't learn both simultaneously. Multiple heads each learn to emphasize different relevance criteria.

---

## Part 5: Attention Visualization

### Example: Analyzing Attention for "it"

Text: "The dog ran because it was sleepy."

Imagine after training, when the model processes "it", here's what might happen:

**Token 0 (The):**
Attention: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
→ Only attends to itself (makes sense, nothing before it)

**Token 1 (dog):**
Attention: [0.1, 0.9, 0.0, 0.0, 0.0, 0.0]
→ Mostly itself, small attention to "the"

**Token 5 (it):**
Attention: [0.05, 0.3, 0.1, 0.05, 0.45, 0.05]
→ Strongest attention to token 4 ("ran")?
   Wait, that doesn't match. Let me reconsider...
   Actually: [0.02, 0.6, 0.05, 0.05, 0.2, 0.08]
→ Strong attention to "dog" (token 1)!
   This helps the model know "it" = "dog"

(In reality, this is distributed across 8 heads in more complex ways.)

---

## Part 6: Why Attention is Powerful

### Problem 1: Long-Range Dependencies

RNNs process left-to-right, one token at a time. Information from early tokens gets diluted by the time you reach late tokens.

Attention solves this: token at position 100 can directly attend to token at position 1.

### Problem 2: Parallelization

RNNs are sequential (can't process in parallel).

Attention: All tokens can be processed simultaneously. No waiting for previous steps.

On GPUs with thousands of cores, this is huge for speed.

### Problem 3: Interpretability

Attention weights show what the model is looking at. You can visualize which tokens influenced a prediction.

```
Example: Predicting word after "she":
- Attention to "girl": 0.3
- Attention to "mary": 0.5
- Attention to "happy": 0.1
- Attention to others: 0.1

→ Model is looking at proper nouns (girl, mary)
→ Makes sense for a pronoun's referent
```

---

## Part 7: Attention Design Decisions

### Decision 1: Scaled Dot-Product vs Alternatives

We use: `Attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V`

**Alternatives:**
- **Additive attention** (Bahdanau): `softmax(W [Q|K]) V`
  - More parameters
  - Slightly better in some cases
  - Slower (requires extra computation)
  
- **Multiplicative attention** (our choice):
  - Fewer parameters
  - Faster (just matrix multiply)
  - Works just as well in practice

### Decision 2: Number of Heads

We use 8 heads.

**Alternatives:**
| Heads | Pros | Cons |
|-------|------|------|
| 1 | Simple, fast | Limited expressiveness |
| 4 | Decent diversity | Still limited |
| **8** | **Good diversity** | **Reasonable compute** |
| 16 | More diverse | More parameters |
| 32 | Very diverse | Expensive |

As a rule: n_heads divides d_model evenly. 512/8 = 64 per head works well.

### Decision 3: Head Dimension

head_size = d_model / n_heads = 512 / 8 = 64

Why not: 512 / 16 = 32 per head?
- Smaller heads learn less rich patterns
- But more heads might capture more diversity

This is a hyperparameter you'd tune based on performance and compute constraints.

### Decision 4: Causal Masking

We mask future tokens to prevent cheating during training.

**Alternative (non-causal):**
No masking; token can attend to all tokens including future ones.

**When to use causal:**
- Language modeling (need to predict next token without seeing it)

**When to use non-causal:**
- Bidirectional models (like BERT for classification)
- Machine translation encoder (can see full source)

---

## Part 8: Common Mistakes and Misconceptions

### Mistake 1: "Attention = Looking Back at Important Words"

**Reality:** Attention is more subtle. It's learned patterns that may not correspond to human intuitions about importance.

```
Model might attend to prepositions that help with grammar,
not just content words.
```

### Mistake 2: "Higher Attention Weight = More Important"

**Reality:** Importance is context-dependent. Sometimes a token with low attention weight matters more (e.g., negations "not" are crucial but may get lower attention if rare).

### Mistake 3: "I Can Understand the Model by Looking at Attention"

**Reality:** Attention weights alone don't fully explain the model's decision. The value vectors being weighted matter too.

```
If attention is uniform [0.25, 0.25, 0.25, 0.25]
and values are wildly different,
the output is heavily influenced by specific high-magnitude values.
```

### Mistake 4: "More Heads = Better Performance"

**Reality:** Diminishing returns. After a certain point, more heads add parameters without proportional gains.

Also depends on other factors (head dimension, total model size, training data).

---

## Part 9: Debugging Attention

### Check 1: Attention Pattern Shape

```python
# In forward pass, add print statements
print(f"Q shape: {q.shape}")  # Should be (B, T, head_size)
print(f"Wei shape: {wei.shape}")  # Should be (B, T, T)
print(f"Out shape: {out.shape}")  # Should be (B, T, head_size)
```

### Check 2: Mask Application

```python
# Manually check mask
tril = torch.tril(torch.ones(4, 4))
print(tril)
# Should look like:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

# After masked_fill with -inf, softmax should zero out upper triangle
wei_masked = wei.masked_fill(tril == 0, float('-inf'))
weights = torch.softmax(wei_masked, dim=-1)
print(weights[0, 2, 3:])  # Should be [0, 0] (no attention to future tokens)
```

### Check 3: Attention Distribution

```python
# Check that attention weights sum to 1
print(weights.sum(dim=-1))  # Should be all 1.0 (might be 1.0 ± floating point error)
```

### Check 4: Causal Constraint

```python
# At position i, attention to positions > i should be 0
for i in range(seq_len):
    for j in range(i+1, seq_len):
        assert weights[b, i, j] < 1e-5, f"Attention at {i} to future {j} is {weights[b, i, j]}"
```

---

## Part 10: Attention Variants Worth Knowing

These are alternatives used in other models:

### Flash Attention
Optimized attention computation for GPUs. Same algorithm, faster execution.

### Sparse Attention
Only attend to k nearest neighbors (not all tokens). Reduces memory from O(T^2) to O(k*T).

Useful for very long sequences.

### Linear Attention
Approximate attention to run in O(T) time instead of O(T^2).

Used for very long sequences but less expressive.

### Local Attention
Each token attends only to nearby tokens (e.g., window of size 64).

### Cross-Attention
Query from one sequence, keys/values from another. Used in machine translation (encoder output attended by decoder).

---

## Part 11: Attention Performance Analysis

### Memory Usage

Attention has quadratic complexity:

```
Attention weight matrix: T × T × float32
For T = 1024 (sequence length):
1024 × 1024 × 4 bytes = 4 MB per layer per batch

For T = 2048:
2048 × 2048 × 4 bytes = 16 MB per layer per batch

For T = 32768 (long-range):
32768^2 × 4 bytes = 4 GB per layer per batch!
```

This is why long-context models are expensive to train and run.

**Our model:** T = 128 (block_size)
- Attention matrix: 128 × 128 × 4 bytes ≈ 65 KB per layer
- Very manageable!

### Computation Time

Matrix multiplications:
```
Q @ K^T:    (B, T, d) @ (d, T) ∝ B * T^2 * d
Softmax:    ∝ B * T^2
@ V:        (B, T, T) @ (B, T, d) ∝ B * T^2 * d
```

Total: O(B * T^2 * d)

For T = 128, this is fast. For T = 10,000, this becomes a bottleneck.

---

## Part 12: Key Takeaways

### Understanding Attention

1. **Attention** computes relevance-weighted combinations of values
2. **Query, Key, Value** are learned projections of the input
3. **Softmax** converts similarity scores to probability distributions
4. **Causal mask** ensures tokens only attend to their past and present
5. **Multi-head** attention learns multiple types of relationships

### Practical Implications

1. **Larger models** can have more heads (more diverse learning)
2. **Longer sequences** hit O(T^2) memory wall
3. **Attention patterns** can be visualized and understood
4. **Causal masking** is essential for language modeling
5. **Scaling laws** apply: more computation → better performance

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| NaN in attention | Softmax of all -inf | Check mask logic |
| Uniform attention weights | Model hasn't learned | Increase training |
| Attending to padding | No mask applied | Add attention mask |
| Memory error | Sequence too long | Reduce block_size or batch_size |

---

## Next Steps

Now that you understand attention:
- Run experiments to visualize attention weights
- Modify number of heads and see the effect
- Try attention on different sequence lengths
- Implement custom attention variants
- Read the original Transformer paper: "Attention Is All You Need" (2017)
