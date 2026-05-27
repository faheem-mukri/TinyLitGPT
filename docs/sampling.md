# Sampling and Generation: Turning Predictions into Text

This document explains how to generate text from a trained model. It covers sampling strategies, how to control generation quality, and practical techniques.

---

## Part 1: From Logits to Generated Text

### The Generation Task

We have a trained model that can predict: "Given these tokens, what's the next token?"

We want to use this for generation: "Generate a full story starting from a seed."

### The Autoregressive Process

```
Seed: "Once upon a"

Step 1: Model predicts next token after "Once upon a"
        Output: token_ID_1234 (represents "time")
        Sequence: "Once upon a time"

Step 2: Model predicts next token after "Once upon a time"
        Output: token_ID_5678 (represents ",")
        Sequence: "Once upon a time,"

Step 3: Model predicts next token after "Once upon a time,"
        Output: token_ID_8901 (represents "there")
        Sequence: "Once upon a time, there"

... repeat until:
- Reach max_tokens (user-specified limit)
- Generate end-of-sequence token
```

Each prediction becomes input for the next prediction. This is **autoregressive generation**.

### File: `generate.py`

```python
def generate(model, sp, start_text, max_tokens=50, temperature=0.8, top_k=40):
    # Encode seed text
    tokens = sp.encode(start_text)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    for _ in range(max_tokens):
        # Get model prediction
        logits, _ = model(tokens[:, -block_size:])
        logits = logits[:, -1, :]  # Last token's logits
        
        # Apply sampling strategy (discussed below)
        next_token = sample_next_token(logits, temperature, top_k)
        
        # Append and continue
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if next_token.item() == sp.eos_id():
            break
    
    return sp.decode(tokens[0].tolist())
```

---

## Part 2: Why Sampling (vs. Greedy)?

### Greedy Decoding: Always Pick the Best

Take the token with highest probability:

```python
next_token = torch.argmax(logits, dim=-1)
```

**Example:**
```
Model outputs: [0.7, 0.2, 0.05, 0.05]  (probabilities for different words)
Greedy picks: token 0 (probability 0.7)
```

### Problem with Greedy

```
Sentence: "I went to the beach and"

Greedy prediction after "and":
→ "I" (model thinks this is most likely)
→ "went" (continues repeating)
Result: "I went to the beach and I went to the beach and I went..."

Repetitive! Boring!
```

**Why does this happen?**
- Common words like "I", "went" get high probability
- Model defaults to safe, common choices
- Lacks diversity and creativity

### Sampling: Add Randomness

Instead of always picking the best, randomly sample according to probabilities:

```python
next_token = torch.multinomial(probs, num_samples=1)
```

**Example:**
```
Probabilities: [0.7, 0.2, 0.05, 0.05]
Sample 1: token 0 (70% chance)
Sample 2: token 1 (20% chance)
Sample 3: token 2 (5% chance)
Sample 4: token 0 (70% chance)
```

**Result:** Diverse outputs! Different runs produce different text.

---

## Part 3: Temperature: Controlling Randomness

### What is Temperature?

Temperature modifies logits to change how "sharp" (confident) vs. "smooth" (uncertain) the probabilities are.

```
logits = logits / temperature
probs = softmax(logits)
```

### Temperature = 1.0 (Neutral)

```python
logits = [2.0, 1.0, 0.5, 0.1]
probs = softmax(logits) = [0.64, 0.23, 0.10, 0.03]

Result: Model's native confidence
```

### Temperature < 1.0 (Cold, More Confident)

Example: temperature = 0.5

```python
logits = [2.0, 1.0, 0.5, 0.1] / 0.5 = [4.0, 2.0, 1.0, 0.2]
probs = softmax(logits) = [0.88, 0.09, 0.02, 0.01]

Result: Sharp distribution! Model is very confident about token 0.
        Sampling is less random (often picks highest probability).
```

**Effect:** Deterministic generation, focuses on likely tokens.

**Use when:** You want consistent, high-quality output (storytelling, factual generation).

### Temperature > 1.0 (Hot, More Uncertain)

Example: temperature = 2.0

```python
logits = [2.0, 1.0, 0.5, 0.1] / 2.0 = [1.0, 0.5, 0.25, 0.05]
probs = softmax(logits) = [0.42, 0.26, 0.20, 0.12]

Result: Flat distribution! All tokens more equally likely.
        Sampling is very random (any token can be picked).
```

**Effect:** Diverse, creative generation, but less coherent.

**Use when:** You want variety and creative outputs (but quality may suffer).

### Temperature = 0 (Greedy)

```python
logits = [2.0, 1.0, 0.5, 0.1] / 0.001 ≈ [very large differences]
probs = [1.0, 0.0, 0.0, 0.0]

Result: Always pick highest probability token.
```

### Practical Guidelines

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | Greedy (deterministic) | Testing, reproducibility |
| 0.5 | Cold (confident) | High-quality output, stories |
| 0.8 | Neutral (default) | Balanced quality and diversity |
| 1.0 | Neutral (default) | Balanced quality and diversity |
| 1.5 | Warm (creative) | Brainstorming, creative writing |
| 2.0 | Hot (random) | Maximum diversity (low coherence) |

Our code uses **temperature = 0.8** by default—a reasonable balance.

---

## Part 4: Top-K Sampling

### The Problem with Temperature Alone

Even at low temperature, the model assigns *some* probability to terrible tokens.

```
Temperature = 0.5

Probabilities: [0.88, 0.09, 0.02, 0.01]

Sampling might pick:
- Token 0 (88%): Good!
- Token 1 (9%): Decent
- Token 3 (1%): Rarely, but still possible

That 1% token might be completely nonsensical.
```

### The Solution: Top-K Sampling

Only sample from the K most likely tokens. Ignore the rest.

```python
top_k = 40

# Get top 40 tokens by probability
top_k_probs, top_k_indices = torch.topk(probs, top_k)

# Renormalize so they sum to 1
top_k_probs = top_k_probs / top_k_probs.sum()

# Sample from top K
next_token = torch.multinomial(top_k_probs, num_samples=1)
next_token = torch.gather(top_k_indices, -1, next_token)
```

### Effect

```
Before top-k:  [0.50, 0.30, 0.12, 0.05, 0.02, 0.001, ...]  (10,000 tokens total)
               ↓
After top-k=5: [0.50, 0.30, 0.12, 0.05, 0.02]  (renormalized)
               ↓
Sampling from these 5 tokens only.
```

**Advantages:**
- ✅ Avoids horrible tokens
- ✅ More coherent output
- ✅ Still diverse (top 5-50 tokens)
- ✅ Controllable via K

**Our code uses top_k=40:**

```python
top_k_probs, top_k_indices = torch.topk(probs, top_k)
```

---

## Part 5: Top-P (Nucleus) Sampling

### Alternative: Probability Mass Threshold

Instead of picking top-K tokens, pick tokens until cumulative probability reaches P.

```python
# Sort probabilities in descending order
sorted_probs = torch.sort(probs, descending=True)

# Compute cumulative sum
cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

# Find tokens with cumsum <= top_p
nucleus = cumsum_probs <= top_p
```

**Example:**

```
Sorted probs: [0.50, 0.30, 0.12, 0.05, 0.02, 0.001, ...]

top_p = 0.9

Cumulative: [0.50, 0.80, 0.92, 0.97, 0.99, 0.991, ...]
            ↑     ↑     ↑ (exceeds 0.9, stop here)

Keep: [0.50, 0.30, 0.12]  (cumsum = 0.92, within 0.9? No, but include stopping token)

Result: Same as top-k ≈ 3 in this case
```

### Top-P vs Top-K

| Method | Pro | Con |
|--------|-----|-----|
| Top-K | Fixed, predictable | Ignores probability distribution |
| Top-P | Adaptive to distribution | May pick many tokens for uniform dist, few for peaked |

**Our code uses top-K** (simpler, more predictable).

---

## Part 6: Repetition Penalty

### The Problem: Repetition

Models often get stuck repeating tokens:

```
"The cat sat on the cat sat on the cat sat on..."
```

Why? Once a token is generated, it becomes part of the context, so the model might predict it again (it's now "in the conversation").

### The Solution: Repetition Penalty

Reduce the probability of tokens that have already been generated.

```python
for token in tokens[0]:  # All previously generated tokens
    logits[0, token] /= 1.35  # Divide logits by penalty factor

# Stronger penalty for recent tokens
recent_tokens = tokens[0][-10:]  # Last 10 tokens
for token in recent_tokens:
    logits[0, token] /= 1.5
```

**Effect:**

```
Before penalty: p("the") = 0.50
After penalty:  p("the") = 0.50 / 1.35 ≈ 0.37

"the" is less likely, but other tokens become more likely (after renormalization).

Result: Model generates different tokens!
```

### Hyperparameters

- **Global penalty (1.35)**: Applied to all previously generated tokens
- **Local penalty (1.5)**: Stronger penalty for last 10 tokens

These are heuristic choices. You could tune them based on your needs.

**Tradeoff:**
- Higher penalty → Less repetition, but less coherent
- Lower penalty → More coherent, but more repetition

---

## Part 7: Our Generation Pipeline

Let's trace through the full generation code.

### Step 1: Encode Seed

```python
start_text = "Once upon a"
tokens = sp.encode(start_text)  # [25, 102, 890]
tokens = torch.tensor(tokens).unsqueeze(0).to(device)  # (1, 3)
```

### Step 2: Generation Loop

```python
for step in range(max_tokens):  # e.g., max_tokens=50
    # Keep only last 128 tokens (our context window)
    tokens_cond = tokens[:, -block_size:]  # (1, seq_len) where seq_len <= 128
    
    # Model forward pass
    logits, _ = model(tokens_cond)  # (1, seq_len, vocab_size)
    
    # Get last token's logits
    logits = logits[:, -1, :]  # (1, vocab_size)
```

### Step 3: Apply Temperature

```python
temperature = 0.8
logits = logits / temperature
```

Modifies sharpness of probability distribution.

### Step 4: Repetition Penalty

```python
# Global penalty
for token in tokens[0]:
    logits[0, token] /= 1.35

# Local penalty (stronger for recent)
recent_tokens = tokens[0][-10:]
for token in recent_tokens:
    logits[0, token] /= 1.5
```

Reduces probability of already-generated tokens.

### Step 5: Softmax to Probabilities

```python
probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
```

Converts logits to probabilities (sum to 1).

### Step 6: Top-K Sampling

```python
top_k = 40

# Get top-k tokens
top_k_values, top_k_indices = torch.topk(probs, top_k)

# Renormalize
top_k_probs = top_k_values / top_k_values.sum(dim=-1, keepdim=True)

# Sample
next_token = torch.multinomial(top_k_probs, num_samples=1)  # (1, 1)
next_token = torch.gather(top_k_indices, -1, next_token)   # (1, 1)
```

Randomly select from top-40 tokens.

### Step 7: Append and Continue

```python
tokens = torch.cat((tokens, next_token), dim=-1)  # Append new token

# Check for end-of-sequence
if next_token.item() == sp.eos_id():
    break
```

Continue until max_tokens or EOS token.

### Step 8: Decode

```python
generated_text = sp.decode(tokens[0].tolist())
```

Convert token IDs back to text.

---

## Part 8: Generation Quality Metrics

### Subjective Metrics (Human Evaluation)

- **Coherence**: Does the text make sense?
- **Grammar**: Is it grammatically correct?
- **Diversity**: Does it avoid repetition?
- **Relevance**: Does it match the prompt?
- **Fluency**: Does it read naturally?

### Objective Metrics (Automatic Evaluation)

#### Perplexity (Lower is Better)

```
Perplexity = exp(average_cross_entropy_loss)
```

How "surprised" the model is by new text. Lower means the model thinks the text is more likely.

- Good text: Perplexity ≈ 10-50
- Bad text: Perplexity > 100

#### BLEU Score (Higher is Better)

Measures n-gram overlap with reference text.

- 1.0 = Perfect match
- 0.0 = No overlap

Commonly used in machine translation.

#### ROUGE Score (Higher is Better)

Recall-Oriented Understudy for Gisting Evaluation.

Similar to BLEU but emphasizes recall.

Used for summarization.

### Practical Approach

In practice, **read the generated text**. Human judgment is often more reliable than metrics.

---

## Part 9: Controlling Generation

### Controlling Length

```python
# Generate longer text
generate(model, sp, start_text, max_tokens=200)

# Generate shorter text
generate(model, sp, start_text, max_tokens=20)
```

Longer text has more chance of going off-track but also more expressiveness.

### Controlling Diversity

```python
# Conservative (high-quality, deterministic)
generate(model, sp, start_text, temperature=0.5, top_k=10)

# Balanced
generate(model, sp, start_text, temperature=0.8, top_k=40)

# Creative (high-diversity, potentially lower-quality)
generate(model, sp, start_text, temperature=1.5, top_k=100)
```

### Controlling Repetition

```python
# Less repetition (more penalty)
# Modify repetition penalty from 1.35 → 2.0

# More repetition (less penalty)
# Modify repetition penalty from 1.35 → 1.1
```

---

## Part 10: Common Generation Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| Repetitive text | Temperature too low, weak penalty | Increase temperature, increase penalty |
| Incoherent text | Temperature too high, top_k too large | Decrease temperature, decrease top_k |
| Short, stops early | Model learned EOS too early | Train longer, check data |
| Same output every time | temperature=0 (greedy) or seed deterministic | Increase temperature, use different seeds |
| Boring output | Model underfitted | Train on more data, longer training |
| Memory error | sequence too long | Reduce max_tokens, reduce batch_size |

---

## Part 11: Advanced Techniques

### Beam Search

Instead of sampling one token, keep top-K hypotheses and expand each.

**Pros:**
- ✅ More global optimization
- ✅ Often higher quality

**Cons:**
- ❌ Much slower
- ❌ Can be rigid (less diverse)

### Nucleus Sampling + Temperature

Combine top-p with temperature for flexibility:

```python
logits = logits / temperature
probs = softmax(logits)
probs = apply_top_p(probs, top_p=0.9)
next_token = multinomial(probs)
```

### Constrained Decoding

Generate text that satisfies constraints:

```python
# Example: Always alternate between adjectives and nouns
# Or: Generate text containing specific keywords
# Or: Generate text of specific length
```

More complex to implement.

### Length Penalty

Encourage longer/shorter sequences:

```python
# Favor shorter sequences
logits[eos_token_id] += length_bonus  # Increase EOS probability

# Favor longer sequences
logits[eos_token_id] -= length_bonus  # Decrease EOS probability
```

---

## Part 12: Debugging Generation

### Check 1: Test Greedy Decoding

```python
generate(model, sp, start_text, temperature=0.0, top_k=1)
```

Should be deterministic and high-quality (if model trained well).

### Check 2: Test High Randomness

```python
generate(model, sp, start_text, temperature=2.0, top_k=1000)
```

Should be very diverse and potentially incoherent.

### Check 3: Different Seeds

```python
for seed in ["Once upon a", "The cat", "I am"]:
    output = generate(model, sp, seed)
    print(f"{seed} → {output}\n")
```

Check if outputs are reasonable for different prompts.

### Check 4: Length Sensitivity

```python
for max_tokens in [10, 50, 100, 200]:
    output = generate(model, sp, start_text, max_tokens=max_tokens)
    print(f"Length {max_tokens}: {output}\n")
```

Does longer generation maintain coherence?

---

## Part 13: Key Takeaways

### Generation Pipeline

1. **Encode** seed text to token IDs
2. **Model forward** pass to get logits
3. **Apply temperature** to adjust confidence
4. **Repetition penalty** to avoid repeating
5. **Top-K filtering** to remove bad tokens
6. **Softmax** to get probabilities
7. **Sample** next token
8. **Decode** tokens to text

### Sampling Strategies

- **Greedy**: Always pick best (deterministic, can repeat)
- **Temperature**: Control randomness (lower = more confident)
- **Top-K**: Only sample from K best tokens (avoid nonsense)
- **Repetition penalty**: Reduce probability of seen tokens
- **Beam search**: Keep multiple hypotheses (slower)

### Hyperparameters

- **temperature**: 0.5-1.0 for quality, 1.5-2.0 for diversity
- **top_k**: 10-50 typical, higher = more diverse
- **max_tokens**: 10-500 typical, higher = longer text
- **Repetition penalty**: 1.2-2.0 typical

### Common Issues

- Repetitive text → Lower temperature, higher penalty
- Incoherent text → Raise temperature, lower top-k
- All same outputs → Increase temperature, randomness
- Going off-track → Lower temperature, lower top-k

---

## Next Steps

- Implement different sampling strategies
- Experiment with hyperparameters
- Implement beam search
- Try constrained decoding
- Compare with different models or seeds
