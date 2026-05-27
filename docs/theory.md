# Transformer Theory: Understanding Language Models from First Principles

This document explains the fundamental concepts behind transformers and language models. If you're new to this, read this first—it builds intuition before diving into the code.

---

## Part 1: What is a Language Model?

### The Core Idea

A **language model** is a statistical machine that learns to predict the next word in a sequence. That's it.

Think of it like this: if I say "The weather is very...", you can predict the next word is likely "sunny," "rainy," or "cold." You're using patterns you've learned from reading thousands of sentences.

A language model does the same thing, but mathematically.

### Example

```
Input:   "Once upon a"
Output:  "time" (with high probability)

Input:   "I am feeling very"
Output:  "happy" (with high probability)
```

The model assigns a **probability** to every possible next word in its vocabulary. The sum of all these probabilities equals 1.

### Why is This Useful?

By chaining these predictions together, we can **generate text**:

```
Seed: "The cat"
→ Predict: "sat"     (now: "The cat sat")
→ Predict: "on"      (now: "The cat sat on")
→ Predict: "the"     (now: "The cat sat on the")
→ Predict: "mat"     (now: "The cat sat on the mat")
```

This is called **autoregressive generation**—we use our own previous predictions as input for the next prediction.

---

## Part 2: The Problem Transformers Solve

### Before Transformers: RNNs and Attention

Before 2017, people used **Recurrent Neural Networks (RNNs)** for language modeling. The idea was simple:

- Process tokens one at a time, left to right
- Keep a "hidden state" that remembers previous context
- Each new token updates the hidden state

**The Problem**: This is sequential. You can't process token 10 until you've processed tokens 1-9. On GPUs with parallel processing, this wastes computational power.

Also, information stored in the hidden state from early tokens often gets "forgotten" by the time you reach later tokens (the vanishing gradient problem).

### A Better Idea: Attention

What if the model could look at **all previous tokens at once** and decide which ones matter for predicting the next word?

Example:
```
Sentence: "The dog ran because it was hungry"

Predicting: what word comes after "it"?

The model attends to:
- "dog" (moderate attention) - subject
- "it" (high attention) - this word depends on the antecedent
- "was hungry" (high attention) - context for why the dog ran

Result: predicts "was" (the model learns that "it was..." is common)
```

This **attention mechanism** lets the model:
1. Process all tokens in parallel (fast!)
2. Look back at any previous token (no forgetting)
3. Learn which tokens are relevant for the current prediction

### Transformers: Attention + Parallelism

The **Transformer** (introduced in "Attention is All You Need", 2017) took this idea and built an entire architecture around attention. No RNN needed.

Key insight: **"Attention is all you need"**—you don't need sequential processing; you can build powerful models with just attention.

---

## Part 3: How Transformers Process Text

### The High-Level Flow

```
Raw Text Input
    ↓
Tokenization (text → token IDs)
    ↓
Token Embeddings (IDs → vectors)
    ↓
Add Positional Information (remember token positions)
    ↓
Transformer Blocks (multiple layers of attention + feedforward)
    ↓
Language Model Head (predict next token probabilities)
    ↓
Probabilities for each vocabulary word
```

### Stage 1: Tokenization

The model doesn't understand text; it understands numbers.

```
"Hello world"
    ↓ (tokenizer)
[25, 102, 890]  # token IDs
```

These IDs are learned during tokenizer training to represent meaningful subwords or characters.

### Stage 2: Embeddings

Each token ID is converted to a **dense vector** (embedding).

```
Token ID: 25
    ↓
Embedding: [0.2, -0.5, 0.1, 0.8, -0.3, ...]  # 512-dimensional vector
```

Why vectors? Because math operations on vectors are efficient on GPUs, and meaning can be encoded in the vector space (similar words have similar vectors).

**Key insight**: The embedding vectors are **learned parameters**. During training, the model adjusts these vectors so that semantically related words have similar embeddings.

### Stage 3: Positional Embeddings

Transformers have a problem: they process all tokens in parallel, so they don't inherently know the order of tokens.

Solution: **positional embeddings**. Each position gets its own embedding that encodes "you are at position 0," "you are at position 1," etc.

```
Token embedding:      [0.2, -0.5, 0.1, ...]    # what the word means
Positional embedding: [0.8, 0.1, -0.3, ...]   # where it is in the sequence

Combined (added):     [1.0, -0.4, -0.2, ...]  # enriched with position info
```

These positional embeddings are learned during training (or can be hardcoded using trigonometric functions).

### Stage 4: Transformer Blocks

Now we have vectors with meaning and position info. We run them through multiple **transformer blocks**.

Each block does:
1. **Self-attention**: Each token attends to all other tokens, learning which ones are relevant
2. **Feedforward network**: A simple multi-layer network that applies non-linear transformations

These are stacked many times (in our model: 8 times).

### Stage 5: Language Model Head

After all the attention and processing, we have a vector for each position in the sequence.

To predict the next token, we take the **last token's vector** and pass it through a final linear layer:

```
Last token vector: [0.5, -0.2, 0.8, ...]  (512-dimensional)
    ↓
Linear layer
    ↓
Logits: [0.1, 2.3, -0.5, 1.9, ...]  (vocab_size-dimensional, e.g., 10,000)
    ↓
Softmax (convert to probabilities)
    ↓
Probabilities: [0.001, 0.7, 0.02, 0.15, ...]  (sum to 1)
```

The highest probability token is the model's "best guess" for the next word.

---

## Part 4: Self-Attention Explained Simply

This is the core mechanism that makes transformers work. Let's build intuition.

### The Problem It Solves

Consider this sentence:
```
"The bank executive called. She had important news."
```

To predict the next word after "She", you need to know what "She" refers to—"the bank executive." But "She" and "bank executive" are far apart. How does the model learn this connection?

**Self-attention** lets the model learn this: when processing "She", look at "bank executive" and figure out they refer to the same entity.

### How It Works: Query, Key, Value

Imagine you're searching through a document:

1. **Query**: "What does 'She' refer to?"
2. **Key**: Each word has a label. "bank" is labeled "noun", "She" is labeled "pronoun"
3. **Value**: Each word has a meaning. "bank executive" has rich meaning

The model:
1. Converts the question into a vector (Query)
2. Converts each word's label into vectors (Keys)
3. Compares Question to all Keys: "Does any key match this question?"
4. Takes the Values of matching words

**Mathematically**:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

Where:
Q = Query matrix (computed from input)
K = Key matrix (computed from input)
V = Value matrix (computed from input)
d = dimension size (for scaling)
```

**What this does**:
- `Q @ K^T`: Compare query to all keys (which words are relevant?)
- `/ sqrt(d)`: Scale to avoid extreme values
- `softmax`: Convert to probabilities (which words to focus on?)
- `@ V`: Combine the values of relevant words

### Concrete Example

Let's say we're computing attention for the word "She" in our sentence.

```
Sentence: "The bank executive called. She had important news."
We're processing: "She"

Step 1: Compute Query for "She"
Query_She = [0.5, 0.2, -0.1, ...]

Step 2: Compute Keys for all words
Key_the = [0.1, 0.0, 0.2, ...]
Key_bank = [0.8, -0.1, 0.0, ...]        ← High similarity to query!
Key_executive = [0.7, 0.1, 0.1, ...]    ← High similarity to query!
...
Key_She = [0.4, 0.3, 0.2, ...]

Step 3: Compute similarities (dot products)
Query_She · Key_the = 0.05 (low)
Query_She · Key_bank = 0.38 (high)
Query_She · Key_executive = 0.35 (high)
...
Query_She · Key_She = 0.35

Step 4: Apply softmax to get attention weights
"the" gets 5% of attention
"bank" gets 15% of attention
"executive" gets 12% of attention
"She" gets 10% of attention
... (sums to 100%)

Step 5: Weighted sum of values
Output = 0.05 * Value_the + 0.15 * Value_bank + 0.12 * Value_executive + ... 
       = vector representing "the bank executive" (through the model's learned perspective)
```

### Why This is Powerful

The model **learns to pay attention to relevant context**. It doesn't need hardcoded rules; it figures out through training which words matter for which predictions.

---

## Part 5: Multi-Head Attention

Why have one attention mechanism when you can have many?

**Multi-head attention** runs the same attention mechanism multiple times, in parallel, each learning different patterns.

### The Intuition

Think of different "heads" as different perspectives:

- **Head 1**: Learns to attend to grammatical agreement (subject-verb matching)
- **Head 2**: Learns to attend to semantic relationships (referents)
- **Head 3**: Learns to attend to long-range dependencies
- **Head 4-8**: Learn other patterns

Each head independently learns what to attend to.

```
Input: [word1, word2, word3, ...]
    ↓
Attention Head 1: [attention1_1, attention1_2, attention1_3, ...]
Attention Head 2: [attention2_1, attention2_2, attention2_3, ...]
Attention Head 3: [attention3_1, attention3_2, attention3_3, ...]
...
    ↓ (concatenate)
Combined: [att1_1, att1_2, ..., att2_1, att2_2, ..., att3_1, ...]
    ↓ (linear projection)
Output: processed information from all perspectives
```

**Why this works**: Different heads learn complementary patterns. By combining them, the model gets a richer representation of context.

In our model, we use **8 heads**, each learning different attention patterns.

---

## Part 6: Feedforward Networks in Transformers

After attention, each transformer block includes a **feedforward network**.

### Why Is It There?

Attention is great for figuring out "which tokens matter." But it doesn't add much *new information*.

The feedforward network is where the model can learn complex non-linear transformations. It's a simple multi-layer neural network:

```
Input: [vector of dimension 512]
    ↓
Linear Layer 1 (512 → 2048): Expand
    ↓
ReLU (non-linearity): Introduce complexity
    ↓
Linear Layer 2 (2048 → 512): Contract back
    ↓
Output: [vector of dimension 512]
```

**What does it learn?** Non-linear transformations that can't be expressed by attention alone. For example:
- "If token 1 is 'verb' AND token 2 is 'past tense', then modify output this way"
- Numerical reasoning patterns
- Rare word combinations

### Intuition

If attention is "which tokens matter," the feedforward is "how to combine and transform them."

---

## Part 7: Residual Connections and Layer Normalization

### The Problem: Deep Networks Are Hard to Train

When you stack many layers, gradients can vanish or explode as they flow backward during training. This makes learning unstable.

### Solution 1: Residual Connections

Instead of:
```
output = Layer(input)
```

Use:
```
output = Layer(input) + input
```

This creates a "highway" for gradients to flow directly backward without diminishing.

**Analogy**: Imagine you're on a steep mountain. Instead of only following the rocky path, you also have a direct shortcut that skips part of the climb. Gradients can flow back through this shortcut.

### Solution 2: Layer Normalization

After adding the residual connection, we normalize:

```
output = LayerNorm(attention(input) + input)
output = LayerNorm(feedforward(output) + output)
```

**Why?** Normalizing keeps the values in a stable range, making training more stable and faster.

**How?** For each vector, subtract the mean and divide by the standard deviation (like standardizing data in statistics).

### In Our Code

```python
class Block(nn.Module):
    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.sa(self.ln1(x))  # ln1 = layer norm before attention
        
        # Apply feedforward with residual connection
        x = x + self.ff(self.ln2(x))  # ln2 = layer norm before feedforward
        
        return x
```

**Key insight**: We apply layer norm *before* each sublayer (this is called "pre-norm"), which is more stable than applying it after.

---

## Part 8: Causal Masking (Why the Model Can't Cheat)

### The Problem

When training a language model, we want it to predict the next token based only on *previous* tokens.

But in attention, each token can attend to *all* tokens, including future ones!

If we let the model peek at future tokens, it's cheating—it's not really learning to predict the future.

### The Solution: Causal Mask

We mask out (set to zero attention) any future tokens:

```
Position 0 can attend to: [0]
Position 1 can attend to: [0, 1]
Position 2 can attend to: [0, 1, 2]
Position 3 can attend to: [0, 1, 2, 3]
...

Visualized (X = can attend, 0 = can't attend):
     0 1 2 3
 0: [X 0 0 0]
 1: [X X 0 0]
 2: [X X X 0]
 3: [X X X X]
```

This is a **triangular mask** (lower triangular matrix).

### In Our Code

```python
# Register a triangular mask
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

# Apply it to attention weights
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

Any position where the mask is 0, we set attention to `-inf`, which becomes 0 probability after softmax.

---

## Part 9: Why Transformers Scale

### What Makes Transformers Different?

1. **Parallelizable**: All tokens are processed at once (not sequentially like RNNs)
2. **Long-range dependencies**: Any token can attend to any other (no forgetting)
3. **Composable**: Stack many layers and the model learns to compose features

### The Scaling Laws

Empirically, we've learned:

- **Bigger models**: More parameters → better performance (up to a point)
- **More data**: More diverse training data → better generalization
- **Longer training**: More optimization steps → better convergence

This is why modern LLMs are enormous (billions of parameters). The same architecture scales from millions (our tiny model) to billions (GPT-4).

---

## Part 10: From Training to Generation

### Training Phase: Predicting Given Context

During training:
```
Input:  [token1, token2, token3, token4]
Target: [token2, token3, token4, token5]

Loss = cross_entropy(predicted_tokens, target_tokens)
```

The model learns: "Given these tokens, predict the next one."

### Generation Phase: Autoregressive Decoding

During generation:
```
Seed: [token1]
→ Predict token2 → [token1, token2]
→ Predict token3 → [token1, token2, token3]
→ Predict token4 → [token1, token2, token3, token4]
→ ... until max length or end-of-sequence token
```

Each prediction becomes part of the context for the next prediction.

### The Context Window Limit

Our model has `block_size = 128`, meaning it can only attend to the last 128 tokens.

During generation:
```python
tokens_cond = tokens[:, -block_size:]  # Keep only the last 128 tokens
```

This limits memory usage but also limits how far back the model can look. Larger models can have larger context windows (GPT-4 can see millions of tokens).

---

## Part 11: Key Takeaways

### Conceptual

1. **Language models predict the next token** based on previous ones
2. **Attention** lets the model learn which previous tokens matter most
3. **Multiple layers** let the model build complex reasoning
4. **Parallelism** makes training fast despite having many parameters

### Practical

1. **Tokenization** converts text to numbers
2. **Embeddings** convert numbers to vectors with meaning
3. **Transformers** process these vectors through attention and feedforward layers
4. **Training** teaches the model to minimize prediction error
5. **Generation** chains predictions together to produce text

### The Beauty of This Approach

We don't hardcode any language rules. We just define the architecture and let the model learn everything from data. Attention patterns, grammatical rules, semantic relationships—all emerge from training.

---

## Part 12: Common Misconceptions

### "The Model Understands"

Not really. It's learning statistical patterns in text. It's very good at pattern matching, but it doesn't have consciousness or true understanding. It's sophisticated statistics.

### "Bigger Always Means Better"

Not always. Bigger models need more data and compute. A small model trained well can outperform a huge model trained poorly.

### "Training Until Loss = 0"

You can't. There's always some randomness and complexity in language. The goal is to lower loss until validation performance plateaus (where adding more training doesn't help).

### "The Model Learns Grammar Rules"

Not explicitly. Through training on text (which follows grammar rules), the model's attention patterns emerge that implement grammar. But there's no explicit "rule engine."

---

## What's Next?

Now that you understand the theory:

- **Tokenization**: Learn how text becomes numbers (see `tokenization.md`)
- **Attention**: Dive deeper into the math (see `attention.md`)
- **Training**: Understand how the model learns (see `training.md`)
- **Sampling**: Learn how to control generation (see `sampling.md`)
- **Experiments**: Run code and test your intuitions (see `experiments.md`)
