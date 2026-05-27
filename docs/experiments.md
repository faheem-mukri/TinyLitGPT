# Experiments: Testing Your Understanding

This document provides hands-on experiments to test and deepen your understanding of transformers. Each experiment is designed to be runnable and educational.

---

## Part 1: Tokenization Experiments

### Experiment 1.1: Inspect the Vocabulary

**Goal:** Understand what the tokenizer learned.

**Code:**
```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

# Print first 20 tokens
print("First 20 tokens:")
for i in range(20):
    piece = sp.id_to_piece(i)
    print(f"  {i}: '{piece}'")

# Print some random tokens
print("\nRandom tokens (ids 5000-5020):")
for i in range(5000, 5020):
    piece = sp.id_to_piece(i)
    print(f"  {i}: '{piece}'")

# Print last 20 tokens
print("\nLast 20 tokens:")
vocab_size = sp.get_piece_size()
for i in range(vocab_size - 20, vocab_size):
    piece = sp.id_to_piece(i)
    print(f"  {i}: '{piece}'")
```

**What to look for:**
- Special tokens at the beginning (</s>, <unk>, etc.)
- Common words/subwords
- Rare subwords towards the end
- Patterns in tokenization

**Questions to answer:**
1. How are spaces represented?
2. What's the smallest unit represented?
3. Do related morphemes share prefixes (e.g., "run", "running")?

---

### Experiment 1.2: Tokenization Consistency

**Goal:** Verify tokenizer is deterministic.

**Code:**
```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

test_strings = [
    "Hello world",
    "Hello world",  # Same, should tokenize identically
    "HELLO WORLD",  # Different case
    "hello world",  # Different case
    "Hello  world",  # Extra space
]

for s in test_strings:
    tokens = sp.encode(s)
    print(f"'{s}' → {tokens}")
```

**Expected behavior:**
- Same string → same tokens (deterministic)
- Different strings may → different tokens
- Case differences → different tokens
- Extra spaces may/may not matter (depends on SentencePiece rules)

**What it teaches:**
Tokenizers have specific rules. Understanding these rules helps debug generation issues.

---

### Experiment 1.3: Encoding/Decoding Round Trip

**Goal:** Verify tokenizer reversibility.

**Code:**
```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

test_strings = [
    "Once upon a time",
    "The quick brown fox",
    "Hello, world!",
    "I'm happy.",
    "Numbers: 123 456",
    "Special: @#$%",
    "Mixed CaSe",
]

for original in test_strings:
    tokens = sp.encode(original)
    decoded = sp.decode(tokens)
    match = "✓" if original == decoded else "✗"
    print(f"{match} '{original}' → {tokens} → '{decoded}'")
    if original != decoded:
        print(f"  Mismatch: '{original}' != '{decoded}'")
```

**Expected:** Most should match. Some might have subtle differences (whitespace normalization).

**What it teaches:**
- Tokenizer reversibility is important
- Edge cases in tokenization
- When round-trip fails, there's info loss

---

### Experiment 1.4: Vocabulary Coverage

**Goal:** See how different texts tokenize.

**Code:**
```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

texts = {
    "Simple": "The cat sat.",
    "Complex": "Pneumonoultramicroscopicsilicovolcanoconiosis.",
    "Technical": "Machine learning model training optimization.",
    "Rare": "Sesquipedalian floccinaucinilicilification.",
    "Numbers": "12345 67890",
    "Mixed": "It's 2024, the year of AI!",
}

for label, text in texts.items():
    tokens = sp.encode(text)
    num_tokens = len(tokens)
    avg_token_length = len(text) / num_tokens if num_tokens > 0 else 0
    print(f"{label:15} {text:50} → {num_tokens:3} tokens (avg: {avg_token_length:.2f} chars/token)")
```

**Pattern to observe:**
- Simple words: ~1-2 tokens
- Rare/complex words: More tokens
- Numbers/special chars: Variable

**What it teaches:**
Tokenizer length varies based on word rarity. Frequent words compress better.

---

## Part 2: Training Experiments

### Experiment 2.1: Learning Rate Comparison

**Goal:** See how learning rate affects training.

**Code:**
Save outputs to different files to compare:

```python
# In train.py, modify learning_rate:

for lr in [1e-5, 1e-4, 3e-4, 1e-3]:
    learning_rate = lr
    model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\n=== Training with LR={learning_rate} ===")
    running_loss = 0
    
    for step in range(1000):  # Short training for comparison
        x, y = get_batch(batch_size, block_size, device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        if step % 100 == 0 and step > 0:
            avg_loss = running_loss / 100
            print(f"  Step {step:4d} | Loss: {avg_loss:.4f}")
            running_loss = 0
```

**What to observe:**
- Which LR trains fastest?
- Which LR is most stable?
- Any divergence (NaN)?
- Final loss achieved by each?

**What it teaches:**
Learning rate is crucial. Too high = instability, too low = slow.

---

### Experiment 2.2: Batch Size Effect

**Goal:** Understand batch size impact.

**Code:**
```python
for batch_size in [1, 4, 16, 64]:
    print(f"\n=== Batch size: {batch_size} ===")
    running_loss = 0
    
    for step in range(1000):
        x, y = get_batch(batch_size, block_size, device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if step % 100 == 0 and step > 0:
            print(f"  Step {step:4d} | Loss: {running_loss/100:.4f}")
            running_loss = 0
```

**What to observe:**
- Training speed (steps/sec)
- Loss smoothness (noisier with small batch)
- Final loss achieved

**What it teaches:**
Batch size = noise level. Trade-off between stability and efficiency.

---

### Experiment 2.3: Gradient Clipping Effect

**Goal:** Understand gradient clipping.

**Code:**
```python
import torch
import matplotlib.pyplot as plt

# Train with different clipping values
for clip_value in [0.1, 0.5, 1.0, 5.0, None]:
    print(f"\n=== Gradient clip: {clip_value} ===")
    
    grad_norms = []
    losses = []
    
    for step in range(500):
        x, y = get_batch(batch_size, block_size, device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Measure gradient norm before clipping
        grad_norm = sum(p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None)
        grad_norms.append(grad_norm)
        
        # Apply clipping
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"  Step {step:4d} | Grad norm: {grad_norm:.4f} | Loss: {loss:.4f}")
    
    # Plot
    plt.figure()
    plt.plot(grad_norms, label="Gradient norm")
    plt.title(f"Gradient norms (clip={clip_value})")
    plt.ylabel("Norm")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig(f"grad_norm_clip_{clip_value}.png")
    plt.close()
```

**What to observe:**
- Which clipping values cause spikes?
- Which ones are stable?
- Any divergence (NaN)?

**What it teaches:**
Gradient clipping prevents explosions. Observe how it constrains gradients.

---

## Part 3: Attention Experiments

### Experiment 3.1: Visualize Attention Weights

**Goal:** Understand what the model attends to.

**Code:**
```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Modify model to save attention weights
class Head(nn.Module):
    def __init__(self, head_size, d_model, block_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attention_weights = None  # Store for visualization

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        
        self.attention_weights = wei  # Save for later
        
        v = self.value(x)
        out = wei @ v
        return out

# Test on a short sequence
tokens = sp.encode("The cat sat on the mat")
tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

with torch.no_grad():
    logits, _ = model(tokens_tensor)

# Extract attention from first head of first layer
att_weights = model.blocks[0].sa.heads[0].attention_weights  # (1, T, T)
att_weights = att_weights[0].cpu().numpy()  # (T, T)

# Visualize
plt.figure(figsize=(8, 8))
plt.imshow(att_weights, cmap="hot")
plt.colorbar()
plt.title("Attention Weights (First Head, First Layer)")
plt.xlabel("Key position")
plt.ylabel("Query position")
plt.xticks(range(len(tokens)), tokens)
plt.yticks(range(len(tokens)), tokens)
plt.tight_layout()
plt.savefig("attention_visualization.png", dpi=100)
plt.close()

print("Attention visualization saved!")
print(f"\nToken sequence: {[sp.id_to_piece(t) for t in tokens]}")
```

**What to look for:**
- Does each token attend to itself? (diagonal should be visible)
- Do related tokens attend to each other?
- Is attention causal? (upper triangle should be empty)
- Any learned patterns?

**What it teaches:**
Attention patterns show what the model learned. Different heads learn different patterns.

---

### Experiment 3.2: Attention Head Analysis

**Goal:** Compare different attention heads.

**Code:**
```python
# Save attention from all 8 heads
tokens = sp.encode("The cat sat on the mat")
tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

with torch.no_grad():
    logits, _ = model(tokens_tensor)

# Plot each head
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for head_idx, ax in enumerate(axes):
    att = model.blocks[0].sa.heads[head_idx].attention_weights[0].cpu().numpy()
    im = ax.imshow(att, cmap="hot")
    ax.set_title(f"Head {head_idx}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("all_heads_attention.png", dpi=100)
plt.close()

print("All heads visualization saved!")
```

**What to observe:**
- Do different heads focus on different things?
- Some heads local (attend nearby tokens), others global?
- Any systematic differences?

**What it teaches:**
Multi-head attention learns diverse patterns. Each head specializes.

---

## Part 4: Generation Experiments

### Experiment 4.1: Temperature Effect

**Goal:** Empirically observe temperature impact.

**Code:**
```python
import sentencepiece as spm
from model.gpt import GPT
import torch
import torch.nn.functional as F

# Load model
model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

def generate_with_temperature(model, sp, start_text, temperature):
    tokens = sp.encode(start_text)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    for _ in range(30):  # Generate 30 tokens
        tokens_cond = tokens[:, -block_size:]
        logits, _ = model(tokens_cond)
        logits = logits[:, -1, :]
        
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if next_token.item() == sp.eos_id():
            break
    
    return sp.decode(tokens[0].tolist())

# Test different temperatures
start_text = "Once upon a"
temperatures = [0.5, 0.8, 1.0, 1.5, 2.0]

for temp in temperatures:
    print(f"\nTemperature: {temp}")
    for i in range(3):
        output = generate_with_temperature(model, sp, start_text, temp)
        print(f"  Run {i+1}: {output}")
```

**What to observe:**
- Temperature 0.5: Consistent, high-quality
- Temperature 0.8: Good balance
- Temperature 1.5: More diverse, sometimes nonsensical
- Temperature 2.0: Very diverse, often incoherent

**What it teaches:**
Temperature controls creativity vs. coherence trade-off.

---

### Experiment 4.2: Top-K Filtering Effect

**Goal:** Observe top-k impact on generation.

**Code:**
```python
def generate_with_topk(model, sp, start_text, top_k):
    tokens = sp.encode(start_text)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    for _ in range(30):
        tokens_cond = tokens[:, -block_size:]
        logits, _ = model(tokens_cond)
        logits = logits[:, -1, :]
        
        probs = F.softmax(logits, dim=-1)
        
        # Top-K filtering
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        top_k_probs = top_k_probs / top_k_probs.sum()
        
        next_token = torch.multinomial(top_k_probs, 1)
        next_token = torch.gather(top_k_indices, -1, next_token)
        
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if next_token.item() == sp.eos_id():
            break
    
    return sp.decode(tokens[0].tolist())

# Test different top-k values
top_k_values = [5, 10, 20, 50, 100]
start_text = "The cat"

for top_k in top_k_values:
    print(f"\nTop-K: {top_k}")
    output = generate_with_topk(model, sp, start_text, top_k)
    print(f"  {output}")
```

**What to observe:**
- top_k=5: Very conservative, safe
- top_k=50: Balanced
- top_k=100+: Diverse but risky

**What it teaches:**
Top-k controls diversity vs. safety trade-off.

---

### Experiment 4.3: Repetition Penalty Analysis

**Goal:** See repetition penalty effect.

**Code:**
```python
def generate_with_penalty(model, sp, start_text, penalty):
    tokens = sp.encode(start_text)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    for _ in range(50):
        tokens_cond = tokens[:, -block_size:]
        logits, _ = model(tokens_cond)
        logits = logits[:, -1, :]
        
        # Apply repetition penalty
        if penalty > 0:
            for token in tokens[0]:
                logits[0, token] /= penalty
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if next_token.item() == sp.eos_id():
            break
    
    return sp.decode(tokens[0].tolist())

# Test different penalties
penalties = [1.0, 1.2, 1.5, 2.0]
start_text = "I like"

for penalty in penalties:
    print(f"\nRepetition Penalty: {penalty}")
    output = generate_with_penalty(model, sp, start_text, penalty)
    print(f"  {output}")
```

**What to observe:**
- penalty=1.0: No penalty, may repeat
- penalty=1.5: Good balance
- penalty=2.0: Over-penalized, awkward phrasing

**What it teaches:**
Balance between avoiding repetition and maintaining naturalness.

---

## Part 5: Model Architecture Experiments

### Experiment 5.1: Layer Count Impact

**Goal:** Compare different number of layers.

**Code:**
```python
# Create models with different layer counts
configs = [
    {"name": "4 layers", "n_layers": 4},
    {"name": "8 layers", "n_layers": 8},
    {"name": "12 layers", "n_layers": 12},
]

for config in configs:
    print(f"\n=== {config['name']} ===")
    
    model = GPT(vocab_size, d_model, n_heads, config['n_layers'], block_size)
    model.to(device)
    model.train()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    
    # Quick training test
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    for step in range(1000):
        x, y = get_batch(batch_size, block_size, device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0 and step > 0:
            print(f"  Step {step:4d} | Loss: {loss.item():.4f}")
```

**What to observe:**
- Parameter count increase
- Training speed (slower with more layers?)
- Final loss (does more layers help?)

**What it teaches:**
Architectural choices have clear trade-offs: capacity vs. speed.

---

### Experiment 5.2: Block Size (Context Window) Impact

**Goal:** Understand context window effects.

**Code:**
```python
# Generate text and observe coherence with different block sizes

prompts = [
    "Once upon a time, there was a girl named Lucy. She had",
    "The most important thing to remember about transformers is",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    output = generate(model, sp, prompt, max_tokens=100)
    
    # Count tokens in output
    tokens = sp.encode(output)
    print(f"Generated {len(tokens)} tokens:")
    print(f"  {output}")
```

**Observation:**
- Early tokens: Coherent with prompt
- Later tokens: May diverge (context window effect)

**What it teaches:**
Context window limits long-range coherence. Larger context = longer consistency.

---

## Part 6: Data and Domain Experiments

### Experiment 6.1: Domain-Specific Training

**Goal:** See how training data affects generation style.

**Code:**
```python
# This would require retraining on different data, but conceptually:

# Train on:
# Dataset A: TinyStories (children's stories)
# Dataset B: Wikipedia (factual articles)
# Dataset C: Poetry (verse)

# Compare outputs:
# "Once upon a" generates story-like text from A, factual from B, poetic from C

# Code sketch (don't actually run without data):
# for dataset_name, dataset_path in [("stories", ...), ("wiki", ...), ("poetry", ...)]:
#     train_on(dataset_path)
#     outputs.append(generate(model, sp, "The"))
```

**What it teaches:**
Model learns the "style" and "voice" of training data. Different domains → different models.

---

### Experiment 6.2: Vocabulary Size Effect

**Goal:** Understand vocab size trade-offs.

**Code:**
```python
# Create tokenizers with different vocab sizes
vocab_sizes = [1000, 5000, 10000, 50000]

for vocab_size in vocab_sizes:
    # Would need to retrain tokenizer
    # spm.SentencePieceTrainer.train(
    #     input="tokenizer/data.txt",
    #     vocab_size=vocab_size,
    #     ...
    # )
    
    # Then compare:
    # - Average tokens per sentence
    # - Model size
    # - Generation quality
    pass
```

**Conceptual findings (from literature):**
- Smaller vocab: Longer sequences, simpler to learn, but inefficient
- Larger vocab: Shorter sequences, more parameters, overkill for small models

---

## Part 7: Analysis Experiments

### Experiment 7.1: Loss Analysis by Token Position

**Goal:** Understand if some positions are harder to predict.

**Code:**
```python
# During training, track loss by position
position_losses = {}

for step in range(1000):
    x, y = get_batch(batch_size, block_size, device)
    logits, loss = model(x, y)
    
    # Compute loss for each position
    logits_flat = logits.view(-1, vocab_size)
    y_flat = y.view(-1)
    
    for pos in range(block_size):
        loss_pos = F.cross_entropy(
            logits_flat[pos::block_size],
            y_flat[pos::block_size],
            reduction='mean'
        )
        
        if pos not in position_losses:
            position_losses[pos] = []
        position_losses[pos].append(loss_pos.item())

# Plot average loss by position
avg_losses = {pos: np.mean(losses) for pos, losses in position_losses.items()}

plt.figure()
plt.plot(avg_losses.keys(), avg_losses.values())
plt.xlabel("Position in sequence")
plt.ylabel("Average loss")
plt.title("Loss by token position")
plt.savefig("loss_by_position.png")
plt.close()
```

**What to observe:**
- Early positions: Higher loss? (harder to predict with less context)
- Late positions: Lower loss? (more context available)

**What it teaches:**
Model accuracy varies by position. Earlier tokens are harder to predict.

---

### Experiment 7.2: Most Common Prediction Errors

**Goal:** Understand what the model gets wrong.

**Code:**
```python
# Track predictions and errors
correct_count = 0
total_count = 0
error_log = {}

with torch.no_grad():
    for _ in range(100):  # Process 100 batches
        x, y = get_batch(batch_size, block_size, device)
        logits, _ = model(x, y)
        
        predictions = torch.argmax(logits, dim=-1)
        
        # Compare predictions to targets
        batch_correct = (predictions == y).sum().item()
        batch_total = y.numel()
        
        correct_count += batch_correct
        total_count += batch_total
        
        # Log errors
        errors = (predictions != y)
        for i in range(errors.shape[0]):
            for j in range(errors.shape[1]):
                if errors[i, j]:
                    target = y[i, j].item()
                    pred = predictions[i, j].item()
                    key = (target, pred)
                    if key not in error_log:
                        error_log[key] = 0
                    error_log[key] += 1

accuracy = correct_count / total_count
print(f"Accuracy: {accuracy:.2%}")

# Top errors
print("\nMost common prediction errors:")
sorted_errors = sorted(error_log.items(), key=lambda x: x[1], reverse=True)
for (true_token, pred_token), count in sorted_errors[:10]:
    true_piece = sp.id_to_piece(true_token)
    pred_piece = sp.id_to_piece(pred_token)
    print(f"  True: {true_piece:20} | Predicted: {pred_piece:20} | Count: {count}")
```

**What to observe:**
- Which words does the model confuse?
- Common semantic errors?
- Grammatical errors?

**What it teaches:**
Error analysis shows what the model struggles with. Guides future improvements.

---

## Part 8: Suggested Exploration Roadmap

**Week 1: Understanding**
- Run tokenization experiments (1.1-1.4)
- Understand vocabulary and encoding

**Week 2: Training**
- Run training experiments (2.1-2.3)
- Tune hyperparameters
- Observe training dynamics

**Week 3: Attention**
- Visualize attention weights (3.1-3.2)
- Understand what model learns

**Week 4: Generation**
- Run generation experiments (4.1-4.3)
- Understand sampling strategies

**Week 5: Architecture**
- Run architecture experiments (5.1-5.2)
- Compare configurations

**Week 6: Analysis**
- Run analysis experiments (7.1-7.2)
- Deep dive into errors

---

## Key Principles for Running Experiments

1. **Change one thing at a time:** Isolate variables
2. **Run multiple times:** Account for randomness
3. **Save results:** Track what you learned
4. **Visualize:** Plot data to spot patterns
5. **Take notes:** Document your findings
6. **Compare:** Run baselines for comparison
7. **Iterate:** Build on previous experiments

---

## Next Steps

After running these experiments:
- Identify what you find most interesting
- Design new experiments based on curiosity
- Read research papers on related topics
- Scale up to larger models
- Apply to downstream tasks
