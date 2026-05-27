# Training: Teaching the Model to Predict

Training is how a transformer learns. This document explains the training loop, loss functions, optimization, and practical considerations.

---

## Part 1: What Does Training Do?

### The Core Idea

Training adjusts the model's parameters (weights and biases) so it makes better predictions.

**Before training:**
```
Random weights → Bad predictions → High loss
```

**After training:**
```
Optimized weights → Better predictions → Lower loss
```

### The Cycle

```
1. Forward Pass:    Input → Model → Predictions (logits)
2. Compute Loss:    Compare predictions to ground truth
3. Backward Pass:   Compute gradients
4. Optimize:        Update weights using gradients
5. Repeat
```

### Loss Function: What We're Minimizing

We want the model to output high probability for the correct next token.

**Example:**

```
Input:  "The cat sat on the"
Target: "mat"

Model outputs probabilities for all vocab words:
"the":     0.01
"mat":     0.7    ← Correct! High probability
"dog":     0.05
"bird":    0.02
... (other words have tiny probabilities)

Loss = how wrong is this prediction?
Since "mat" has 0.7 probability (pretty confident), loss is low.

If model predicted:
"the":     0.4    ← Oops, predicted wrong word with high confidence
"mat":     0.01   ← Missed the correct answer
Loss = high (bad prediction!)
```

---

## Part 2: Cross-Entropy Loss

We use **cross-entropy loss**, the standard for classification tasks.

### The Math

```
CE Loss = -log(p_correct)

Where p_correct = model's predicted probability for the correct word
```

**Intuition:**
- If p_correct = 0.9 (confident): -log(0.9) ≈ 0.1 (low loss)
- If p_correct = 0.5 (uncertain): -log(0.5) ≈ 0.7 (medium loss)
- If p_correct = 0.1 (wrong): -log(0.1) ≈ 2.3 (high loss)

The model is incentivized to assign high probability to the correct token.

### In Code

```python
# In model/gpt.py forward pass
logits = self.lm_head(x)  # Shape: (B, T, vocab_size)
                          # Unnormalized scores for each vocab word

if targets is not None:
    # Flatten for loss computation
    B, T, C = logits.shape
    logits = logits.view(B * T, C)      # (B*T, vocab_size)
    targets = targets.view(B * T)       # (B*T,)
    
    # Cross-entropy: softmax + log + negative mean
    loss = F.cross_entropy(logits, targets)
```

**What `F.cross_entropy` does:**
1. Apply softmax to logits (normalize to probabilities)
2. Compute -log(p_correct) for each sample
3. Average over all samples

### Why Logits, Not Probabilities?

Why not compute softmax first, then do cross-entropy?

```
# Inefficient:
probs = softmax(logits)
loss = -log(probs[correct_idx])

# Efficient (what PyTorch does):
loss = cross_entropy(logits, correct_idx)  # Uses log-sum-exp trick
```

PyTorch combines them for numerical stability (avoids underflow/overflow).

---

## Part 3: The Training Loop

### File: `train.py`

Let's walk through the full training process.

### Step 1: Setup

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use GPU if available (much faster)

batch_size = 4
block_size = 128
d_model = 512
n_heads = 8
n_layers = 8

learning_rate = 3e-4  # 0.0003
max_iters = 5000      # Training steps
eval_interval = 100   # Print loss every 100 steps
```

**Hyperparameter choices:**
- **batch_size=4**: Small but reasonable (larger would be faster on GPU)
- **learning_rate=3e-4**: Standard for transformers (not too large, not too small)
- **max_iters=5000**: Enough to see convergence; could go higher
- **eval_interval=100**: Check progress frequently

### Step 2: Load Tokenizer and Data

```python
sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")
vocab_size = sp.get_piece_size()  # 10,000

# Data loading happens in get_batch()
# Loaded on-demand from data/train.pt
```

### Step 3: Initialize Model

```python
model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
model = model.to(device)  # Move to GPU if available
model.train()  # Set to training mode
```

**Why `model.train()`?** Some layers (like BatchNorm, Dropout) behave differently in training vs. eval. We use `model.train()` for training.

### Step 4: Setup Optimizer

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

**AdamW:** Adaptive learning rate optimization with weight decay.

**Alternatives:**
- SGD: Simple but often slower
- Adam: Simpler version (without weight decay)
- AdamW: Adam with weight decay (better generalization) ✓

**Weight decay** helps prevent overfitting by penalizing large weights.

### Step 5: Training Loop

```python
running_loss = 0

for step in range(max_iters):
    # Get a batch of data
    x, y = get_batch(batch_size, block_size, device)
    # x: (batch_size, block_size) token IDs
    # y: (batch_size, block_size) target token IDs (shifted by 1)
    
    # Forward pass: compute predictions and loss
    logits, loss = model(x, y)
    
    # Zero gradients from previous step
    optimizer.zero_grad()
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Gradient clipping (discussed later)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step: update weights
    optimizer.step()
    
    # Track loss
    running_loss += loss.item()
    
    # Print progress
    if step % eval_interval == 0 and step > 0:
        avg_loss = running_loss / eval_interval
        print(f"Step {step} | Loss: {avg_loss:.4f}")
        running_loss = 0
        
        # Save checkpoint
        if step % 1000 == 0:
            torch.save(model.state_dict(), f"checkpoint_{step}.pth")

# Save final model
torch.save(model.state_dict(), "model.pth")
```

---

## Part 4: Gradients and Backpropagation

### What Are Gradients?

Gradients tell us: "Which direction should I adjust each weight to lower the loss?"

```
Loss = f(weights)

Gradient = ∂Loss / ∂weights

If ∂Loss/∂w = positive: Increasing w increases loss (bad) → decrease w
If ∂Loss/∂w = negative: Increasing w decreases loss (good) → increase w
```

### How Backpropagation Works

Consider a simple computation:

```python
x = 2
w = 3
y = x * w
loss = y^2
```

Forward pass:
```
y = 2 * 3 = 6
loss = 6^2 = 36
```

Backward pass (chain rule):
```
∂loss/∂loss = 1
∂loss/∂y = 2 * y * ∂y/∂y = 2 * 6 = 12
∂loss/∂w = ∂loss/∂y * ∂y/∂w = 12 * x = 12 * 2 = 24
```

Update:
```
w_new = w - learning_rate * ∂loss/∂w = 3 - 0.01 * 24 = 2.76
```

In PyTorch:
```python
x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)  # We want to track gradients for w
y = x * w
loss = y ** 2
loss.backward()  # Compute gradients
print(w.grad)    # ∂loss/∂w = 24.0
```

### In Our Model

In practice, we have millions of parameters. PyTorch automatically computes gradients for all of them using backpropagation.

```python
loss.backward()  # Computes gradients for all trainable parameters
# model.token_embedding.weight.grad  = computed
# model.position_embedding.weight.grad = computed
# model.blocks[0].sa.heads[0].query.weight.grad = computed
# ... (many thousands more)
```

**The beauty:** We don't manually compute these. PyTorch does it for us.

---

## Part 5: Gradient Clipping

### The Problem: Exploding Gradients

Sometimes, gradients can become huge:

```
∂loss/∂w = 1000

w_new = w - 0.001 * 1000 = w - 1

If w was 0.5, now it's -0.5. Huge jump!
This can destabilize training.
```

### The Solution: Gradient Clipping

After computing gradients, scale them down if they're too large:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**What it does:**
1. Compute norm of all gradients: `total_grad_norm = sqrt(sum(g^2 for all g))`
2. If `total_grad_norm > max_norm`, scale: `gradients *= max_norm / total_grad_norm`

**Effect:**
- Gradients with small norm: unchanged
- Gradients with large norm: scaled down to the max norm

```
Before: [100, 200, 50]  (norm = sqrt(100^2 + 200^2 + 50^2) ≈ 223)
Scaling factor: 1.0 / 223 ≈ 0.0045
After: [0.45, 0.9, 0.23]  (norm = 1.0)
```

### Why This Matters

Without gradient clipping, training can become:
- Unstable (loss jumps around)
- Divergent (loss increases, gradients explode)
- Prone to NaN (invalid numbers)

With clipping:
- Stable training
- Smooth loss curves
- Robust convergence

---

## Part 6: Understanding the Loss Curve

### Expected Behavior

```
Step    Loss
0       5.2
100     3.8
200     2.9
300     2.4
400     1.9
500     1.8
600     1.85  ← Starting to level off
700     1.83
800     1.82  ← Convergence zone
...
```

### Stages of Training

```
Stage 1: Steep descent (Steps 0-1000)
         Model learning rapid fundamentals
         Loss drops quickly
         
Stage 2: Gradual descent (Steps 1000-3000)
         Refined learning
         Loss dropping slowly
         
Stage 3: Plateau (Steps 3000+)
         Diminishing returns
         Small loss improvements
         Eventually: overfitting (if training too long)
```

### Red Flags

**Problem: Loss increasing or oscillating**
```
Cause: Learning rate too high
       Gradients too large
       Unstable training
Solution: Lower learning rate
          Increase gradient clipping max_norm
```

**Problem: Loss decreasing very slowly**
```
Cause: Learning rate too low
       Model capacity too small
       Data too noisy
Solution: Increase learning rate
          Add more training data
          Increase model size
```

**Problem: Loss spikes to NaN**
```
Cause: Gradient explosion
       Numerical instability
       Bad learning rate
Solution: Lower learning rate
          Increase gradient clipping
          Check for bad data
```

---

## Part 7: Validation vs Training Loss

### Why We Track Both

```
Training loss:   Loss on training data (what we optimize)
Validation loss: Loss on held-out data (generalization measure)
```

### The Overfitting Problem

```
Training Loss:   0.5  ← Very low! Memorized data
Validation Loss: 3.2  ← Very high! Can't generalize

This is overfitting: Model memorized training data
but can't predict new examples.
```

### Healthy Training

```
Training Loss:   1.2
Validation Loss: 1.5  ← Only slightly higher

Gap between train and val should be small.
This means: Model is learning patterns, not memorizing.
```

### In Our Code

We don't explicitly compute validation loss here (could add it):

```python
# Could add:
if step % eval_interval == 0:
    # Compute training loss on a batch
    x_train, y_train = get_batch(batch_size, block_size, device)
    _, train_loss = model(x_train, y_train)
    
    # Compute validation loss (would need a separate validation set)
    x_val, y_val = load_validation_batch()
    _, val_loss = model(x_val, y_val)
    
    print(f"Step {step} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
```

---

## Part 8: Batch Size and Learning Rate

### Batch Size Effects

**Large batch (e.g., 256):**
- ✅ More stable gradients (average over more samples)
- ✅ Better GPU utilization
- ❌ Less frequent updates (more iterations for same data)
- ❌ May converge to sharper minima (generalize worse)

**Small batch (e.g., 4):**
- ✅ More frequent updates
- ✅ Better generalization (noisier gradients help escape bad local minima)
- ❌ Noisier loss curve (harder to debug)
- ❌ Less efficient GPU usage

**Our choice: batch_size = 4**

For a 50M model on small datasets, this works well. Larger models use larger batches (128, 256) for efficiency.

### Scaling Learning Rate

A rule of thumb:

```
learning_rate ~ sqrt(batch_size)

Small batch: 1e-4 to 1e-3
Large batch: 1e-3 to 1e-2
```

**Why?** Larger batches have more stable gradients, so you can afford larger steps.

Our choice: **learning_rate = 3e-4** with batch_size = 4.

If we increased batch_size to 64, we'd probably increase learning_rate to 1e-3 or higher.

---

## Part 9: Checkpointing and Resuming Training

### Saving Checkpoints

```python
if step % 1000 == 0 and step > 0:
    torch.save(model.state_dict(), f"checkpoint_{step}.pth")
    print(f"Saved checkpoint at step {step}")
```

**What `state_dict()` contains:**
All learnable parameters:
- Token embeddings
- Position embeddings
- All attention weights and biases
- All feedforward weights and biases
- Layer normalization parameters

### Resuming from Checkpoint

```python
# Load pre-trained weights
model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
model.load_state_dict(torch.load("checkpoint_1000.pth"))

# Continue training from step 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(1000, max_iters):  # Start from 1000
    # Training loop continues
```

### Saving the Final Model

```python
torch.save(model.state_dict(), "model.pth")
```

Later, during generation:

```python
model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set to evaluation mode
```

**Why `model.eval()`?** Disables dropout, batch norm variations. We want deterministic behavior.

---

## Part 10: Practical Training Tips

### Tip 1: Warm-up Learning Rate

Start with a small learning rate, gradually increase:

```python
if step < 500:  # Warm-up phase
    lr = learning_rate * (step / 500)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Why?** Large initial updates can overshoot good solutions.

### Tip 2: Learning Rate Schedules

Reduce learning rate as training progresses:

```python
# Exponential decay
lr = learning_rate * (0.99 ** step)

# Or step decay
if step % 1000 == 0:
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5  # Halve learning rate every 1000 steps
```

**Why?** Small learning rates in later training allow fine-tuning.

### Tip 3: Track Multiple Metrics

```python
metrics = {
    'loss': [],
    'grad_norm': [],
    'learning_rate': [],
}

# During training:
total_grad_norm = sum(p.grad.data.norm(2).item() for p in model.parameters())
metrics['grad_norm'].append(total_grad_norm)
```

Plot these to understand training dynamics.

### Tip 4: Validate Regularly

```python
if step % eval_interval == 0:
    model.eval()
    
    # Generate sample text
    sample = generate(model, sp, "The cat", max_tokens=20)
    print(f"Step {step}: {sample}")
    
    model.train()
```

Qualitative evaluation (does generated text make sense?) often matters more than loss alone.

---

## Part 11: Distributed Training (Brief Overview)

Our training is single-GPU/CPU. For larger models:

### Data Parallelism
```
Batch splits across multiple GPUs
Each GPU processes subset of batch
Gradients averaged across GPUs
```

### Model Parallelism
```
Model splits across multiple GPUs
Different layers on different devices
Requires careful communication
```

### Practical Result
Larger models train faster on multiple GPUs. Single-device training is simple but slower.

---

## Part 12: Common Training Issues and Solutions

| Issue | Symptom | Cause | Solution |
|-------|---------|-------|----------|
| Exploding gradients | Loss → NaN, Loss spikes | Learning rate too high | Increase clipping, lower LR |
| Vanishing gradients | Loss plateau, no improvement | Network too deep | Use residual connections (we do) |
| Overfitting | Val loss >> train loss | Train too long | Add early stopping, regularization |
| Underfitting | High train loss | Model too small | Increase model size, train longer |
| Slow convergence | Loss drops very slowly | Learning rate too low | Increase LR, larger batches |
| Unstable training | Loss oscillates | Batch too small, LR too high | Increase batch, lower LR |

---

## Part 13: Key Takeaways

### Training Process

1. **Forward pass** computes predictions
2. **Loss** measures prediction error (cross-entropy)
3. **Backward pass** computes gradients
4. **Gradient clipping** prevents explosion
5. **Optimizer** updates weights
6. **Repeat** until convergence

### Hyperparameter Tuning

- **Learning rate**: 1e-4 to 1e-2 (depending on batch size)
- **Batch size**: Balance between stability and efficiency
- **Max iters**: Until loss plateaus
- **Gradient clip**: Usually 1.0 works well

### Red Flags

- NaN in loss → Lower learning rate
- Loss not decreasing → Increase learning rate, check data
- Val loss >> train loss → Overfitting, stop training
- Loss decreasing very slowly → Model too small or LR too low

### Practical Tools

- Save checkpoints frequently
- Track multiple metrics (loss, grad norm, LR)
- Generate samples to evaluate quality
- Use validation set to detect overfitting
- Monitor hardware (GPU memory, compute)

---

## Next Steps

- Implement learning rate schedule
- Add validation set and track validation loss
- Experiment with different batch sizes and learning rates
- Monitor and visualize training metrics
- Implement early stopping
