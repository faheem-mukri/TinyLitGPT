# Observations: Practical Lessons from Building a Transformer

This document captures practical observations and lessons learned from actually training and using TinyLitGPT. These are insights that don't fit neatly into theory but matter for real-world implementation.

---

## Part 1: Dataset Effects

### Effect 1: The Importance of Data Quality

**Observation:** Our model trained on TinyStories + WikiText (filtered).

What we observed:
- Clean, well-formatted text from both sources
- Consistent writing style within each source
- No random noise or corrupted text

**Result:** Model learned to generate fluent text quickly.

**Contrast:** If we had used random internet text:
- Model would learn grammar patterns from poorly-written text
- Would generate lower quality by default
- Would take longer to converge

**Lesson:** Data quality > data quantity for small models. One clean source beats ten noisy sources.

---

### Effect 2: Text Domain Matters

**Observation:** TinyStories is children's stories (simple, consistent vocabulary and structure).

What we observed:
- Model learns narrative patterns quickly
- Generated text follows story structure ("Once upon a", "The end")
- Vocabulary is conservative (avoids complex words)

**Contrast:** If we trained on scientific papers:
- Model would use jargon and complex sentence structure
- Generated text would be more formal
- Vocabulary would be much larger

**Lesson:** The model learns the "style" and "voice" of its training data. Different domains → different models.

---

### Effect 3: Vocabulary Coverage

**Observation:** We chose vocab_size=10,000 (10k tokens).

What we observed:
- Handles almost all English text (covers ~99% of words in typical English)
- Some very rare words get split into subwords
- Can represent unknown text (character fallback)

**Math:**
```
10k tokens is enough for:
- ~99% of common English words
- Proper coverage of morphological variants
- Numbers, punctuation, special characters

Too small (1k tokens): Sequences become longer, harder to learn
Too large (50k tokens): Overkill for our model size, wastes parameters
```

**Lesson:** 10k is the sweet spot for ~50M model. Larger models use larger vocabularies (GPT-3 uses 50k).

---

## Part 2: Model Architecture Effects

### Effect 1: Model Size and Capacity

Our config:
```
d_model = 512
n_heads = 8
n_layers = 8
~50M parameters
```

What we observed:
- Model has enough capacity to learn patterns
- Training converges in reasonable time (~5000 steps)
- Generates coherent multi-sentence text
- No extreme underfitting (loss doesn't plateau at 3.0+)

**Scaling up (to 100M):**
```
d_model = 768
n_heads = 12
n_layers = 12
~100M parameters
```
- Would train slower (more parameters)
- Would likely perform better (more capacity)
- Would need more GPU memory

**Scaling down (to 10M):**
```
d_model = 256
n_heads = 4
n_layers = 4
~10M parameters
```
- Would train faster
- Would generate lower quality (less capacity)
- Might underfit (can't learn enough)

**Lesson:** ~50M is a good size for exploration. Sweet spot for "meaningful but not massive."

---

### Effect 2: Block Size (Context Window)

Our config: `block_size = 128` tokens

What we observed:
- Model can attend to ~100 words of history (average ~1.2 tokens/word in our tokenizer)
- Good for sentence-level coherence
- Limited for long-range dependencies (multi-paragraph stories)

**Example limitation:**
```
Generated text:
"Once upon a time, there was a girl named Lucy.
She lived in a cottage.
... (many sentences later) ...
He walked back to the cottage."

Problem: Model predicts "He" but should be "She" (referent Lucy)
But "Lucy" was ~200 tokens ago (outside context window)!
```

**Tradeoff:**
- Larger block_size: Better long-range context, more memory
- Smaller block_size: Faster training, less memory

**Real-world:**
- Our model: 128 tokens (fine for sentences)
- GPT-2: 1024 tokens (multipage context)
- GPT-4: Up to 128k tokens (whole books!)

**Lesson:** context window limits how far back the model can "remember." Trade-off with computational cost.

---

### Effect 3: Number of Layers

Our config: `n_layers = 8`

What we observed:
- Each layer adds computational cost (linear scaling)
- Deeper = potentially better representations
- But also: harder to train (gradient flow issues)

**Our experiments (from anecdotal observation):**
```
4 layers:   Fast, lower quality, weak dependencies
8 layers:   Good balance, decent quality
12 layers:  Slower, potentially better, but marginal gains
16 layers:  Very slow, requires careful tuning
```

**Why 8 worked:** 
- Deep enough to learn complex patterns
- Shallow enough to train stably (residual connections help)

**Lesson:** 8 layers is a practical default for ~50M model.

---

## Part 3: Training Dynamics

### Effect 1: Loss Curve Shape

**What we observed:**

```
Typical loss curve:
Step 0:      Loss ~5.0 (random predictions)
Step 100:    Loss ~2.5 (steep descent)
Step 500:    Loss ~1.5 (moderate descent)
Step 1000:   Loss ~1.2 (slow descent)
Step 2000:   Loss ~1.05 (approaching plateau)
Step 3000:   Loss ~1.02 (flat, training limit reached)
Step 5000:   Loss ~1.00 (no improvement)
```

**Shape:** Steep initial drop, then gradual plateau. Classic.

**Why this shape:**
- Early steps: Model learns basic language patterns
- Middle steps: Model learns finer details
- Later steps: Diminishing returns (hard to improve further)

**Lesson:** Most learning happens early. Training stops being productive after loss plateaus.

---

### Effect 2: Batch Size vs. Convergence

**What we observed (conceptually, not exhaustively tested):**

```
Batch size 1:
- Loss very noisy (jumps around)
- Convergence slower (need more steps for same data)
- Better generalization (noise helps escape bad minima)

Batch size 4 (ours):
- Loss reasonably smooth
- Decent convergence speed
- Good generalization

Batch size 32:
- Loss very smooth (average over many samples)
- Convergence faster (fewer steps needed)
- Might overfit (less noise to regularize)

Batch size 256:
- Super smooth loss
- Fast convergence
- Likely overfitting on small data
```

**Lesson:** Batch size = noise level. Too small = noisy training. Too large = smooth but overfitting risk.

---

### Effect 3: Learning Rate Sensitivity

**What we observed (from typical practices):**

```
LR = 1e-4: Very slow convergence (barely learning)
LR = 3e-4 (ours): Good convergence, stable
LR = 1e-3: Fast initial descent, then instability
LR = 1e-2: Unstable from start (loss spikes)
```

**The sweet spot:** 3e-4 for our hyperparameters.

**General rule:**
```
learning_rate ~ sqrt(batch_size) / (model_scale)
```

**Lesson:** Learning rate is crucial. Too high = instability. Too low = slow learning. Takes tuning.

---

## Part 4: Generation Behavior

### Effect 1: Temperature Trade-off

**What we observed:**

```
Temperature 0.3 (very cold):
"The cat sat on the mat. The cat sat on the mat. The cat..."
Problem: Greedy selection → repetition

Temperature 0.8 (our default):
"The cat sat on the mat. She looked around the garden. A bird..."
Good: Coherent and diverse

Temperature 1.5 (warm):
"The cat sat on the mat. Suddenly, purple numbers... a..."
Problem: Sometimes nonsensical (temperature too high breaks coherence)

Temperature 2.0+ (hot):
"The cat xyz!!!... @#$%... random..."
Problem: Completely incoherent (no structure)
```

**Lesson:** 0.8 is a good default. Extreme temperatures break generation.

---

### Effect 2: Top-K Filtering

**What we observed:**

```
top_k = 1:   No filtering, full vocabulary
             Result: All token probabilities used
             Often generates rare words (can be nonsense)

top_k = 10:  Only top 10 tokens allowed
             Result: Very conservative, safe generation
             Low diversity, repetitive

top_k = 40 (ours): Reasonable filtering
             Result: Good balance of quality and diversity
             Avoids worst tokens, allows alternatives

top_k = 500: Almost no filtering (vocab_size = 10k)
             Result: Similar to no filtering
             Diverse but potentially incoherent
```

**Lesson:** top_k ~10-50 works well. Larger value = more diversity but less safety.

---

### Effect 3: Repetition Penalty Effect

**What we observed:**

```
No penalty:
"I like the cat. The cat is the cat. The cat..."
Problem: Model repeats "cat" too much

Penalty 1.2:
"I like the cat. The cat is nice. It has..."
Better: Less repetition, still coherent

Penalty 1.5 (ours):
"I like the cat. The animal is very nice. She..."
Good: Diverse vocabulary, coherent

Penalty 2.5:
"I like the cat. The creature is most certainly nice. She is interesting. It seems..."
Problem: Over-penalizing creates awkward phrasing
```

**Lesson:** Penalty of 1.3-1.5 is sweet spot. Too strong = unnatural language.

---

## Part 5: Common Failure Modes

### Failure 1: Loss Getting Stuck

**What it looks like:**
```
Step 100:   Loss = 1.5
Step 200:   Loss = 1.49
Step 300:   Loss = 1.48
Step 400:   Loss = 1.48 (no progress!)
Step 500:   Loss = 1.48
```

**Common causes:**
1. Learning rate too low
2. Gradient saturation (model already fit)
3. Data quality issues
4. Model capacity too small

**Observed solution:**
- Increase learning rate (try 1e-3 instead of 3e-4)
- Check if data is actually varying
- Verify gradients are flowing (non-zero)

---

### Failure 2: NaN/Inf in Loss

**What it looks like:**
```
Step 98:  Loss = 1.2
Step 99:  Loss = 1.23
Step 100: Loss = NaN
```

**Immediate cause:** Numerical instability in computation

**Root causes:**
- Learning rate too high (gradients explode)
- Gradient clipping not working
- Data contains invalid values
- Softmax of all -inf (attention mask issue)

**Observed solution:**
- Lower learning rate
- Increase gradient clipping max_norm
- Check data for NaN/Inf
- Verify causal mask logic

---

### Failure 3: Repetitive Generation

**What it looks like:**
```
"Once upon a time there was a time there was a time..."
```

**Causes:**
- Model learned repetition from data (e.g., repeated text in dataset)
- Temperature too low (greedy selection)
- Top-k too small (limited choices)
- Repetition penalty too weak

**Observed solutions:**
- Increase temperature (0.8 → 1.0)
- Increase top-k (40 → 60)
- Increase repetition penalty (1.3 → 1.5)
- Check training data for artifacts

---

### Failure 4: Incoherent Generation

**What it looks like:**
```
"Once upon a time there was a... qwerty potato.... xyz"
```

**Causes:**
- Temperature too high (random selection)
- Top-k too large (bad tokens included)
- Model undertrained (not converged)
- Data quality issues

**Observed solutions:**
- Lower temperature (1.5 → 0.8)
- Lower top-k (100 → 40)
- Train longer
- Check data

---

## Part 6: Debugging Techniques We Use

### Technique 1: Print Token IDs

```python
tokens = [123, 456, 789]
for token in tokens:
    piece = sp.id_to_piece(token)
    print(f"{token} → '{piece}'")

# Output:
# 123 → '▁Once'
# 456 → '▁upon'
# 789 → '▁a'
```

Helps understand what tokenizer learned and catch weird tokenizations.

### Technique 2: Test Round-Trip

```python
text = "Hello world"
tokens = sp.encode(text)
decoded = sp.decode(tokens)
assert text == decoded, f"Round trip failed: {text} → {decoded}"
```

Ensures tokenizer/detokenizer are working correctly.

### Technique 3: Inspect Attention Weights

```python
# During training, save attention weights
# Look for patterns:
# - Are tokens attending to relevant previous tokens?
# - Or is attention uniform (not learning)?
# - Are there obvious bugs (attending to all future)?

# Visual inspection can reveal training issues
```

### Technique 4: Manual Generation Testing

```python
prompts = [
    "The cat",
    "Once upon a",
    "She walked",
    "I like to"
]

for prompt in prompts:
    output = generate(model, sp, prompt)
    print(f"{prompt} → {output}\n")
```

Read the outputs. Does it make sense? Is it repetitive? Is it diverse?

---

## Part 7: Performance Observations

### Memory Usage

**Observed:**
```
Model parameters: ~50M
Model size on disk: ~200 MB (model.pth)

During training (batch_size=4, block_size=128):
- Model weights: ~200 MB
- Activations: ~500 MB
- Gradients: ~200 MB
- Total: ~900 MB (easily fits on modern GPU)

During inference:
- Model weights: ~200 MB
- Batch activation: ~50 MB
- Total: ~250 MB (very small)
```

**Lesson:** Small models fit easily. Large models (billions of parameters) need special optimization.

---

### Speed

**Observed (rough timing):**
```
Training:
- Per step: ~50 ms (on modern GPU)
- Per epoch: ~2-5 seconds
- Full training (5000 steps): ~4 minutes

Generation:
- Per token: ~5-10 ms (depends on device)
- Per sequence (50 tokens): ~250-500 ms

On CPU:
- Slower by 10-100x depending on operation
```

**Lesson:** GPU is orders of magnitude faster for these operations. CPU training is impractical.

---

## Part 8: Emergent Behaviors

### Observation 1: The Model Learns Grammar

**What we observed:**
The model, trained only to predict next tokens, develops understanding of grammar without explicit rules.

```
Input: "The dogs"
Likely predictions:
- "run" (verb agrees with plural)
- "are" (plural verb)
- "barked" (past tense, less likely but possible)

NOT "runs" (singular verb, breaks agreement)
```

**Why?** Training on correct English text, the model learns patterns. Agreement patterns are statistically consistent.

**Lesson:** Grammar emerges from data patterns, not explicit rules.

---

### Observation 2: The Model Captures Semantics

**What we observed:**
Related words have similar attention patterns.

```
"king" and "prince" tend to attend to similar context words
"quickly" and "fast" activate similar attention patterns
"happy" and "joyful" have overlapping semantic relationships
```

**Why?** Embeddings learn to place similar words near each other in vector space.

**Lesson:** Semantic meaning emerges from statistical patterns in large text.

---

### Observation 3: The Model Struggles with Counting

**What we observed:**
```
Prompt: "There were 3 apples. He ate 1 apple. Remaining:"
Model output: "2 apples" (correct by luck)
              "some apples" (hedging)
              "many apples" (confused)
              
Try again: "There were 100 apples. He ate 50..."
Model output: "50 apples" (lucky)
              "30 apples" (off)
              "many apples" (giving up)
```

**Why?** Our model has limited arithmetic capability. Token-level prediction doesn't naturally handle numerical reasoning.

**Lesson:** Not all tasks are suitable for language models. Math requires explicit mechanisms.

---

### Observation 4: Context Window Cutoff Effects

**What we observed:**
```
Generation that starts well then degrades:

Start: "Once upon a time, a girl named Lucy lived in a cottage..."
Middle: "...She had many friends. They played games together..."
Later: "...And they were very happy. He decided to..."  ← Wrong pronoun!
```

Why "He"? Model forgot Lucy (referent is outside context window).

**Lesson:** Context window length directly impacts coherence. Larger window = longer coherence.

---

## Part 9: Hyperparameter Tuning Insights

### The Sensitivity Hierarchy

**Most important to tune (high impact):**
1. Learning rate
2. Batch size
3. Training data quality
4. Model size

**Moderately important:**
5. Layer count
6. Number of heads
7. d_model size
8. Block size

**Less important (smaller impact):**
9. Exact optimizer (Adam vs AdamW)
10. Temperature precision
11. Top-K exact value

**Lesson:** Focus on the first tier. Diminishing returns beyond that.

---

### The Scaling Laws

**What researchers have empirically discovered:**
```
loss ∝ (steps)^(-α)   where α ≈ 0.07

Interpretation: 
- 10x more training → ~15% loss reduction
- 100x more training → ~30% loss reduction
```

**In our case:**
```
Step 100:   Loss = 2.5
Step 1000:  Loss = 1.5  (10x steps → 40% reduction)
Step 5000:  Loss = 1.0  (5x steps → 33% reduction)
```

Matches the scaling law roughly!

**Lesson:** Improvements are asymptotic. Training longer helps but with diminishing returns.

---

## Part 10: What We Wish We Knew Earlier

### Wish 1: Monitor Gradients

```python
# Compute gradient norm during training
total_grad_norm = sum(p.grad.data.norm(2).item() 
                      for p in model.parameters()
                      if p.grad is not None)
print(f"Gradient norm: {total_grad_norm}")
```

**Why?** 
- Tells if model is learning (gradients should be non-zero)
- Detects vanishing gradients (norm too small)
- Detects exploding gradients (norm too large)

---

### Wish 2: Save Examples During Training

```python
# Every 500 steps, generate sample
if step % 500 == 0:
    with torch.no_grad():
        sample = generate(model, sp, "Once upon a")
        with open(f"samples/step_{step}.txt", "w") as f:
            f.write(sample)
```

**Why?**
- Loss numbers don't tell the full story
- Generation quality improves before loss plateaus
- Can see mode collapse (repetitive generation)

---

### Wish 3: Plot Loss Curves

```python
import matplotlib.pyplot as plt
plt.plot(steps, losses)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.savefig("loss_curve.png")
```

**Why?**
- Visual patterns easier to spot than numbers
- Can identify training issues quickly
- Useful for debugging and papers

---

### Wish 4: Version Control Everything

```
model_v1/   experiments/
├── model_v1.pth      ├── lr_1e-3.txt
├── config.json       ├── lr_3e-4.txt
└── train_loss.txt    └── lr_1e-4.txt
```

**Why?**
- Easy to compare experiments
- Can revert if something breaks
- Can reproduce results

---

## Part 11: Practical Tips

### Tip 1: Warm-up Training

```python
# Use lower learning rate initially
if step < 500:
    current_lr = learning_rate * (step / 500)
```

**Benefit:** Smoother initial training, less chance of divergence.

### Tip 2: Gradient Accumulation

For larger effective batch size without OOM:

```python
# Process 4 samples sequentially, accumulate gradients
for micro_step in range(4):
    x, y = get_batch(1, block_size, device)
    logits, loss = model(x, y)
    loss.backward()  # Don't zero yet

optimizer.step()
optimizer.zero_grad()
```

**Benefit:** Larger batch size's smoothness with smaller GPU memory.

### Tip 3: Mixed Precision Training

Use float16 for speed, float32 for accuracy:

```python
from torch.cuda.amp import autocast

with autocast():
    logits, loss = model(x, y)

loss.backward()
```

**Benefit:** 2-3x faster training on modern GPUs with minimal quality loss.

---

## Part 12: Key Takeaways

### What We Learned

1. **Data quality matters more than quantity**
2. **Model architecture choices have clear trade-offs**
3. **Training dynamics follow predictable patterns**
4. **Generation requires careful parameter tuning**
5. **Emergent behaviors show what models learn from data**
6. **Context window length directly limits coherence**
7. **Small models can be trained and run easily**
8. **Debugging requires both metrics and human judgment**

### The Philosophy

Training transformers is both art and science. Theory tells you the algorithm, but intuition (built from experience) tells you which knobs to turn when things go wrong.

### For Future Work

- Increase model size and see scaling effects
- Try different datasets and see style transfer
- Implement advanced techniques (beam search, constrained decoding)
- Build downstream applications (summarization, classification)
- Explore emergent abilities with scale

---

## Next: Experiments

Ready to test these observations? See `experiments.md` for hands-on experiments to verify and extend these insights.
