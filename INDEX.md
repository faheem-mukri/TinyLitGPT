# TinyLitGPT: Complete Educational Documentation

Welcome to the educational documentation for TinyLitGPT—a transformer-based language model built from scratch for learning.

This documentation is organized as a complete learning journey from theory to practice.

---

## 📚 Documentation Structure

### Getting Started Path
1. **Start here:** Read `theory.md` first for conceptual understanding
2. **Then:** `tokenization.md` to understand text processing
3. **Deep dive:** `attention.md` for the core mechanism
4. **Training:** `training.md` to understand optimization
5. **Generation:** `sampling.md` for text generation control
6. **Reflect:** `observations.md` for practical insights
7. **Experiment:** `experiments.md` for hands-on learning

---

## 📖 File Guide

### 1. **theory.md** — Foundation (START HERE)
*Beginner-friendly conceptual introduction to transformers*

**Covers:**
- What is a language model?
- Problems transformers solve
- Self-attention mechanism intuition
- Multi-head attention
- Residual connections and layer norm
- Causal masking
- Why transformers scale
- Training to generation flow

**Best for:**
- First-time learners
- Building conceptual foundation
- Understanding the "why" before the "how"

**Time:** ~30-45 minutes to read thoroughly

**Key concepts:**
```
Language modeling → Attention → Transformers → Scaling
```

---

### 2. **tokenization.md** — Text Processing
*How raw text becomes numbers the model understands*

**Covers:**
- Why tokenization is necessary
- Character vs. word vs. subword tokenization
- Byte Pair Encoding (BPE) algorithm
- SentencePiece implementation
- Our 10k vocabulary
- Tokenizer training process
- Practical tokenization in our code
- Common tokenization problems

**Best for:**
- Understanding text encoding
- Debugging tokenization issues
- Learning about vocabulary trade-offs

**Time:** ~25 minutes

**Key insight:**
```
Text → Tokenizer → Token IDs → Embeddings → Model
```

---

### 3. **attention.md** — The Core Mechanism
*Deep dive into self-attention that makes transformers work*

**Covers:**
- Attention intuition with real examples
- Query, Key, Value matrices
- Scaled dot-product attention equation
- Mathematical breakdown step-by-step
- Causal masking for language modeling
- Multi-head attention design
- Why attention is powerful
- Attention in our code
- Debugging attention
- Performance considerations

**Best for:**
- Understanding attention mechanism thoroughly
- Learning the math with intuition
- Seeing code implementation

**Time:** ~40 minutes

**Core equation:**
```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

---

### 4. **training.md** — Learning Process
*How models learn through backpropagation and optimization*

**Covers:**
- What training does conceptually
- Cross-entropy loss function
- The full training loop
- Gradients and backpropagation
- Gradient clipping for stability
- Loss curves and interpretation
- Batch size and learning rate effects
- Validation vs. training loss
- Checkpointing and resuming
- Common training problems
- Tips and tricks

**Best for:**
- Understanding how models learn
- Debugging training issues
- Tuning hyperparameters
- Monitoring training progress

**Time:** ~35 minutes

**Training cycle:**
```
Forward → Loss → Backward → Optimize → Repeat
```

---

### 5. **sampling.md** — Text Generation
*How to control generation quality and diversity*

**Covers:**
- Autoregressive generation process
- Greedy vs. sampling decoding
- Temperature control (randomness)
- Top-K filtering
- Top-P (nucleus) sampling
- Repetition penalties
- Full generation pipeline in our code
- Quality metrics
- Controlling generation properties
- Common generation problems
- Advanced techniques (beam search, etc.)

**Best for:**
- Generating better text
- Understanding sampling strategies
- Controlling creativity vs. coherence
- Debugging generation issues

**Time:** ~30 minutes

**Key trade-off:**
```
Low temperature → Coherent, deterministic
High temperature → Diverse, creative
```

---

### 6. **observations.md** — Practical Insights
*Real lessons learned from actually building and training*

**Covers:**
- Dataset quality effects
- Model architecture trade-offs
- Training dynamics patterns
- Generation behavior observations
- Common failure modes and solutions
- Emergent behaviors the model learns
- Scaling laws
- Performance metrics
- Debugging techniques
- Hyperparameter sensitivity

**Best for:**
- Understanding practical considerations
- Learning from real implementation
- Debugging real problems
- Building intuition

**Time:** ~30 minutes

**Philosophy:**
```
Theory is a map; observations are the terrain
```

---

### 7. **experiments.md** — Hands-On Learning
*Concrete experiments to test and deepen understanding*

**Covers:**
- Tokenization experiments (inspect vocab, round-trip tests)
- Training experiments (LR comparison, batch size effects)
- Attention experiments (visualization, head analysis)
- Generation experiments (temperature, top-k, penalties)
- Architecture experiments (layer count, block size)
- Data and domain experiments
- Analysis experiments (position loss, error analysis)

**Best for:**
- Learning by doing
- Verifying theoretical concepts
- Building experimental intuition
- Discovering new insights

**Time:** Variable (depends on experiments)

**Approach:**
```
Hypothesis → Experiment → Observe → Learn
```

---

## 🎯 Learning Paths

### Path 1: Theory-First Learner
For people who like understanding before doing:

1. Read `theory.md` completely
2. Read `tokenization.md` completely  
3. Read `attention.md` completely
4. Read `training.md` completely
5. Read `sampling.md` completely
6. Skim `observations.md`
7. Run **Experiment 3.1** (attention visualization)
8. Run **Experiment 4.1** (temperature effect)

**Time:** ~4-5 hours

---

### Path 2: Learn-By-Doing Learner
For people who like to experiment:

1. Quick skim of `theory.md` (parts 1-2)
2. Run all **Tokenization Experiments (1.1-1.4)**
3. Run **Experiment 2.1** (learning rate)
4. Read `training.md` to understand results
5. Run **Experiment 3.1** (attention)
6. Read `attention.md` to understand results
7. Run **Experiment 4.1** (temperature)
8. Read `sampling.md` to understand results

**Time:** ~6-8 hours

---

### Path 3: Focused Deep-Dive
For people with specific interests:

**If interested in:** How models learn → Read `training.md` → Run Experiments 2.1-2.3

**If interested in:** Text encoding → Read `tokenization.md` → Run Experiments 1.1-1.4

**If interested in:** Attention mechanism → Read `attention.md` → Run Experiments 3.1-3.2

**If interested in:** Text generation → Read `sampling.md` → Run Experiments 4.1-4.3

---

## 🔍 Quick Reference

### Key Concepts by File

| File | Main Concept | Key Equation |
|------|--------------|--------------|
| theory.md | How transformers work | - |
| tokenization.md | Text → numbers | Text → Tokenizer → Token IDs |
| attention.md | Relevance weighting | Softmax(QK^T / √d) @ V |
| training.md | Learning via gradients | L = -log(p_correct) |
| sampling.md | Controlled generation | next_token ~ Softmax(logits/T) |
| observations.md | Practical lessons | Empirical findings |
| experiments.md | Hands-on testing | Hypothesis testing |

---

### Quick Question Answers

**Q: How do I make generations more creative?**
- A: See `sampling.md` → Increase temperature to 1.2-1.5

**Q: Why is my loss not decreasing?**
- A: See `training.md` → Check learning rate and gradient clipping

**Q: What does each attention head do?**
- A: See `attention.md` Part 4 + `experiments.md` Experiment 3.2

**Q: How does tokenization work?**
- A: See `tokenization.md` Part 3 for details

**Q: My generations are repetitive, how to fix?**
- A: See `sampling.md` Part 6 + `observations.md` Part 5

**Q: What's the difference between temperature and top-k?**
- A: See `sampling.md` Parts 3-4

**Q: How many layers should my model have?**
- A: See `observations.md` Part 2 + `experiments.md` Experiment 5.1

---

## 📚 Reading Depth Levels

### Level 1: Surface Understanding (1-2 hours)
Read: theory.md (parts 1-3), sampling.md (parts 1-2)

Learn: What is language modeling? How is attention used?

### Level 2: Practical Understanding (3-4 hours)
Read: All theory through sampling sections

Run: Experiments 1.1, 3.1, 4.1

Learn: How to use the model, basic architecture choices

### Level 3: Deep Understanding (6-8 hours)
Read: All documentation thoroughly

Run: All or most experiments

Learn: Design decisions, debugging, practical trade-offs

### Level 4: Expert Understanding (8+ hours)
Read: All documentation multiple times, reference source code

Run: Design and run custom experiments

Learn: Deep mathematical intuition, novel discoveries

---

## 🛠️ How to Use These Docs

### As a Reference
Each file is self-contained. Jump to what you need:
- Need attention math? → `attention.md` Part 2
- Having training issues? → `training.md` Part 10
- Want to tune generation? → `sampling.md` Part 3

### As a Study Guide
Follow a path above for systematic learning

### As Experiment Companion
Read theory, then run corresponding experiment

### As a Teaching Material
Share specific files with others learning transformers

---

## 💡 Tips for Maximum Learning

1. **Don't just read, trace through code**
   - Read a concept in the doc
   - Find that code in `model/gpt.py` or `generate.py`
   - Understand how theory → implementation

2. **Run experiments after reading**
   - Experiments verify what you learned
   - They build intuition that reading alone can't give

3. **Modify the code**
   - Change temperature values
   - Add print statements
   - Break things and fix them
   - This is how you really learn

4. **Take notes**
   - Write down key insights
   - Sketch diagrams
   - Record surprising findings

5. **Revisit concepts**
   - Come back to these docs after coding
   - Early readings build foundation
   - Later readings reveal deeper patterns

6. **Connect to practice**
   - Try the model on your own text
   - Generate completions for different prompts
   - Debug when it behaves unexpectedly

---

## 🎓 Learning Outcomes

After working through all this material, you'll understand:

✅ How transformers work from first principles
✅ How text is converted to numbers
✅ The attention mechanism in detail
✅ How models learn through backpropagation
✅ How to control text generation quality
✅ Practical trade-offs in model design
✅ Common problems and solutions
✅ How to experiment systematically
✅ How theory connects to implementation
✅ What emergent behaviors look like

---

## 📖 File Organization

```
docs/
├── theory.md           ← START HERE (conceptual foundation)
├── tokenization.md     ← Text encoding
├── attention.md        ← Core mechanism  
├── training.md         ← Learning process
├── sampling.md         ← Text generation
├── observations.md     ← Practical insights
├── experiments.md      ← Hands-on testing
└── INDEX.md (this file)
```

---

## 🤔 Frequently Asked Questions

**Q: In what order should I read these?**
- A: Start with theory.md, then follow the "Getting Started Path" above

**Q: Do I need to read everything?**
- A: No. Pick a learning path based on your interests

**Q: Should I run experiments while reading?**
- A: Ideally yes, but not required. Some prefer reading first, experimenting later

**Q: Can I skip some sections?**
- A: Yes, but theory.md is really the foundation. Others can be read selectively

**Q: How long does this take?**
- A: 4-8 hours depending on depth level and pace

**Q: Should I read the source code?**
- A: Yes, especially after reading relevant doc sections

**Q: Are there external resources I should read?**
- A: See specific docs for references. The original Transformer paper is excellent

---

## 🚀 After This Documentation

Once you've worked through these materials:

1. **Implement variations:**
   - Add new features to the model
   - Try different architectures
   - Experiment with new training techniques

2. **Scale up:**
   - Train on more data
   - Use larger models
   - Optimize for your hardware

3. **Apply downstream:**
   - Text classification
   - Summarization
   - Q&A systems

4. **Read papers:**
   - "Attention Is All You Need" (2017)
   - "Language Models are Unsupervised Multitask Learners" (2019)
   - Recent transformer improvements

5. **Contribute:**
   - Fix issues in the code
   - Add better experiments
   - Share insights

---

## 📝 Documentation Philosophy

These docs follow these principles:

1. **Beginner-friendly first:** Theory before math
2. **Practical emphasis:** Code > abstract theory
3. **No assumed knowledge:** Explain jargon  
4. **Real examples:** Not toy examples
5. **Connected theory:** All theory links to code
6. **Experimental validation:** Encourage verification
7. **Honest about complexity:** Don't oversimplify

---

## 🎯 Your Learning Journey

```
Start
  ↓
Read: theory.md ✓
  ↓
Read: tokenization.md ✓
  ↓
Read + Experiment: attention.md + Exp 3.1 ✓
  ↓
Read + Experiment: training.md + Exp 2.1 ✓
  ↓
Read + Experiment: sampling.md + Exp 4.1 ✓
  ↓
Read: observations.md ✓
  ↓
Run: experiments.md (choose your interests) ✓
  ↓
Modify code + build projects → Mastery
```

---

## 💬 Happy Learning!

These materials are designed to teach transformers thoroughly and practically. Take your time, run experiments, and most importantly—enjoy the process of understanding how these powerful models work.

The field of AI is rapidly evolving. Use this as a foundation to understand new papers and techniques as they emerge.

Good luck! 🚀
