# Contributing to TinyLitGPT

Thank you for your interest in improving TinyLitGPT! We welcome contributions that make this educational transformer model more efficient, clear, or feature-rich.

## How Can I Help?
* **Bug Fixes**: Correcting errors in the attention mechanism, loss calculation, or generation scripts.
* **Feature Additions**: Implementing modern LLM techniques (e.g., RMSNorm, GQA, Rotary Embeddings, or FlashAttention).
* **Documentation**: Enhancing explanations in the code to help others learn how transformers work.

## Development Setup

To make sure changes do not conflict with the existing PyTorch environment, please set up your workspace as follows:

1. **Fork and Clone** the repository:
   ```bash
   git clone https://github.com
   cd TinyLitGPT
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Pull Request Guidelines

Before submitting a Pull Request, please ensure your contribution adheres to these guidelines:
* **Code Style**: Follow standard Python formatting. Keep variable names closely tied to the mathematical concepts they represent (e.g., `q`, `k`, `v` for attention heads).
* **No Unnecessary Dependencies**: TinyLitGPT aims to be minimal and educational. Avoid adding heavy external libraries unless absolutely necessary.
* **Test Your Changes**: Verify that your modifications do not break the training pipeline or cause tensor shape mismatches during inference. Run a quick, small-scale training run to ensure loss still converges.
