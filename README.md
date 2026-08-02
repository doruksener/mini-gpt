# GPT-From-Scratch (2018 Edition)

> *“No AI-generated code here. Just pure PyTorch tensors, matrix multiplications, late-night debugging, and raw 2018 transformer mathematics.”*

Welcome to my personal implementation of a Generative Pre-Trained Transformer (GPT), built completely from scratch. Inspired directly by OpenAI's June 2018 paper, *"Improving Language Understanding by Generative Pre-Training"*.

The goal of this repository is simple: **No high-level abstraction wrappers, no black boxes.** Just understanding how text turns into tokens, how queries, keys, and values dance through multi-head attention, and how an autoregressive model learns to whisper the next word.

---

## 🛠️ The Blueprint (What We Are Building)

Instead of relying on modern bloated libraries, we are coding the core components manually, step-by-step, just like we would have in a terminal back in 2018:

1. **The Tokenizer & Vocabulary**: Mapping raw characters/words from our `input.txt` corpus into discrete integer IDs and back.
2. **Embeddings & Positional Encodings**: Giving the model a sense of meaning and word order in continuous vector space.
3. **Scaled Dot-Product Attention**: The mathematical core of the transformer block:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
4. **Masked Multi-Head Self-Attention**: Ensuring the model can't "cheat" by looking at future tokens during autoregressive training.
5. **Position-wise Feed-Forward Networks & LayerNorm**: Adding non-linearity and stabilizing gradients across deep blocks.

---

## 📂 Project Structure

```text
.
├── README.md       # You are here (the developer's log)
├── model.py        # Hand-crafted PyTorch Module classes for GPT layers & attention
├── train.py        # Custom data loader, loss computation, and optimization loop
└── input.txt       # Raw training corpus (Shakespeare?? or local text file)
