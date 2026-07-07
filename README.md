# Train Your Own GPT

Building a GPT from zero on my own GPU — no shortcuts.

This is a learning-in-public repo. The goal is to understand large language models
by building every layer myself, in code I can read and run, instead of calling an
API and trusting the magic. It follows the ladder Andrej Karpathy lays out in
**"Neural Networks: Zero to Hero"**, tuned for the machine I actually have: an
**Apple M2 Pro, 16 GB**, running PyTorch on the MPS (Metal) backend.

The rule for the whole repo: build the piece, run it, and verify it against a known
answer before moving on. Stage 1 already checks its hand-built autograd against
PyTorch and gets identical gradients.

## Why

Most "learn LLMs" material stops at the API call. You import a client, send a
prompt, get text back, and the actual machinery — gradients, backprop, attention,
the training loop — stays a black box. I wanted the opposite: write the autograd
engine by hand so backprop is code I typed, not a function I called. Understand it
from the number up, then let PyTorch do the same thing faster once I know what it's
doing.

Doing it on a 16 GB laptop is a deliberate constraint. It forces honesty about what
actually fits in memory and what needs a datacenter, and it keeps every experiment
small enough to run in seconds and inspect by hand.

## The ladder

| Stage | File | What it teaches | Status |
|-------|------|-----------------|--------|
| 0 | `stage0_check.py` | PyTorch sees the Apple GPU (MPS) and does real work | Done, verified |
| 1 | `stage1_micrograd.py` | Build an autograd engine from scratch — gradients & backprop as code you wrote | Done, verified |
| 2 | `stage2_makemore.py` | Char-level language model: embeddings, a bigram/MLP model, a real training loop | Planned |
| 3 | `stage3_nanogpt.py` | A real transformer with self-attention, trained on the GPU, generating text | Planned |
| 4 | `stage4_yourdata.py` | Train on my own text corpus and scale up as far as the machine allows | Planned |

Stages 2–4 are not written yet. The `names.txt` dataset (32k names) is already in
the repo, staged for makemore. I'd rather list them honestly as planned than pretend
they exist.

## What I've built so far

### Stage 0 — the GPU check

A ten-line sanity script: confirm `torch.backends.mps.is_available()`, then run
twenty 2000×2000 matmuls on the GPU and time them. On this M2 Pro that's ~0.37s.
It exists so that every later stage starts from a known-good device.

### Stage 1 — micrograd, an autograd engine from scratch

This is the core of the repo so far. `stage1_micrograd.py` builds reverse-mode
automatic differentiation on single numbers — the same thing PyTorch does on
tensors, just slow enough to watch.

- **`Value`** wraps a number and remembers how it was made: its inputs and the
  operation that produced it. Doing math on `Value`s builds a computation graph.
- Each operation (`+`, `*`, `**`, `tanh`) defines its own local derivative — the
  chain-rule piece. `backward()` topologically sorts the graph, seeds the output
  gradient at 1, and walks it in reverse, letting each node push gradient to its
  inputs. After that, every `Value.grad` holds "how much does the loss change if I
  nudge this number."
- On top of `Value` there's a tiny neural net built with **no PyTorch at all** —
  `Neuron`, `Layer`, `MLP` — a `[3, 4, 4, 1]` multilayer perceptron (41 parameters).
- It trains for 100 steps on a 4-example toy dataset with the full loop: forward →
  mean-squared-error loss → zero grads → `backward()` → step each parameter against
  its gradient. Loss drops and the predictions land on their targets.
- **The proof:** it then differentiates `tanh(a*b + c)^2` by hand and with PyTorch
  and prints both. They match to six decimals:

  ```
  micrograd grads:  a=0.104945  b=-0.078709  c=-0.052473
  pytorch   grads:  a=0.104945  b=-0.078709  c=-0.052473
  ```

  If those two lines agree, the hand-built autograd is correct.

### The `explore_*` scripts

Five small companion scripts that isolate one idea each, so a concept can be poked
at on its own before it's buried inside a bigger file:

- `explore_01_value.py` — a `Value` is a number that remembers how it was made; see
  the computation graph form.
- `explore_02_two_boxes.py` — watch a weight and its error co-evolve as training
  drives the error toward zero.
- `explore_03_bytes.py` — how many bytes a weight takes in fp32 / fp16 / bfloat16 /
  int8, and what that means for "will this model fit in 16 GB?"
- `explore_04_backprop.py` — backprop on `(w - 10)^2` step by step, rebuilding the
  hand formula `2*(w-10)` and matching PyTorch.
- `explore_05_network_forward.py` — the forward pass through a 13-parameter net with
  real numbers, ReLU activations, all by hand.

### The HTML lessons

Two self-contained pages I wrote to explain the ideas visually before coding them:
`course.html` (the Zero-to-Hero course notes) and `backprop-animated.html` (an
animation of forward filling `data` and backward filling `grad`). Open either
directly in a browser.

## Hardware notes

- **Machine:** Apple M2 Pro, 16 GB unified memory.
- **Backend:** PyTorch 2.12 on MPS (Metal). Stage 0 confirms MPS is available and
  actually runs work on the GPU; `torch.mps.synchronize()` is needed before timing,
  since MPS work is queued asynchronously.
- **What this machine can honestly do:**
  - Everything in Stages 1–3 runs comfortably here.
  - A char-level GPT on TinyShakespeare should train in minutes and generate text.
  - Full GPT-2 124M to original quality would need a cloud GPU to finish — the
    16 GB machine can start it but not take it all the way.
  - GPT-3 from scratch is a datacenter job, not a laptop one. The repo is about
    understanding the architecture, not matching frontier scale.

## Following along

Everything runs through the repo's own virtual environment (Python 3.12 + PyTorch),
created with [`uv`](https://github.com/astral-sh/uv). The `.venv` is not checked in;
recreate it, then run any script directly:

```bash
# from the repo root
uv venv                       # create .venv (or: python3.12 -m venv .venv)
uv pip install torch numpy    # torch pulls in the MPS build on Apple Silicon

# Stage 0 — is the GPU there?
.venv/bin/python stage0_check.py

# Stage 1 — build autograd, train the toy net, and check grads against PyTorch
.venv/bin/python stage1_micrograd.py

# poke at a single idea in isolation
.venv/bin/python explore_01_value.py
```

Every script prints what it's doing and ends with a check you can read, so you can
tell it worked rather than assuming it did.

---

Built and documented in the open by **Mugilan Sakthivel** — [mugilans.in](https://mugilans.in).
