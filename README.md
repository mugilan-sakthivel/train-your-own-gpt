# Train Your Own GPT — Learn Everything By Doing

A hands-on ladder from "what is a gradient" all the way to "I trained a GPT on my Mac."
Built for an **M2 Pro, 16 GB**. We *build* each piece, run it, and verify it — no black boxes.

This mirrors Andrej Karpathy's **"Neural Networks: Zero to Hero"**, tailored to your machine.

## Setup (done)
- Python 3.12 venv at `.venv` (created with `uv`)
- `torch` 2.12 + `numpy`, GPU (MPS) verified working

Run anything with:
```bash
.venv/bin/python <script.py>
```

## The ladder

| Stage | File | You learn | Status |
|-------|------|-----------|--------|
| 0 | `stage0_check.py` | GPU/MPS works | ✅ |
| 1 | `stage1_micrograd.py` | **Build autograd from scratch** — gradients & backprop become code you wrote | ▶ now |
| 2 | `stage2_makemore.py` | Char language model: embeddings + training loop in real code | ⬜ |
| 3 | `stage3_nanogpt.py` | **A real transformer (attention), trained on your GPU → generates text** | ⬜ |
| 4 | `stage4_yourdata.py` | Train on YOUR own text corpus; scale up | ⬜ |

## Honest hardware limits
- ❌ GPT-3 from scratch — impossible anywhere but a datacenter.
- 🟡 Full GPT-2 124M to original quality — needs cloud GPU to finish.
- ✅ Char-level GPT on TinyShakespeare — trains here in minutes, generates text.
- ✅ Everything in Stages 1–3 — comfortably on this Mac.
