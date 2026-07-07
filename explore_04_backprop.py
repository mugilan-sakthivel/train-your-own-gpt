"""
EXPERIMENT 4 — BACKPROP, step by step, on (w - 10)^2 at w = 4.

We break the error into its two real steps:
    d = w - 10        (subtract step,  local rate = 1)
    e = d * d         (square step,    local rate = 2*d)

Then we walk BACKWARD, multiplying local rates (the chain rule), to fill grads.
Goal: show backprop rebuilds the hand formula 2*(w-10) = -12, and matches PyTorch.

Run it:
    cd /Users/mugilansakthivel/Developer/learning/train-your-own-gpt
    .venv/bin/python explore_04_backprop.py
"""

w = 4.0

# ---------- FORWARD: walk the trail w -> d -> e ----------
d = w - 10        # step 1: subtract
e = d * d         # step 2: square  (this is the error)
print("FORWARD pass (compute the values):")
print(f"  w = {w}")
print(f"  d = w - 10 = {d}")
print(f"  e = d * d  = {e}   <- the error\n")

# ---------- BACKWARD: walk e -> d -> w, multiplying local rates ----------
print("BACKWARD pass (fill the grads by multiplying rates backward):")

e_grad = 1.0                      # seed: error vs itself
print(f"  e.grad = 1                                  (the seed)")

# square step: e = d*d, local rate of e w.r.t d is 2*d
rate_d_to_e = 2 * d
d_grad = rate_d_to_e * e_grad
print(f"  d.grad = (rate d->e) x e.grad = (2*{d}) x {e_grad} = {d_grad}")

# subtract step: d = w - 10, local rate of d w.r.t w is 1
rate_w_to_d = 1
w_grad = rate_w_to_d * d_grad
print(f"  w.grad = (rate w->d) x d.grad = ({rate_w_to_d}) x {d_grad} = {w_grad}")

print("\n" + "-" * 55)
print("CHECK — three ways to get the weight's grad, all should match:")
print(f"  1) backprop (above)        : {w_grad}")
print(f"  2) hand formula 2*(w-10)   : {2*(w-10)}")

import torch
tw = torch.tensor(4.0, requires_grad=True)
te = (tw - 10) ** 2
te.backward()
print(f"  3) PyTorch autograd        : {tw.grad.item()}")

print("\n✅ All three give -12. Backprop = the chain rule, walked backward.")
print("   The hand formula was never needed — the chain produced it automatically.")
