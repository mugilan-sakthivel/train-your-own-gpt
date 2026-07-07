"""
STAGE 1 — micrograd: build the autograd engine yourself.

This is the single most important file for understanding gradients & backprop.
PyTorch does exactly this internally, just faster and on tensors. Here we do it
on single numbers so you can SEE every step.

Big idea:
  - Every number is wrapped in a `Value` that remembers HOW it was computed.
  - That history forms a graph (the "computation graph").
  - `backward()` walks the graph in reverse, applying the chain rule, to fill in
    `.grad` on every Value = "how much does the final loss change if I nudge this?"
  - That `.grad` IS the gradient. Gradient descent = nudge each value against its grad.

Run:  .venv/bin/python stage1_micrograd.py
"""

import math
import random


# ───────────────────────────────────────────────────────────────────────────
# 1. The Value: a number that remembers how it was made, and can back-propagate.
# ───────────────────────────────────────────────────────────────────────────
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data            # the actual number
        self.grad = 0.0             # dLoss/dThis — starts at 0, filled by backward()
        self._backward = lambda: None   # how to push grad to my inputs (set per op)
        self._prev = set(_children)     # the Values that produced me
        self._op = _op                  # what op made me (for debugging/printing)

    # --- the operations. Each one defines its LOCAL derivative (the chain rule piece) ---

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  → grad flows straight through, unchanged
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a  → each input gets the OTHER's value
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # d(a^k)/da = k * a^(k-1)
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh)/dx = 1 - tanh(x)^2   (this is the "activation" non-linearity)
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # --- conveniences so we can write normal math ---
    def __neg__(self):        return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ── THE HEART: reverse-mode autodiff ──────────────────────────────────────
    def backward(self):
        # 1) topological order: every node appears after all nodes that feed it
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # 2) seed: the loss's gradient w.r.t. itself is 1
        self.grad = 1.0
        # 3) walk in reverse, each node pushes grad to its inputs via the chain rule
        for v in reversed(topo):
            v._backward()


# ───────────────────────────────────────────────────────────────────────────
# 2. A tiny neural net built ONLY from Value (no PyTorch). Same neuron as the
#    HTML lesson: weighted sum + bias → tanh.
# ───────────────────────────────────────────────────────────────────────────
class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # Σ w·x + b
        return act.tanh()                                          # activation

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, sizes):  # e.g. [3, 4, 4, 1]
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ───────────────────────────────────────────────────────────────────────────
# 3. Train it. Tiny dataset, 4 examples, binary targets (+1 / -1).
#    This is the full Part-1 loop: predict → loss → backward → step.
# ───────────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)

    xs = [[2.0, 3.0, -1.0],
          [3.0, -1.0, 0.5],
          [0.5, 1.0, 1.0],
          [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]   # the answers we want

    model = MLP([3, 4, 4, 1])     # 3 inputs → 4 → 4 → 1 output
    n_params = len(model.parameters())
    print(f"Model has {n_params} parameters (weights + biases) — all built from scratch.\n")

    lr = 0.05
    for step in range(100):
        # --- forward: predict every example ---
        preds = [model(x) for x in xs]
        # --- loss: mean squared error (one number = total wrongness) ---
        loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0))

        # --- backward: zero old grads, then fill .grad on EVERY parameter ---
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # --- gradient descent: nudge each parameter downhill ---
        for p in model.parameters():
            p.data += -lr * p.grad      # the minus sign = step AGAINST the gradient

        if step % 10 == 0 or step == 99:
            print(f"step {step:3d}  loss = {loss.data:.6f}")

    print("\nFinal predictions vs targets:")
    for x, y in zip(xs, ys):
        print(f"  pred {model(x).data:+.3f}   target {y:+.1f}")

    # ───────────────────────────────────────────────────────────────────────
    # 4. PROOF our hand-built gradient is correct: compare to PyTorch.
    # ───────────────────────────────────────────────────────────────────────
    print("\n--- Gradient check vs PyTorch (same tiny expression) ---")
    # expression:  L = tanh(a*b + c)^2,  with a=1.5, b=-2.0, c=0.5
    a = Value(1.5); b = Value(-2.0); c = Value(0.5)
    L = (a * b + c).tanh() ** 2
    L.backward()
    print(f"micrograd grads:  a={a.grad:.6f}  b={b.grad:.6f}  c={c.grad:.6f}")

    import torch
    ta = torch.tensor(1.5, requires_grad=True)
    tb = torch.tensor(-2.0, requires_grad=True)
    tc = torch.tensor(0.5, requires_grad=True)
    tL = torch.tanh(ta * tb + tc) ** 2
    tL.backward()
    print(f"pytorch   grads:  a={ta.grad:.6f}  b={tb.grad:.6f}  c={tc.grad:.6f}")
    print("\n✅ If those two lines match, you just reimplemented PyTorch's autograd.")


if __name__ == "__main__":
    main()
