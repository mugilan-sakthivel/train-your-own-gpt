"""
EXPERIMENT 1  —  "A Value is a number that remembers how it was made."

We reuse the SAME Value class from stage1_micrograd.py.
Goal: SEE that when you do math on Values, each result secretly stores
      the inputs and the operation that produced it. That stored history
      is called the "computation graph".

Run it:
    train-your-own-gpt/.venv/bin/python train-your-own-gpt/explore_01_value.py
"""

from stage1_micrograd import Value

print("=" * 60)
print("STEP 1 — make two plain numbers, but wrapped in Value")
print("=" * 60)
a = Value(3.0)
b = Value(4.0)
print("a =", a)          # Value(data=3.0000, grad=0.0000)
print("b =", b)          # grad is 0 for now — we haven't asked for gradients yet
print()

print("=" * 60)
print("STEP 2 — do some math. Watch the RESULT remember its parents.")
print("=" * 60)
c = a + b                # c is 7, but it also remembers it came from a and b via '+'
print("c = a + b  ->", c)
print("   c was made by operation:", repr(c._op))          # '+'
print("   c's parents (_prev):", [p.data for p in c._prev]) # [3.0, 4.0]
print()

d = c * 2                # d is 14, remembers it came from c (and the number 2) via '*'
print("d = c * 2  ->", d)
print("   d was made by operation:", repr(d._op))           # '*'
print("   d's parents (_prev):", [p.data for p in d._prev]) # [7.0, 2.0]
print()

print("=" * 60)
print("STEP 3 — the KEY idea")
print("=" * 60)
print("d does not just hold the number 14.")
print("It holds a little MAP back to everything that built it:")
print("   d  <- (c, 2)        via '*'")
print("   c  <- (a, b)        via '+'")
print()
print("This chain a,b -> c -> d is the COMPUTATION GRAPH.")
print("In Experiment 2 we'll use this exact map to push gradients backward.")
print()
print("Nothing here is magic yet — grads are still 0.0. That's expected.")
print("We are just proving the network 'remembers its history'.")
