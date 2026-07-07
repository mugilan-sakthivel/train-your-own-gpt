"""
EXPERIMENT 3 — bits, bytes, and "will this model fit on my Mac?"

A weight is ONE number. How many BYTES that number takes depends on its format.
memory to hold a model = (number of weights) x (bytes per weight).

Run it:
    cd /Users/mugilansakthivel/Developer/learning/train-your-own-gpt
    .venv/bin/python explore_03_bytes.py
"""
import torch

print("=" * 60)
print("1. The SAME weight value, stored in different formats")
print("=" * 60)
val = 5.2
for dtype, name in [(torch.float32, "float32 (FP32)"),
                    (torch.float16, "float16 (FP16)"),
                    (torch.bfloat16, "bfloat16"),
                    (torch.int8,    "int8 (quantized)")]:
    t = torch.tensor(val if dtype != torch.int8 else int(val), dtype=dtype)
    print(f"  {name:18s} -> {t.element_size()} byte(s) per weight  "
          f"(= {t.element_size()*8} bits)")

print("\n" + "=" * 60)
print("2. Model memory = weights x bytes-per-weight")
print("=" * 60)
models = [("7B  model", 7e9), ("70B model", 70e9), ("175B (GPT-3)", 175e9)]
print(f"  {'model':14s} | {'FP32 (4B)':>11s} | {'16-bit (2B)':>12s} | {'int8 (1B)':>10s}")
print("  " + "-" * 56)
for name, n in models:
    gb = lambda bytes_each: n * bytes_each / 1e9
    print(f"  {name:14s} | {gb(4):8.0f} GB | {gb(2):9.0f} GB | {gb(1):7.0f} GB")

print("\n" + "=" * 60)
print("3. What fits in YOUR Mac (16 GB)?")
print("=" * 60)
ram_gb = 16
for bytes_each, label in [(4, "FP32"), (2, "16-bit"), (1, "int8")]:
    max_weights = ram_gb * 1e9 / bytes_each
    print(f"  at {label:6s} ({bytes_each} byte/weight): up to ~{max_weights/1e9:.0f} billion weights")
print("\n  (minus overhead for activations, OS, etc. — so be conservative)")
print("  => ~7B models fit in 16-bit; bigger needs int8 quantization or won't fit.")
