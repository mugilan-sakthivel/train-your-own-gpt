"""STAGE 0 — confirm PyTorch sees your Apple GPU (MPS) and can do real work."""
import torch, time

print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", dev)

a = torch.randn(2000, 2000, device=dev)
b = torch.randn(2000, 2000, device=dev)
t = time.time()
for _ in range(20):
    c = a @ b
if dev == "mps":
    torch.mps.synchronize()
print(f"20x 2000x2000 matmuls on {dev}: {time.time()-t:.3f}s")
print("ok ✅")
