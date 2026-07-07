"""
EXPERIMENT 2  —  Watch the TWO BOXES change as training drives error 36 -> 0.

This is the exact loop you described, printed step by step so you can SEE it:
  - WEIGHT box : data = the weight (we tune it),  grad = its effect on the error
  - ERROR box  : data = the error (loss),         grad = always 1 (it affects itself)

We use the hand formulas you already know:
  error      = (w - 10)^2
  weight grad = 2 * (w - 10)
(Next lesson, backprop will compute that weight grad automatically instead.)

Run it:
    cd /Users/mugilansakthivel/Developer/learning/train-your-own-gpt
    .venv/bin/python explore_02_two_boxes.py
"""

w = 4.0          # WEIGHT box data — our random starting guess
step = 0.1       # how big a nudge we take each round (the "small value")

print("Goal: push the ERROR down to 0 by tuning the WEIGHT toward 10.\n")
print("round |        WEIGHT box        |        ERROR box         | new weight")
print("      |   data        grad       |   data        grad       |")
print("-" * 78)

for round_num in range(1, 26):
    # STEP 1: calculate the error from the current weight  -> fills ERROR box data
    error = (w - 10) ** 2

    # STEP 2: fill the grads
    weight_grad = 2 * (w - 10)   # weight box grad: how w affects the error
    error_grad = 1               # error box grad: always 1 (error vs itself)

    # show both boxes BEFORE we update
    new_w = w - step * weight_grad
    print(f"  {round_num:2d}  |  data={w:6.3f}  grad={weight_grad:7.3f} "
          f"|  data={error:7.3f}  grad={error_grad}     |  {new_w:6.3f}")

    # STEP 3: update the WEIGHT box data with the new weight
    w = new_w

    # stop early once we're basically perfect
    if error < 0.001:
        break

print("-" * 78)
print(f"\nDone. Final weight ≈ {w:.3f} (target was 10), final error ≈ {(w-10)**2:.5f} (target 0).")
print("\nNotice:")
print("  • WEIGHT box data climbs 4 -> 10")
print("  • WEIGHT box grad shrinks toward 0 (less and less left to fix)")
print("  • ERROR box data falls 36 -> 0   (this is 'learning')")
print("  • ERROR box grad stays 1 every single round (the seed)")
