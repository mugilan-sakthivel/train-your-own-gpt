"""
EXPERIMENT 5 — the FORWARD pass through the 13-parameter network, with real numbers.

Network: 2 inputs -> 3 hidden neurons -> 1 output.
  - 9 weights (on the wires)
  - 4 biases  (one INSIDE each of the 4 neurons: h1,h2,h3,output)
  = 13 parameters total.

Each neuron:  output = activation( in1*w1 + in2*w2 + ... + bias )
Activation = ReLU = max(0, x).

Run it:
    cd /Users/mugilansakthivel/Developer/learning/train-your-own-gpt
    .venv/bin/python explore_05_network_forward.py

Try changing the inputs or any weight/bias and re-run to see the data change.
"""

def relu(x):
    return max(0.0, x)

# ---- the inputs: two numbers describing ONE example ----
in1, in2 = 0.5, 1.0

# ---- the 9 weights (on the wires) ----
# input -> hidden
w_i1h1, w_i2h1 = 0.4, -0.2     # into h1
w_i1h2, w_i2h2 = 0.1,  0.7     # into h2
w_i1h3, w_i2h3 = -0.5, 0.3     # into h3
# hidden -> output
w_h1o, w_h2o, w_h3o = 0.6, -0.4, 0.5

# ---- the 4 biases (inside the neurons) ----
b_h1, b_h2, b_h3 = 0.1, -0.3, -0.2
b_o = 0.05

target = 1.0   # the correct answer we wanted

print(f"INPUTS:  in1 = {in1},  in2 = {in2}\n")

# ---- hidden layer: each neuron = weighted sum + bias, then ReLU ----
sum_h1 = in1*w_i1h1 + in2*w_i2h1 + b_h1
h1 = relu(sum_h1)
print(f"h1: {in1}*{w_i1h1} + {in2}*{w_i2h1} + bias {b_h1} = {sum_h1:.2f}  -> ReLU = {h1:.2f}")

sum_h2 = in1*w_i1h2 + in2*w_i2h2 + b_h2
h2 = relu(sum_h2)
print(f"h2: {in1}*{w_i1h2} + {in2}*{w_i2h2} + bias {b_h2} = {sum_h2:.2f}  -> ReLU = {h2:.2f}")

sum_h3 = in1*w_i1h3 + in2*w_i2h3 + b_h3
h3 = relu(sum_h3)
print(f"h3: {in1}*{w_i1h3} + {in2}*{w_i2h3} + bias {b_h3} = {sum_h3:.2f}  -> ReLU = {h3:.2f}  (clipped to 0 if negative)")

# ---- output layer: combine the 3 hidden values ----
sum_o = h1*w_h1o + h2*w_h2o + h3*w_h3o + b_o
prediction = sum_o   # linear output (no activation on the final neuron here)
print(f"\nout: {h1:.2f}*{w_h1o} + {h2:.2f}*{w_h2o} + {h3:.2f}*{w_h3o} + bias {b_o} = {prediction:.3f}")

# ---- the error ----
error = (prediction - target) ** 2
print(f"\nprediction = {prediction:.3f},  target = {target}")
print(f"error = (prediction - target)^2 = ({prediction:.3f} - {target})^2 = {error:.4f}")

print("\nThe 'data' at each step:")
print(f"  h1={h1:.2f}  h2={h2:.2f}  h3={h3:.2f}  ->  out={prediction:.3f}  ->  error={error:.4f}")
print("\nThat was FORWARD (filling data). BACKWARD would now fill each of the 13 grads.")
