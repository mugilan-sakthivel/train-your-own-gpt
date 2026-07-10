# Neural Networks — Master Learning Notes

> My complete source of truth. Everything I learned, step by step, in the order it made
> sense — including every doubt that confused me and exactly how it got cleared. The chat
> will disappear; this file won't. Read it top to bottom and it all comes back.

**How to read this:** Each part builds on the one before. Boxes marked **⚠️ This confused
me** are the exact spots that tripped me up and the fix. Boxes marked **🔑 Key idea** are
the things to never forget. All numbers are real and worked out.

---

## PART 0 — The one mental model to hold onto

A neural network is **a big pile of numbers (called weights) plus a simple formula that
turns an input into an output.** "Learning" = slowly adjusting those numbers until the
answers are good.

That's the whole thing. Everything below is just: *what are those numbers, how does the
formula work, and how do they get adjusted?*

---

## PART 1 — The big map: AI → ML → Deep Learning → Neural Networks

These four words are nested, biggest to smallest:

| Word | Meaning |
|---|---|
| **AI** | Making a machine do a task intelligently / making a machine smarter. Biggest umbrella. |
| **Machine Learning (ML)** | Making a machine smarter by learning patterns **from data**, not hand-written rules. |
| **Deep Learning (DL)** | ML done with **neural networks that have many layers** ("deep" = many layers). |
| **Neural Network (NN)** | The machine itself: weights + a formula. |

So **AI ⊃ ML ⊃ Deep Learning ⊃ (uses) Neural Networks.**

**Normal program vs neural network** — the key flip:
- **Normal program:** *you write the rules.* `if temperature > 100: alert()`. You decide everything.
- **Neural network:** *you show examples* ("this is right, this is not"), and *the network
  figures out the rules itself.* You never write the rule.

> **🔑 Key idea:** A normal program = you give rules → it gives answers. A neural network =
> you give data + answers → it figures out the rules.

**Deep learning, precisely:** a stack of neurons = a **layer**; a stack of layers = "deep";
running and training it many times = deep learning. That's the box we work inside.

---

## PART 2 — The Box: the foundation of everything

Every single number in a network lives in a **box** (in code it's called a `Value`). A box
holds **two** numbers in two slots:

```
┌─────────────┐
│ data = 0.4  │  ← the value itself
│ grad = -12  │  ← how much this value affects the loss
└─────────────┘
```

Think of it as a **sticky note** stuck on every number, with two lines written on it.

- The **forward pass** writes the top number (`data`).
- The **backward pass** writes the bottom number (`grad`).

> **⚠️ This confused me:** *Is a box the same as a neuron?* **No.** A box is **one number**.
> A neuron is **made of many boxes** (its weights, its bias, its output). A box ≠ a neuron.

> **⚠️ This confused me:** *Does `data` always mean "weight"?* **No.** `data` just means "the
> number in this box." A weight-box's data is a weight; the loss-box's data is the loss; an
> input-box's data is the input. **`data` = whatever number this particular box holds.**

### data — the value
`data` = the current number inside a box. Simplest thing. If a weight is set to 0.4, that
weight-box's `data` is 0.4.

### grad — the effect on the loss
`grad` = how much this box affects the final loss. It answers one question:

> "If I nudge this number up a tiny bit, does the loss go up or down, and by how much?"

`grad` has **two parts**:
- the **sign** (+ or −) → *which way* to turn the knob.
- the **number** (size) → *how strongly* this affects the loss = *how big a step* it takes.

> **🔑 Key idea about the sign:** **minus → turn the weight UP** (to reduce loss);
> **plus → turn the weight DOWN.** (Because we always want to reduce the loss.)

**Default values:**
- A weight/bias box starts with `data` = a **random number** (a first guess).
- Every box starts with `grad` = **0**, and stays 0 until `backward()` runs.

---

## PART 3 — The pieces of a network

### Weight — the line, and a box
A **weight** is a learned number on one input, saying "how much does this input matter?"

> **⚠️ This confused me:** *In the drawing, where is a weight?* **The weight IS the line**
> connecting two circles. That line is also a box (its `data` = the weight value, its `grad`
> = its effect on the loss). So "weight = a box" and "weight = a line" are the same thing.

- Each input into a neuron has its **own** weight. 2 inputs → 2 weights into that neuron.
- A **negative** weight makes that input push the answer *down*.
- Weights start as **random** numbers; training improves them.

### Bias — one per neuron, inside it, and it LEARNS
A **bias** is a single number that belongs to a neuron (not to any input, not on a wire).
It's added **after** the weighted sum, inside the neuron. It shifts the result up or down.

> **⚠️ This confused me:** *How many biases, and does the bias change?*
> - **One bias per neuron.** A layer of 3 neurons → 3 biases. (Count the circles except
>   inputs → that's the bias count.)
> - **Yes, the bias DOES change during training.** It's a parameter exactly like a weight:
>   it has a grad, and gradient descent updates it → `new bias = old bias − lr × grad`.

### Input — fixed, never changed
An **input** is one of the numbers describing the example you feed in (e.g. sunlight = 0.5,
water = 1.0). It's the data / the question.

> **⚠️ This confused me:** *Do inputs change? Do they affect the loss?* Inputs **do** affect
> the loss, but **we never change them** — they're given. **The only boxes we ever update are
> weights and biases.** Inputs and the true answer stay fixed.

### The neuron — the building block
A **neuron** does exactly **three steps**:
1. **Weighted sum** — multiply each input by its weight, add them all.
2. **Add the bias.**
3. **Activation** — pass the result through an activation function.

```
neuron output = activation( in₁×w₁ + in₂×w₂ + … + bias )
                            └──── weighted sum ────┘   └┬─┘
                                                      bias
```

**Which boxes a single neuron involves:** its input boxes (coming from before), its weight
boxes (the lines coming in), its one bias box, and the output box it produces.

Here's a single neuron drawn as boxes:

```
 INPUT boxes        WEIGHT boxes (the lines)
 ┌──────────┐       ┌──────────┐
 │ in₁ = 0.5│──×────│ w₁ = 0.4 │──┐
 └──────────┘       └──────────┘  │
                                (add) ──▶ + bias(0.1) ──▶ ReLU ──▶ ┌─────────┐
 ┌──────────┐       ┌──────────┐  │                                │ out=0.10│
 │ in₂ = 1.0│──×────│ w₂ =-0.2 │──┘                                └─────────┘
 └──────────┘       └──────────┘
```

### Activation function — bends the straight line into a curve
The **activation** is applied at the end of the neuron. Without it, stacking many neurons
would collapse into **one big straight line** — it could only learn straight-line
relationships. The activation adds a **bend** so the network can learn **complex** patterns.

| Activation | What it does | Used for |
|---|---|---|
| **ReLU** | if negative → 0, else keep it | the default, inside most networks |
| **Sigmoid** | squish any number into 0…1 | output as a probability |
| **Softmax** | turn a list into probabilities summing to 1 | picking among many options |

> **🔑 Key idea:** ReLU = `max(0, x)`. It clips negatives to 0. That's why in the plant
> example below, h₃'s −0.15 becomes 0.

### Layer — a column of neurons, and how they connect
A **layer** = one vertical **column** of neurons. Data flows left → right, one layer at a
time. Input layer (first column) → hidden layers (middle) → output layer (last column).

> **⚠️ This confused me:** *Rows or columns — which is a layer? And how do neurons connect?*
> - **A layer = a column.** Left→right = going through layers (this is the "depth"). Top→bottom
>   within a column = the neurons of that one layer (the "width").
> - **Neurons in the SAME layer do NOT connect to each other.** Each independently takes the
>   same inputs and computes its own output.
> - **Every neuron's output feeds EVERY neuron in the next layer** ("fully connected"). The
>   outputs of layer 1 become the inputs of layer 2, and so on.

```
LAYER 1            LAYER 2
 n₁ ──┬──────────▶ m₁
      ├──────────▶ m₂     (each m gets BOTH n₁ and n₂;
 n₂ ──┴──────────▶ m₃      n₁ and n₂ never touch each other)
```

**Deep learning** = a network with **many** hidden layers. More columns = deeper; more
neurons per column = wider.

### The output neuron — just another neuron
> **⚠️ This confused me:** *How is the final output calculated — is it summing all the
> neurons?* **No.** The output is **just another neuron**, using the same formula: it takes
> the previous layer's outputs as its inputs, multiplies each by a weight, adds them, adds
> its bias, (optionally an activation). It's not special — it's simply the last neuron.

---

## PART 4 — The forward pass (real numbers)

**Forward pass** (a.k.a. **inference**) = running the inputs left → right through every
neuron of every layer to get the final output. (When you chat with an AI, this is what
happens.)

**Our example — the plant network:** 2 inputs → 3 hidden → 1 output. 13 parameters (9
weights + 4 biases). Inputs: sunlight = 0.5, water = 1.0. Activation = ReLU. True answer = 1.

```
Hidden layer:
h₁: 0.5×0.4  + 1.0×(−0.2) + 0.1  =  0.10  → ReLU → 0.10
h₂: 0.5×0.1  + 1.0×0.7    + (−0.3) = 0.45 → ReLU → 0.45
h₃: 0.5×(−0.5)+ 1.0×0.3   + (−0.2) = −0.15 → ReLU → 0.00   (negative, clipped to 0)

Output neuron (inputs = h₁,h₂,h₃; weights 0.6,−0.4,0.5; bias 0.05):
out: 0.10×0.6 + 0.45×(−0.4) + 0.00×0.5 + 0.05  =  −0.07   ← the prediction
```

With random weights, it predicts **−0.07** but the true answer was **1** — badly wrong,
which is expected before training. *(Run it: `train-your-own-gpt/explore_05_network_forward.py`.)*

---

## PART 5 — The loss (how wrong we are)

The **loss** (same word as **error**) is one number saying how wrong the prediction is.
Bigger = worse, `0` = perfect. A common formula is **squared error**: `(prediction − answer)²`.

```
prediction = −0.07,  true answer = 1
loss = (−0.07 − 1)² = (−1.07)² = 1.1449
```

> **⚠️ This confused me:**
> - "loss" and "error" are the **same thing**, two names.
> - There is only **ONE loss box for the entire network**, at the very end — **not** one per
>   neuron.

**The entire goal of training is to make this one loss number as small as possible.**

---

## PART 6 — The TWO different formulas (this confused me a lot)

There are **two separate formulas doing two separate jobs.** Mixing them up was my biggest
early confusion.

| | **Loss formula** | **Grad formula** |
|---|---|---|
| tells you | **how wrong** you are | **how fast the wrongness changes** if you nudge the weight |
| example | `loss = (w − 10)²` | `grad = 2(w − 10)` |
| think of it as | the **height** of the ground | the **steepness** of the slope where you stand |

**The grad is the *slope* of the loss.** Loss = how high up (how wrong). Grad = how steep
(which way and how fast wrongness changes as you move the weight).

Watch both change, with numbers:
```
w = 4  →  loss = (4−10)²  = 36
w = 5  →  loss = (5−10)²  = 25     (loss dropped by 11)
w = 6  →  loss = (6−10)²  = 16     (loss dropped by 9)
```
Two things happen at once: the loss changes (36→25→16), **and** the amount it drops changes
(11, then 9…) — the slope is getting gentler near the bottom. That "slope at your current
spot" is the **grad**. At w=4 it's steep (−12); near w=10 it's almost flat (≈0).

> **🔑 Key idea — grad = rate = derivative (ALL the same thing):** The number I kept calling
> "the rate" **IS the grad.** `2(w−10)` is the grad, "rate" is just another name, and
> "derivative" (the math word) is a third name. **One number, three names.** It's the rate
> of change of the loss as you change the weight.

---

## PART 7 — data & grad, deep (the two boxes)

Take the simplest case to see the two boxes clearly: a **weight box** and an **error box**.

```
   WEIGHT BOX                 ERROR BOX
┌────────────────┐         ┌────────────────┐
│ data = 4       │         │ data = 36      │
│ grad = -12     │         │ grad = 1       │   ← always 1
└────────────────┘         └────────────────┘
```

> **⚠️ This confused me:** *What is the grad in the error box?* **It's always 1.** Because
> "grad" means "effect on the final loss," and the loss's effect on itself is one-to-one → 1.
> **This `1` is the SEED where backprop starts.**

**Which boxes have a value?** *All of them.* Every box (weight, input, bias, output, loss)
has a `data`. What that number *represents* depends on the box.

**The order things get filled, in one round:**
1. Weight box exists → `data = 4`, `grad = 0`.
2. Calculate the error → error box appears → `data = 36`, `grad = 0`.
3. `backward()` fills grads: error box `grad = 1` first (the seed), then weight box `grad = −12`.
4. Update the weight.

---

## PART 8 — Backpropagation (how grad is computed automatically)

**Backpropagation** = the process that computes the `grad` of **every** box automatically.
It starts at the loss, walks **backward** to the first layer, and fills each box's grad.

**Why we need it:** for one weight I could measure the grad by hand (`2(w−10)`). But a real
network has millions of weights — no human can write a formula for each. Backprop does them
all automatically.

> **🔑 My own analogy that worked:** the computation graph is like a **linked list** — each
> box stores a link back to the boxes that made it, so backprop can walk backward through
> the chain. (That intuition was correct.)

**The trail:** a weight doesn't touch the loss directly — it goes through steps. The tiny
loss `(w−10)²` is really **two steps**:
```
step 1 (subtract): d = w − 10
step 2 (square):   e = d²        ← the loss
```
So the trail is `w → d → e`.

**Each step knows only its own simple "local rate":**
- subtract step (`d = w − 10`): nudge w up by 1 → d goes up by 1. Local rate = **1**.
- square step (`e = d²`, at d=−6): local rate = **2×d = −12**. *(You can verify by nudging:
  at d=−6 loss=36; at d=−5.99 loss=35.8801; change/nudge = −11.99 ≈ −12.)*

**Why you MULTIPLY the rates (the chain rule):**
> Currency analogy: $1 = 80 rupees, and 1 rupee = 100 paise, so $1 = 80 × 100 = **8000 paise**.
> The conversion **rates multiply** along the chain. Same here — to get how `w` affects the
> loss, multiply the local rates along the trail.

**Walk the trail backward (this IS backprop):**
```
e.grad = 1                          ← the seed (loss vs itself)
d.grad = (rate d→e) × e.grad = (−12) × 1  = −12
w.grad = (rate w→d) × d.grad = (1) × (−12) = −12
```
Backprop rebuilt `w.grad = −12` — the **same** number as the hand formula `2(w−10)`, and the
**same** as PyTorch's real autograd. All three agree. *(Run: `explore_04_backprop.py`.)*

> **⚠️ This confused me (the Karpathy video):** "derivative of g with respect to a" is just
> fancy math for **"a's grad"** = "how a affects g." **derivative = grad = rate.** The word
> "derivative" was hiding a thing I already understood. Also, his `a.grad = 138` means
> "nudge a up a little and g grows at rate 138" — exactly the same idea.

**Forward vs backward as two waves:** forward sends a **green wave** left→right filling every
box's `data`; backward sends an **orange wave** right→left filling every box's `grad`.
*(See the animation `train-your-own-gpt/backprop-animated.html`.)*

---

## PART 9 — Gradient descent (using the grads to update)

**Gradient descent** = the step that actually **updates** the weights and biases using their
grads. The formula:

```
new = old − (learning rate × grad)
```

The **minus sign** steps each knob *against* its grad → downhill → toward less loss.

> **🔑 Key idea — backprop vs gradient descent (these are DIFFERENT steps):**
> - **grad** = a number (the answer).
> - **backpropagation** = how we GET all the grads.
> - **gradient descent** = how we USE the grads to move the weights & biases.
> backprop computes; gradient descent acts.

> **⚠️ This confused me — what does the grad's NUMBER do?** The **size** of the grad decides
> **how big the weight's update is.** With lr = 0.1:
> ```
> BIG grad (−12): step = 0.1 × 12 = 1.2  → weight moves a LOT
> SMALL grad (−2): step = 0.1 × 2  = 0.2  → weight moves a little
> ```
> So: **sign = which way, number = how far.** Both feed the one formula `new = old − lr × grad`.

Worked example — w=4, grad −12, lr 0.1:
```
new w = 4 − (0.1 × −12) = 4 + 1.2 = 5.2
new loss = (5.2 − 10)² = 23.04    (was 36 — it dropped!)
```

**Picture:** a ball rolling downhill into a valley. The grad = the slope where the ball sits;
the learning rate = the step size. This is *why* training slows near the answer — as the loss
flattens, the grad shrinks toward 0, so steps get tiny and the weight gently settles.

---

## PART 10 — The learning rate (where it comes from)

The **learning rate (lr)** = how big a step gradient descent takes. A small number like 0.1
or 0.01.

> **⚠️ This confused me — is it random?** **No.** You **choose** it (it's not random, and the
> network never learns it). It's a **hyperparameter** — a setting you provide, usually fixed
> during training.

**How you find a good value: trial and tuning.** Pick a common value, run training, watch the loss:
```
loss drops smoothly     → good lr ✅ keep it
loss explodes / bounces → TOO BIG,  make it smaller (÷10)
loss crawls very slowly → TOO SMALL, make it bigger  (×10)
```
*(I saw this live: lr = 0.1 worked; lr = 1.05 made the loss explode.)*

---

## PART 11 — The training loop (the whole thing)

Training = repeating **four steps** over and over:

1. **Forward** — run inputs through → get prediction.
2. **Loss** — measure how wrong (the loss box).
3. **Backprop** — fill every box's grad (loss → back to first layer).
4. **Gradient descent** — update every weight **and bias**: `new = old − lr × grad`.

…then repeat, **millions of times.** Each loop the loss gets a little smaller.

**Epoch** = one full pass over all your data. **Batch (minibatch)** = a small handful of
examples used for one update step. Real training: for each epoch → for each batch → forward,
loss, backward, update.

---

## PART 12 — Parameters vs Hyperparameters

| | who sets it | examples |
|---|---|---|
| **Parameter** | the **machine** learns them | weights, biases |
| **Hyperparameter** | **you** choose them, before training | learning rate, #layers, neurons per layer, batch size |

---

## PART 13 — Bits, bytes & "billions of parameters"

**Parameters** = all weights + all biases together. "175 billion parameters" = 175 billion
of these numbers.

- **bit** = one 0/1. **byte** = 8 bits. Each weight is a number stored in some bytes.
- bytes per weight: **float32 = 4 bytes**, **float16 = 2 bytes**, **int8 = 1 byte**.
- **Memory = (number of weights) × (bytes per weight).** The "≈ 2 bytes" rule = 16-bit storage.

| model | weights | × 2 bytes (16-bit) | = memory |
|---|---|---|---|
| 7B | 7,000,000,000 | ×2 | 14 GB |
| 70B | 70,000,000,000 | ×2 | 140 GB |
| 175B (GPT-3) | 175,000,000,000 | ×2 | 350 GB |

- The "doubling" = going 1→2→4 bytes (int8 → 16-bit → FP32) doubles the memory each step.
- **Quantization** = fewer bytes per weight (e.g. int8) → smaller model, fits smaller hardware.
- **My Mac (16 GB)** fits ~7B in 16-bit; bigger needs quantization.
- **One weight = one connection** (one line). A model is the same training loop run over
  millions/billions of these at once; backprop fills every grad in one backward pass.

---

## PART 14 — The two complete worked examples

### Example A — plant forward pass (13 params) — see Part 4.
Prediction −0.07 vs answer 1 → loss 1.1449.

### Example B — one weight learning (loss → 0)
Simplest network: input x=1, one weight w, prediction = w, target = 10 → loss = (w−10)²,
grad = 2(w−10). Start w=4, lr 0.1:

| round | w | loss (w−10)² | grad 2(w−10) | new w = w − 0.1×grad |
|---|---|---|---|---|
| 1 | 4.00 | 36.00 | −12.0 | 5.20 |
| 2 | 5.20 | 23.04 | −9.6 | 6.16 |
| 3 | 6.16 | 14.75 | −7.68 | 6.93 |
| 4 | 6.93 | 9.44 | −6.14 | 7.54 |
| … | → 10 | → 0 | → 0 | → 10 |

Weight climbs 4→10, loss falls 36→0, grad shrinks toward 0. *(Run: `explore_02_two_boxes.py`.)*

---

## PART 15 — Every "aha" nuance (the small things that triggered my doubts)

A quick-scan list of every small clarification that unlocked something:

1. A **box ≠ a neuron.** A box is one number; a neuron is made of many boxes.
2. **`data` ≠ "weight."** data = "the number in this box," whatever kind.
3. **error = loss** (same word).
4. **grad = rate = derivative** (all the same number, three names).
5. A **weight is the LINE** in the drawing (and it's a box).
6. There is **one loss box** for the whole network, at the end.
7. The **error box's grad = 1** always — the seed of backprop.
8. **Bias is one per neuron**, added inside, **and it learns** (updates like a weight).
9. **Inputs never change** — only weights & biases update. (Inputs still affect the loss.)
10. The **output is just another neuron** (same formula, not "summing neurons").
11. **Same-layer neurons don't connect**; every neuron feeds every neuron in the next layer.
12. A **layer = a column**; deep = many columns.
13. The **grad's sign = direction, the grad's number = step size** (used in gradient descent).
14. **backprop computes grads; gradient descent uses them** — two separate steps.
15. **Learning rate = you choose it** (a hyperparameter), tuned by watching the loss.
16. Weights start **random**; grads start **0** until backward runs.
17. Training **slows near the answer** because the grad shrinks toward 0.
18. The **chain rule = multiply local rates** along the trail (currency analogy).

---

## PART 16 — Formula cheat-sheet

| thing | formula |
|---|---|
| Neuron | `output = activation( Σ(input × weight) + bias )` |
| ReLU | `ReLU(x) = max(0, x)` |
| Loss (squared error) | `loss = (prediction − answer)²` |
| Grad (= rate) for that loss | `grad = 2 × (weight − answer)` |
| Backprop seed | `loss.grad = 1` |
| Chain rule | grad of a box = product of local rates along the trail back to it |
| Gradient descent update | `new = old − (learning_rate × grad)` (for every weight AND bias) |

---

## PART 17 — Glossary (every term)

- **AI** — making a machine do tasks intelligently.
- **Machine Learning** — making a machine smarter by learning patterns from data.
- **Deep Learning** — ML with neural networks that have many layers.
- **Neural network** — weights + a formula that turns inputs into an output.
- **Box (Value)** — one number, with two slots: data & grad.
- **data** — the current value in a box.
- **grad / gradient / rate / derivative** — how much a box affects the loss; sign = direction,
  number = step size. All the same word.
- **Weight** — a learned number on an input (the line in the drawing).
- **Bias** — a learned number inside a neuron, added after the weighted sum (it learns too).
- **Input** — the given numbers describing one example (never changed).
- **Weighted sum** — each input × its weight, all added.
- **Neuron** — weighted sum + bias → activation.
- **Activation (ReLU/sigmoid/softmax)** — the bend that adds non-linearity.
- **Layer** — a column of neurons; hidden layers are the middle ones.
- **Fully connected** — every neuron feeds every neuron in the next layer.
- **Forward pass / inference** — running inputs through to get a prediction.
- **Loss / error** — one number for how wrong the prediction is.
- **Backpropagation** — walking backward from the loss, multiplying local rates, to fill every grad.
- **Chain rule** — rates multiply along a chain of steps.
- **Gradient descent** — new = old − learning_rate × grad (update every weight & bias).
- **Learning rate** — how big each step is (you choose it — a hyperparameter).
- **Training loop** — forward → loss → backprop → gradient descent, repeated.
- **Parameter** — a number the machine learns (weights, biases).
- **Hyperparameter** — a setting you choose (learning rate, #layers, batch size).
- **Epoch / batch** — one full pass over all data / a small handful of examples per step.
- **Overfitting** — memorizing training data instead of learning the pattern (not yet gone deep).
- **Generalization** — doing well on new, unseen data (not yet gone deep).
- **Quantization** — storing weights in fewer bytes so the model fits smaller hardware.

---

## The scripts I can run (all in `train-your-own-gpt/`, run with `.venv/bin/python <file>`)

- `explore_01_value.py` — a box remembers how it was made (the computation graph).
- `explore_02_two_boxes.py` — watch the weight box & error box as loss drops 36 → 0.
- `explore_03_bytes.py` — bits/bytes/memory; what fits on my Mac.
- `explore_04_backprop.py` — backprop step by step; matches hand formula & PyTorch.
- `explore_05_network_forward.py` — the 13-parameter plant network forward pass.
- `stage1_micrograd.py` — the full autograd engine + a tiny trained network.

**Companion visual pages** (in `neural-networks-explained/`): `neural-networks-complete.html`
(19-chapter lesson), `backprop-animated.html`, `deep-dive.html`, `index.html`,
`transformers.html`; hub at `train-your-own-gpt/course.html`.

---

*End of notes. This is the complete Neural Networks foundation. Next: overfitting/batches
(short), then Stage 2 — build & train a real model.*
