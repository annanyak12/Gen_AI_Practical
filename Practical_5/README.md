# Attention Mechanism using Dot-Product Attention

## 📌 Overview

This project demonstrates a simple **Attention Mechanism** using Python and NumPy. It shows how a decoder can focus on different encoder outputs while generating the next word.

The implementation calculates:
1. Encoder outputs
2. Decoder hidden state
3. Alignment scores using dot product
4. Attention weights using Softmax
5. Context vector using a weighted sum

The code can be executed directly in **Google Colab**.

---

## 🧠 Concept

In an encoder-decoder architecture, the encoder produces representations of the input sequence. The decoder uses an **attention mechanism** to determine which encoder outputs are more relevant to the current decoding step.

The basic flow is:

```text
Encoder Outputs
       ↓
Decoder Hidden State
       ↓
Alignment Scores
       ↓
Softmax
       ↓
Attention Weights
       ↓
Context Vector
```

---

## 📊 Input

The encoder produces two output vectors:

```python
encoder_outputs = np.array([
    [0.2, 0.8],
    [0.9, 0.1]
])
```

The decoder's current hidden state is:

```python
decoder_hidden_state = np.array([0.7, 0.4])
```

Each encoder output is compared with the decoder hidden state.

---

## ⚙️ Methodology

### 1. Encoder Output

The encoder outputs represent information from the input sequence.

```text
[0.2, 0.8]
[0.9, 0.1]
```

### 2. Decoder Hidden State

The decoder hidden state represents what the decoder is currently trying to generate.

```text
[0.7, 0.4]
```

### 3. Alignment Scores

A **dot product** is used to measure the similarity between each encoder output and the decoder hidden state.

Formula:

```text
Score = Encoder Output · Decoder Hidden State
```

For the given values:

```text
Score 1 = (0.2 × 0.7) + (0.8 × 0.4)
        = 0.46

Score 2 = (0.9 × 0.7) + (0.1 × 0.4)
        = 0.67
```

Therefore:

```text
Alignment Scores = [0.46, 0.67]
```

### 4. Attention Weights

Softmax converts the alignment scores into probabilities.

The weights indicate how much attention the decoder gives to each encoder output.

```text
Attention Weights ≈ [0.448, 0.552]
```

The weights sum to 1.

### 5. Context Vector

The context vector is calculated as the weighted sum of the encoder outputs.

Formula:

```text
Context = Σ (Attention Weight × Encoder Output)
```

For this example:

```text
Context Vector ≈ [0.586, 0.414]
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Google Colab

---

## 📂 Code

```python
import numpy as np

# Encoder outputs
encoder_outputs = np.array([
    [0.2, 0.8],
    [0.9, 0.1]
])

# Print Encoder Output
print(f"\nEncoder Output (4 words, 2 features each):\n{encoder_outputs}")

# Dummy Decoder Hidden State
decoder_hidden_state = np.array([0.7, 0.4])

print(f"\nDecoder Hidden State (what we are looking for): {decoder_hidden_state}")

# Calculate Alignment Scores
alignment_scores = np.dot(encoder_outputs, decoder_hidden_state)

print(f"\nAlignment Scores (similarity of each encoder output to decoder state): {alignment_scores}")

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Calculate Attention Weights
attention_weights = softmax(alignment_scores)

print(f"\nAttention Weights: {attention_weights}")

# Compute Context Vector
context_vector = np.sum(
    attention_weights[:, np.newaxis] * encoder_outputs,
    axis=0
)

print(f"\nContext Vector: {context_vector}")
```

---

## 📈 Expected Output

```text
Encoder Output:
[[0.2 0.8]
 [0.9 0.1]]

Decoder Hidden State:
[0.7 0.4]

Alignment Scores:
[0.46 0.67]

Attention Weights:
[0.44769209 0.55230791]

Context Vector:
[0.58638387 0.41361613]
```

---

## 🎯 Key Learning

This implementation demonstrates that:

- **Alignment scores** measure similarity between encoder outputs and the decoder state.
- **Softmax** converts scores into attention probabilities.
- Higher attention weight means the decoder gives more importance to that encoder output.
- The **context vector** summarizes the important information from the encoder.
- Attention helps the decoder selectively focus on relevant parts of the input.

---

## 🚀 How to Run

1. Open **Google Colab**.
2. Create a new notebook.
3. Copy the Python code into a code cell.
4. Run the cell.
5. Observe the alignment scores, attention weights, and context vector.

---
