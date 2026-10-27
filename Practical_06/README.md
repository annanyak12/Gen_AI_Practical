# Implementation of Self-Attention, Scaled dot product and Multi-Head Self-Attention in Transformer Networks

## Project Overview

This project demonstrates how the **Self-Attention mechanism and Multi-Head Attention** work using PyTorch. It uses the example sentence `Man eat cheese` to illustrate how tokens interact with one another through attention scores and weighted representations.

The notebook implements Query, Key, and Value projections, calculates attention scores, applies Softmax, generates attention output vectors, and visualizes attention distributions using heatmaps.

It also extends the basic attention mechanism to Multi-Head Attention, demonstrating how multiple attention heads process the same input representations independently.

## Objectives

- Understand the Query, Key, and Value (Q, K, V) concept.
- Implement dot-product attention using PyTorch.
- Understand scaled dot-product attention and Softmax normalization.
- Calculate attention output vectors.
- Implement Multi-Head Self-Attention.
- Visualize attention distributions using heatmaps.
- Understand the role of attention in Transformer architectures.

## Technologies Used

- **Python** – Programming language
- **PyTorch** – Tensor operations and attention calculations
- **NumPy** – Numerical computations
- **Matplotlib** – Plotting and visualization
- **Seaborn** – Attention heatmaps
- **Google Colab** – Cloud-based development environment

## Part 1: Scaled Dot-Product Attention

### 1. Input Tokens and Embeddings

The notebook uses three input tokens:

```python
tokens = ["Man", "eat", "cheese"]
```

Each token is represented by a randomly initialized embedding vector of dimension 16.

| Parameter | Value |
|---|---:|
| Sequence length | 3 |
| Embedding dimension (`d_model`) | 16 |
| Query dimension (`d_k`) | 8 |
| Value dimension (`d_v`) | 8 |

These embeddings are randomly generated for demonstration purposes rather than obtained from a pretrained model.

### 2. Query, Key, and Value Projections

Three weight matrices project the input embeddings into Query, Key, and Value representations.

```python
Q = torch.matmul(embeddings, W_q)
K = torch.matmul(embeddings, W_k)
V = torch.matmul(embeddings, W_v)
```

- **Query (Q):** Represents what a token is looking for.
- **Key (K):** Represents the information a token makes available for matching.
- **Value (V):** Contains the information combined according to attention weights.

Each projection produces a tensor of shape `(3, 8)`.

### 3. Calculate Attention Scores

The dot product between queries and keys measures their compatibility.

```python
attn_scores = torch.matmul(Q, K.transpose(0, 1))
```

The resulting attention score matrix has shape `(3, 3)`. Each row corresponds to a query token, and each column corresponds to a key token.

### 4. Apply Softmax

The Softmax function converts the scores into normalized attention weights.

```python
attn_distribution = F.softmax(attn_scores, dim=-1)
```

Each row sums to 1, representing the distribution of attention across the three tokens for a particular query.

**Important:** This first attention calculation uses raw dot products without scaling. The standard scaled dot-product attention formula divides the scores by the square root of the key dimension before applying Softmax.

### 5. Calculate Attention Output

The attention weights are multiplied by the Value matrix to produce contextualized output vectors.

```python
attention_output = torch.matmul(attn_distribution, V)
```

The output shape is `(3, 8)`, meaning each token receives an attention-based representation containing information from the input sequence.

## Part 2: Step-by-Step Attention Mathematics

The notebook manually calculates the dot product between each query and every key.

For each query-key pair, it displays:

1. Raw dot product.
2. Scaled score, calculated by dividing the dot product by \(\sqrt{d_k}\).
3. Softmax probability distribution.

The standard scaled dot-product attention equation is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Here, \(d_k\) is the dimension of the key vectors. Scaling helps control the magnitude of the dot products before Softmax.

The manual calculation illustrates the scaled scores and Softmax probabilities. However, the notebook does not print the manually calculated probability values, and the first attention output is calculated from unscaled scores.

## Part 3: Attention Visualization

The notebook uses Seaborn to generate a heatmap of the attention distribution.

- **Rows:** Query tokens.
- **Columns:** Key tokens.
- **Cell values:** Attention weights.
- **Color intensity:** Relative attention assigned to each token.

The heatmap helps visualize how each token distributes attention across the input sequence.

Because the embeddings and projection weights are randomly initialized, the attention weights do not represent learned linguistic relationships.

## Part 4: Multi-Head Self-Attention

Multi-Head Attention extends the basic attention mechanism by calculating multiple attention distributions in parallel.

### 1. Define Attention Heads

```python
num_heads = 2
d_k_head = d_model // num_heads
d_v_head = d_model // num_heads
```

The embedding dimension is 16, and the notebook uses two attention heads. Each head therefore operates on an 8-dimensional representation.

### 2. Linear Projections

The input embeddings are projected into Query, Key, and Value matrices, each with shape `(3, 16)`.

### 3. Split into Multiple Heads

The projected tensors are reshaped and transposed to separate the two attention heads.

The resulting shapes are:

| Tensor | Shape |
|---|---|
| `Q_split` | `(2, 3, 8)` |
| `K_split` | `(2, 3, 8)` |
| `V_split` | `(2, 3, 8)` |

Each head independently calculates its own attention scores and distribution.

### 4. Scaled Attention for Each Head

```python
scores = torch.matmul(
    Q_split,
    K_split.transpose(-2, -1)
) / (d_k_head ** 0.5)

distributions = F.softmax(scores, dim=-1)
head_outputs = torch.matmul(distributions, V_split)
```

The attention scores are scaled by the square root of the per-head dimension before Softmax.

The resulting attention distributions have shape `(2, 3, 3)`, representing two heads, three queries, and three keys.

### 5. Concatenate Head Outputs

The outputs of the individual heads are combined to restore the original embedding dimension.

```python
concat_output = (
    head_outputs.transpose(0, 1)
    .contiguous()
    .view(seq_len, d_model)
)
```

The concatenated output has shape `(3, 16)`.

### 6. Final Output Projection

A final weight matrix transforms the concatenated representation.

```python
final_output = torch.matmul(concat_output, W_o)
```

The final Multi-Head Attention output has shape `(3, 16)`.

## Multi-Head Attention Visualization

The notebook generates separate heatmaps for both attention heads.

Each heatmap displays the attention distribution learned by that head's randomly initialized projections.

The visualizations help demonstrate that different heads can produce different attention patterns. In this toy example, however, these patterns are random rather than learned from training data.

## How to Run

1. Open the notebook in Google Colab.
2. Run the cells sequentially.
3. Review the printed embeddings, tensor shapes, attention scores, and attention distributions.
4. Examine the step-by-step dot-product calculations.
5. View the basic self-attention heatmap.
6. Inspect the Multi-Head Attention outputs and individual head heatmaps.

### Required Libraries

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

These libraries are commonly available in Google Colab. If required, install missing packages using `pip`.

## Key Concepts Learned

- Self-Attention
- Query, Key, and Value projections
- Dot-product similarity
- Scaled dot-product attention
- Softmax normalization
- Attention output vectors
- Multi-Head Attention
- Tensor reshaping and dimension management
- Attention weight visualization
- Transformer architecture fundamentals

## Limitations

- Uses only three tokens and randomly generated embeddings.
- Does not train the projection matrices or learn from a dataset.
- Does not use a tokenizer or pretrained language model.
- The initial attention calculation does not apply scaling.
- Does not implement masking, residual connections, layer normalization, or a complete Transformer block.
- The input is not processed using a causal mask, so every token can attend to all tokens, including itself.

## Future Improvements

- Implement scaled dot-product attention consistently in both calculations.
- Train attention projections on a real dataset.
- Use pretrained token embeddings for meaningful language representations.
- Compare attention maps before and after training.
- Add causal masking for autoregressive language modeling.
- Implement a complete Transformer encoder block.
- Experiment with different numbers of attention heads and embedding dimensions.

## Author

**Annanya Kesharwani**
