# Implementation of Positional Encoding and Layer Normalization in Transformer architecture

## Project Overview

This project demonstrates two fundamental concepts used in deep learning: **Sinusoidal Positional Encoding** in Transformer architectures and **Layer Normalization** in a feed-forward neural network.

The first implementation generates positional encodings using TensorFlow and NumPy to represent the positions of tokens in a sequence. The second implementation uses PyTorch to build a simple neural network with a linear layer, Layer Normalization, ReLU activation, and an output layer.

These implementations provide a basic understanding of how Transformers represent sequence positions and how normalization can be incorporated into neural network architectures.

## Objectives

- Understand the purpose of positional encoding in Transformer models.
- Implement sinusoidal positional encoding using sine and cosine functions.
- Generate positional encodings for 50 positions with an embedding dimension of 512.
- Understand Layer Normalization in neural networks.
- Build a feed-forward neural network using PyTorch.
- Observe the shape and output of neural network predictions.

## Technologies Used

- **Python** – Programming language
- **NumPy** – Numerical computations
- **TensorFlow** – Tensor creation and data type conversion
- **PyTorch** – Neural network implementation
- **Google Colab** – Cloud-based development environment

## Part 1: Sinusoidal Positional Encoding

### What is Positional Encoding?

Transformers do not inherently process tokens in a sequential order like recurrent neural networks. Positional encoding provides information about the position of each token in a sequence.

It generates numerical vectors that are added to token embeddings in a Transformer model to help represent token positions.

### Implementation

The notebook defines the following function:

```python
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000,
        (2 * (np.arange(d_model)[np.newaxis, :] // 2))
        / np.float32(d_model)
    )

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
```

### How It Works

1. **Generate position indices:** Creates position values from 0 to 49.
2. **Generate dimension indices:** Creates 512 embedding dimensions.
3. **Calculate angles:** Uses position and dimension indices to calculate the sinusoidal angle values.
4. **Apply sine and cosine:** Even dimensions use sine, while odd dimensions use cosine.
5. **Add a batch dimension:** Expands the array to include a leading dimension.
6. **Convert to TensorFlow tensor:** Returns the result as a `float32` tensor.

### Mathematical Formula

The positional encoding is calculated using:

\[
PE(pos,2i)=\sin\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
\]

\[
PE(pos,2i+1)=\cos\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
\]

Where:

- \(pos\) represents the token position.
- \(i\) represents the dimension index within a sine-cosine pair.
- \(d_{\text{model}}\) is the embedding dimension, set to 512.

### Configuration and Output

| Parameter | Value |
|---|---:|
| Number of positions | 50 |
| Embedding dimension | 512 |
| TensorFlow data type | `float32` |
| Output shape | `(1, 50, 512)` |

The leading dimension of 1 is added by `np.newaxis`. The resulting tensor contains positional information for 50 positions, with 512 values per position.

## Part 2: Neural Network with Layer Normalization

### What is Layer Normalization?

Layer Normalization is a technique that normalizes the features of each individual input example across a specified dimension.

It helps control the scale of intermediate activations and is widely used in modern neural network architectures, including Transformers.

### Network Architecture

The notebook defines a neural network using PyTorch.

| Layer | Configuration | Output shape |
|---|---|---|
| Input | 64 features | `(10, 64)` |
| Fully connected layer | 64 → 128 | `(10, 128)` |
| Layer Normalization | 128 features | `(10, 128)` |
| ReLU activation | Applied element-wise | `(10, 128)` |
| Output layer | 128 → 10 | `(10, 10)` |

### Implementation

```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

### How It Works

1. **Input:** Receives a batch of 10 examples, each containing 64 features.
2. **Fully connected layer:** Transforms the 64 input features into 128 features.
3. **Layer Normalization:** Normalizes the 128 features for each example.
4. **ReLU activation:** Introduces non-linearity by replacing negative activation values with zero.
5. **Output layer:** Converts the 128 features into 10 output values.

The network returns a tensor of shape `(10, 10)`, containing 10 output values for each of the 10 input examples.

## How to Run

1. Open the notebook in Google Colab.
2. Run the cells sequentially.
3. Review the generated positional encoding tensor and its shape.
4. Observe the neural network output tensor.

### Required Libraries

```python
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
```

NumPy, TensorFlow, and PyTorch are generally available or can be installed in Google Colab if needed.

## Key Concepts Learned

- Sinusoidal positional encoding
- Sine and cosine functions
- Transformer positional information
- Tensor dimensions and shapes
- Fully connected neural networks
- Layer Normalization
- ReLU activation
- Forward propagation
- TensorFlow and PyTorch basics

## Limitations

- The positional encoding is generated independently and is not integrated into a complete Transformer.
- The neural network uses randomly generated input data rather than a real dataset.
- The neural network is not trained, so its output is not a meaningful prediction.
- The implementation does not include a loss function, optimizer, or evaluation metrics.

## Future Improvements

- Visualize positional encodings using a heatmap.
- Add positional encodings to token embeddings.
- Integrate positional encoding into a Transformer encoder.
- Train the neural network on a sample classification dataset.
- Compare neural network behavior with and without Layer Normalization.
- Evaluate the trained model using suitable performance metrics.

## Author

**Annanya Kesharwani**
