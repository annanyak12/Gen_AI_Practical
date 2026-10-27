# Implementation of an LSTM-based Encoder-Decoder Architecture for Sequence-to-Sequence learning

## Project Overview

This project implements a **Sequence-to-Sequence (Seq2Seq) model using LSTM (Long Short-Term Memory)** to reverse input text character by character.

The model learns to generate the characters of an input word in reverse order. For example, given the input `hello`, the expected output is `olleh`.

The implementation uses TensorFlow and Keras to build, train, and evaluate an encoder-decoder architecture. It reads training data from a text file and automatically creates a sample corpus if the file is unavailable.

## Objectives

- Understand the Seq2Seq encoder-decoder architecture.
- Learn how character-level tokenization works.
- Implement LSTM networks for sequence processing.
- Train a decoder to generate reversed character sequences.
- Understand teacher forcing, padding, embeddings, and inference.
- Generate predictions using a trained model.

## Technologies Used

- **Python** – Programming language
- **NumPy** – Numerical operations and random seed management
- **TensorFlow** – Deep learning framework
- **Keras** – Neural network layers and model training
- **Google Colab** – Cloud-based development environment

## Model Architecture

The project uses an encoder-decoder architecture consisting of the following components:

### 1. Character-Level Tokenization

Each unique character in the training corpus is assigned a numerical ID. Special IDs are reserved for padding, the start token, and the end token.

For example, the word `cat` is represented as a sequence of character IDs.

### 2. Embedding Layer

The embedding layer converts each character ID into a dense vector of **32 dimensions**.

This representation is passed to the LSTM layers for processing.

### 3. Encoder LSTM

The encoder reads the input character sequence and generates two states:

- **Hidden state (h):** Represents the encoder's current output information.
- **Cell state (c):** Maintains information across the sequence.

Both states are passed to the decoder to initialize its processing.

### 4. Decoder LSTM

The decoder uses the encoder's final states to generate the reversed sequence one character at a time.

It receives a start token and predicts the next character at each step until it generates an end token or reaches the generation limit.

### 5. Dense Output Layer

A Dense layer with Softmax activation calculates the probability of each possible character or special token being the next output.

The character with the highest predicted probability is selected during inference.

## Data Preprocessing

The program performs the following preprocessing steps:

1. Reads text from `Int.txt`.
2. Converts text to lowercase and removes leading or trailing whitespace.
3. Keeps sequences containing at least three characters.
4. Creates a character vocabulary and maps characters to numerical IDs.
5. Adds start and end tokens to the target sequences.
6. Reverses each input sequence to create the expected target.
7. Pads sequences to consistent lengths.
8. Creates training and validation datasets using a seeded random permutation.

If `Int.txt` is unavailable, the program creates a sample corpus containing words such as `hello`, `world`, `tensorflow`, and `encoder`.

## Training Configuration

| Parameter | Value |
|---|---|
| Embedding dimension | 32 |
| LSTM hidden units | 128 |
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Maximum epochs | 100 |
| Batch size | 64 |
| Early stopping patience | 10 epochs |
| Random seed | 42 |

The model uses weighted sparse categorical accuracy as a training metric. Padding tokens are assigned zero sample weight so they do not contribute to the loss or weighted metric.

Early stopping monitors validation loss and restores the best model weights when training stops.

## Training Process

The training process follows these steps:

1. The encoder processes the input sequence.
2. The encoder passes its hidden and cell states to the decoder.
3. The decoder receives the target sequence shifted by one position.
4. The model predicts the next character at each time step.
5. The predicted probabilities are compared with the expected character IDs.
6. The Adam optimizer updates the model parameters.
7. Validation loss is monitored to help prevent unnecessary training.

The shifted decoder inputs allow the model to learn next-character prediction during training, a technique commonly known as **teacher forcing**.

## Inference and Prediction

After training, separate encoder and decoder inference models are created.

The `reverse_with_model()` function performs the following operations:

1. Converts the input to lowercase.
2. Checks whether all characters exist in the training vocabulary.
3. Checks whether the input exceeds the maximum trained length.
4. Encodes and pads the input sequence.
5. Obtains the encoder's hidden and cell states.
6. Starts decoding with the start token.
7. Predicts one character at a time.
8. Stops when an end token or another stopping condition is reached.
9. Returns the generated character sequence.

The notebook tests the model on the first five training examples and displays the input, expected reversed output, and predicted output.

## How to Run

1. Open the notebook in Google Colab.
2. Upload `Int.txt` to the current working directory, or let the notebook generate a sample corpus.
3. Run all cells in order.
4. Wait for the model to complete training.
5. Review the best validation loss.
6. Check the predicted reversed sequences against their expected outputs.

Required libraries:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

These libraries are generally available in Google Colab.

## Expected Output

The notebook displays:

- Number of training examples
- Character vocabulary size
- Maximum input sequence length
- Neural network architecture summary
- Training and validation loss
- Best validation loss
- Input words, expected reversed words, and model predictions

Example of the expected task:

| Input | Expected Output |
|---|---|
| hello | olleh |
| world | dlrow |
| keras | sarek |

**Note:** These are examples of the intended task, not guaranteed predictions from every training run. The model may produce incorrect outputs, especially with a small training corpus or insufficient training.

## Key Concepts Learned

- Sequence-to-Sequence learning
- Character-level language representation
- LSTM encoder-decoder architecture
- Embedding layers
- Hidden and cell states
- Teacher forcing
- Padding and masking
- Sparse categorical cross-entropy
- Early stopping
- Autoregressive decoding and inference

## Limitations

- The model learns from a small text corpus and may not generalize to unseen words or characters.
- The validation set may be very small when the corpus contains few lines.
- The task is limited to reversing character sequences rather than translating or understanding natural language.
- The implementation does not use an attention mechanism; the decoder relies on the encoder's final hidden and cell states.
- A low validation loss does not necessarily guarantee correct reversal of every input word.

## Future Improvements

- Train on a larger and more diverse text corpus.
- Evaluate the model on a separate test dataset.
- Calculate exact-match accuracy for reversed sequences.
- Add attention to help the decoder access individual encoder outputs.
- Experiment with different embedding dimensions and LSTM units.
- Save and reload the trained model for future predictions.

## Author

**Annanya Kesharwani**
