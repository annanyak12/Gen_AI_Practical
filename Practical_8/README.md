# Implentation of a Transformer-based model using PyTorch and Hugging Face Transformers for Text classification or Text summarization, incorporating Prompt engineering techniques.

## Project Overview

This project demonstrates three Natural Language Processing (NLP) tasks using pretrained Transformer models: **text classification, text summarization, and question answering**.

The user enters a text, which is then processed by different models to classify its content, generate a short summary, and answer a question based on the provided text.

The project uses the Hugging Face Transformers library along with PyTorch and Pandas to perform these tasks.

## Objectives

- Understand the basic applications of Natural Language Processing.
- Perform text classification using a pretrained model.
- Generate a concise summary of user-provided text.
- Extract answers from a given context using a question-answering model.
- Learn how pretrained Transformer models can be used without training them from scratch.

## Technologies Used

- **Python** – Programming language
- **Hugging Face Transformers** – Pretrained NLP models and pipelines
- **T5-Small** – Text summarization
- **DistilBERT** – Extractive question answering
- **PyTorch** – Model inference
- **Pandas** – Displaying classification results
- **Google Colab** – Cloud-based development environment

## Project Workflow

The notebook follows these steps:

1. Accepts text from the user.
2. Performs text classification.
3. Generates a summary of the input text.
4. Answers a predefined question using the original text as context.
5. Displays the classification results, summary, and answer.

## Part 1: Text Classification

The notebook uses the Hugging Face `pipeline()` function to perform text classification.

```python
from transformers import pipeline

classifier = pipeline("text-classification")
outputs = classifier(text)
```

The model predicts a class label and a confidence score for the given text. The results are displayed in a Pandas DataFrame.

**Purpose:** To demonstrate how a pretrained model assigns a category or label to text.

Note: The default classification pipeline may select a sentiment-analysis model, depending on the installed Transformers configuration. Classification labels depend on the model selected.

## Part 2: Text Summarization

The notebook uses the pretrained `t5-small` model to generate a summary.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tok = AutoTokenizer.from_pretrained(model_name)
```

### How It Works

1. Adds the `summarize:` prefix to the input text.
2. Tokenizes the text into model-readable tokens.
3. Passes the tokens to the T5 model.
4. Generates a shorter version of the original text.
5. Decodes the generated tokens into readable text.

The model uses beam search with four beams and limits the generated summary to a maximum of 56 tokens.

**Purpose:** To extract the main idea of a longer text in a shorter form.

## Part 3: Question Answering

The notebook uses `distilbert-base-cased-distilled-squad`, a pretrained question-answering model.

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

qa_model_name = "distilbert-base-cased-distilled-squad"

qa_model = AutoModelForQuestionAnswering.from_pretrained(
    qa_model_name
)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
```

### How It Works

1. Defines a question and the input text as context.
2. Tokenizes both the question and context.
3. Passes the tokens through the question-answering model.
4. Predicts the start and end positions of the answer.
5. Converts the selected tokens into readable text.

The notebook uses the question:

`What can lead to big changes?`

The expected answer from the sample text is:

`Small steps`

**Purpose:** To demonstrate extractive question answering, where the model identifies an answer within the supplied context.

## How to Run

1. Open the notebook in Google Colab.
2. Install the required library:

```python
!pip install transformers
```

3. Run the cells in order.
4. Enter your text when prompted.
5. Wait for the pretrained models to load and process the input.
6. Review the classification, summary, and question-answering results.

PyTorch and Pandas are also required. Install them if they are not already available in your environment.

## Expected Output

The notebook produces three outputs:

| Task | Output |
|---|---|
| Text classification | Predicted label and confidence score |
| Text summarization | Short summary of the input |
| Question answering | Answer extracted from the context |

The exact classification label, summary, and answer depend on the input text and model predictions.

## Key Concepts Learned

- Natural Language Processing (NLP)
- Pretrained Transformer models
- Text classification pipelines
- Sequence-to-sequence models
- Text summarization
- Extractive question answering
- Tokenization and decoding
- Model inference using PyTorch
- Hugging Face Transformers

## Limitations

- The question is predefined rather than entered dynamically by the user.
- The summarization model may produce incomplete or inaccurate summaries.
- The question-answering model may return incorrect or empty answers if the relevant information is unclear.
- The default classification pipeline may not match every type of text.
- The first model download and initialization may take time.
- The notebook runs three separate models, which can require additional memory and processing time.

## Future Improvements

- Allow users to enter their own questions.
- Let users select different classification models.
- Add error handling for empty or very long inputs.
- Display classification, summary, and answers in a single organized output.
- Add support for multiple languages.
- Improve answer extraction by validating predicted start and end positions.

## Author

**Annanya Kesharwani**
