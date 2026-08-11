# 🧠 NLP Analysis Studio

An interactive **Streamlit** dashboard for exploring the complete **Natural Language Processing (NLP) pipeline** using **NLTK** and **spaCy**. The application allows users to perform and visualize fundamental NLP tasks through a simple and user-friendly web interface.

Developed by **Annanya Kesharwani**.

---

## 📖 Overview

This project demonstrates the implementation of essential NLP techniques using Python libraries **NLTK** and **spaCy**. The application is built with **Streamlit**, providing an interactive interface where users can enter any paragraph and analyze it through different stages of the NLP pipeline.

The project is implemented inside a **Google Colab Notebook**, which generates the Streamlit application (`nlp_app.py`) using the `%%writefile` command.

---

## ✨ Features

The application performs the following NLP operations:

### 1. Sentence Segmentation
Splits the input paragraph into individual sentences.

### 2. Word Tokenization
Breaks each sentence into individual words or tokens.

### 3. Stop Word Removal
Removes commonly occurring words (such as *the*, *is*, *and*) that carry little semantic meaning.

### 4. Stemming
Reduces words to their root form using the **Porter Stemmer**.

### 5. Lemmatization
Converts words into their dictionary base form using **WordNet Lemmatizer**.

### 6. Part-of-Speech (POS) Tagging
Identifies the grammatical role of each word.

### 7. Named Entity Recognition (NER)
Detects entities such as:

- Person
- Organization
- Location
- Date
- Time
- Money
- Others

using spaCy's pre-trained model.

### 8. Dependency Parsing
Visualizes grammatical relationships between words using spaCy's dependency parser.

### 9. Noun Phrase Chunking
Extracts meaningful noun phrases from the input text.

### Additional Features

- Interactive Streamlit dashboard
- Sidebar-based NLP pipeline selection
- Editable input text area
- Run individual NLP steps
- Run complete NLP pipeline
- Structured tables for outputs
- Dependency tree visualization
- Responsive and modern UI

---

## 🛠 Technologies Used

- Python
- Streamlit
- NLTK
- spaCy
- Google Colab
- pyngrok (for deployment in Colab)

---

## 📚 Python Libraries

```text
streamlit
nltk
spacy
pyngrok
```

---

## 📂 Project Structure

```
.
├── NLP_Analysis_Studio.ipynb      # Google Colab notebook containing the project
├── nlp_app.py                     # Generated Streamlit application (created using %%writefile)
└── README.md                      # Project documentation
```

---

## ⚙ Installation

### Install Required Libraries

```bash
pip install streamlit nltk spacy pyngrok
```

---

### Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the Project Locally

After generating or saving `nlp_app.py`, run:

```bash
streamlit run nlp_app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## ☁ Running in Google Colab

### Step 1

Install required packages.

```python
!pip install -q streamlit nltk spacy pyngrok
```

---

### Step 2

Download the spaCy model.

```python
!python -m spacy download en_core_web_sm
```

---

### Step 3

Create the Streamlit application.

```python
%%writefile nlp_app.py
# Paste the Streamlit code here
```

---

### Step 4

Run the Streamlit server.

```python
!streamlit run nlp_app.py >/dev/null 2>&1 &
```

---

### Step 5

Create a public URL using ngrok.

```python
from pyngrok import ngrok
import os

ngrok.set_auth_token(os.environ["NGROK_AUTH_TOKEN"])
