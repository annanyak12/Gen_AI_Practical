# 🧠 NLP Analysis Studio

An interactive **Streamlit** dashboard for exploring a complete Natural Language Processing pipeline — from tokenization to dependency parsing — powered by **NLTK** and **spaCy**.

Built by **Annanya Kesharwani**.

---

## ✨ Features

The app walks through 9 core NLP pipeline steps, run individually or all at once:

1. **Sentence Segmentation** — split raw text into sentences
2. **Word Tokenization** — break text into individual tokens
3. **Stop Word Removal** — filter out common stop words
4. **Stemming** — reduce words to their root form (Porter Stemmer)
5. **Lemmatization** — reduce words to their dictionary base form (WordNet)
6. **POS Tagging** — assign part-of-speech tags to each token
7. **Named Entity Recognition (NER)** — detect entities like people, places, and organizations
8. **Dependency Parsing** — visualize grammatical relationships between tokens
9. **Noun Phrase Chunking** — extract noun phrases from the text

Additional UI features:
- Dark, professional theme with a custom control panel
- Editable input text box with live preview
- Run any single step or the entire pipeline in one click
- Results rendered as styled chips, cards, and interactive tables

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [NLTK](https://www.nltk.org/) — tokenization, stop words, stemming, lemmatization, POS tagging
- [spaCy](https://spacy.io/) — NER, dependency parsing, noun chunking (`en_core_web_sm` model)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install streamlit nltk spacy
python -m spacy download en_core_web_sm
```

### 3. Run the app

```bash
streamlit run nlp_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Running in Google Colab

Since Colab can't expose local ports directly, use [ngrok](https://ngrok.com/) to create a public URL:

```python
!pip install -q streamlit nltk spacy pyngrok
!python -m spacy download en_core_web_sm

# ... write nlp_app.py ...

!streamlit run nlp_app.py >/dev/null 2>&1 &

from pyngrok import ngrok
import os

# Set your ngrok auth token as an environment variable or Colab secret —
# never hardcode it in the notebook.
ngrok.set_auth_token(os.environ["NGROK_AUTH_TOKEN"])

public_url = ngrok.connect(addr=8501)
print(public_url)
```

> ⚠️ **Never commit an ngrok auth token (or any API key/secret) directly in a notebook or script.** Use environment variables, Colab secrets, or a `.env` file excluded via `.gitignore`.

---

## 📂 Project Structure

```
.
├── nlp_app.py       # Main Streamlit application
└── README.md        # Project documentation
```

---

## 📋 Requirements

- Python 3.8+
- streamlit
- nltk
- spacy (with `en_core_web_sm` model)

---

## 📝 License

This project is open source and available for personal and educational use.
