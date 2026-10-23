# Word Sense Disambiguation (WSD) using WordNet

A beginner-friendly NLP mini-project that identifies the likely meaning of an ambiguous word in a sentence using WordNet and a simplified Lesk algorithm.

## Features
- Explore possible WordNet senses for an ambiguous word.
- Display definitions, examples, synonyms, and lexical relations.
- Disambiguate a target word in a sentence using context overlap (Lesk-style).
- Run the algorithm on sample sentences.
- Export annotations and predictions to CSV/JSON.
- Create a short analysis report.
- Streamlit interface plus a command-line runner.

## Requirements
- Python 3.10 or newer
- Internet access for the first-time NLTK WordNet download (unless WordNet is already installed)
- Packages listed in `requirements.txt`

## Windows PowerShell setup (VS Code)

Open PowerShell in this project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_nltk.py
streamlit run app.py
```

If PowerShell blocks activation, you can avoid activation and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_nltk.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

### Command-line mode
After installing requirements and downloading WordNet:

```powershell
python wsd_cli.py --word bank --sentence "She deposited money at the bank."
python wsd_cli.py --samples
```

### If WordNet cannot be downloaded
Run `python setup_nltk.py` again while connected to the internet. On a restricted network, download the NLTK `wordnet` and `omw-1.4` corpora using NLTK's downloader or place them in a standard NLTK data directory. The app displays an actionable error if WordNet is unavailable.

## Project structure

```text
word_sense_disambiguation_project/
├── app.py                 # Streamlit user interface
├── wsd.py                 # WordNet utilities and Lesk-style algorithm
├── wsd_cli.py             # Command-line interface
├── setup_nltk.py          # Downloads required NLTK data
├── requirements.txt
├── data/sample_sentences.csv
├── outputs/               # Generated CSV, JSON, and report files
└── tests/test_wsd.py      # Unit tests for tokenization/scoring helpers
```

## Algorithm overview
1. Normalize the sentence and tokenize it.
2. Retrieve WordNet synsets for the target word.
3. For each sense, collect words from its definition, examples, lemma names, and related synsets.
4. Compare the sense vocabulary with the sentence context, excluding the target word.
5. Select the sense with the largest overlap. If all scores are zero, use WordNet's first (most frequent) sense as a fallback.
6. Display the predicted sense and supporting evidence.

This is a lightweight educational WSD system, not a substitute for human annotation. The simplified Lesk method can make mistakes, especially when context is short.

## Deliverables
- Sense inventory for a target word
- WSD prediction and annotation output for sample text
- CSV/JSON exports
- Short analysis report
- Source code and tests

## Example words
`bank`, `bat`, `apple`, `light`, `plant`, `crane`, `rock`, `spring`

## Troubleshooting
- **`ModuleNotFoundError: No module named 'streamlit'`**: activate the project environment and run `pip install -r requirements.txt`.
- **WordNet resource not found**: run `python setup_nltk.py`.
- **No sense found**: check spelling and use a single English word, e.g. `bank`.
