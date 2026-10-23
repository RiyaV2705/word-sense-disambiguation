# Mini Project Report: Word Sense Disambiguation using WordNet

## 1. Aim
To develop a word sense disambiguation system that identifies the intended meaning of an ambiguous word from its surrounding sentence context.

## 2. Objectives
- Identify multiple senses of ambiguous English words.
- Explore definitions, synonyms, examples, hypernyms, and hyponyms in WordNet.
- Apply a Lesk-style overlap algorithm to sample sentences.
- Generate machine-readable predictions and a short analysis report.

## 3. Tools and Technologies
Python, NLTK WordNet, Pandas, Streamlit, CSV, JSON, and unittest.

## 4. Methodology
The system accepts a target word and a sentence. It retrieves the word's WordNet synsets, creates a signature for each sense using its definition, examples, lemma names, and selected related synsets, then measures overlap with the sentence context. The highest-overlap sense is selected. If all scores are zero, the system uses WordNet's first sense as a baseline fallback.

## 5. Deliverables
- Interactive sense inventory and lexical-relation explorer.
- Single-sentence WSD prediction.
- Batch predictions for the bundled sample sentences.
- CSV and JSON exports.
- Automatically generated analysis report.

## 6. Evaluation and Discussion
The bundled examples cover bank, bat, light, plant, crane, spring, rock, and apple. Results are intended for inspection rather than claiming a validated accuracy score: the sample set is small, and the baseline uses lexical overlap rather than a trained supervised model. A zero-overlap result triggers the first-sense fallback. Expected labels in the sample CSV are illustrative descriptions, not formal WordNet synset IDs.

## 7. Conclusion
The project demonstrates how lexical resources and contextual overlap can be combined for a basic WSD system. WordNet provides sense inventories and semantic relations, while the simplified Lesk method offers an explainable baseline. Human review remains important because lexical overlap can miss implicit context.
