"""WordNet helpers and a simple Lesk-style word sense disambiguation algorithm."""
from __future__ import annotations
import re
from typing import Any

try:
    from nltk.corpus import wordnet as wn
except ImportError:  # Gives a clearer message if dependencies are not installed.
    wn = None

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "i", "you", "his", "her", "their", "my", "our",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "not", "very", "into", "after", "before", "up", "down", "about"
}

def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens, with common stopwords removed."""
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOPWORDS]

def _require_wordnet():
    if wn is None:
        raise RuntimeError("NLTK is missing. Install dependencies with: pip install -r requirements.txt")
    try:
        wn.synsets("bank")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK WordNet data is missing. Run `python setup_nltk.py` while connected to the internet."
        ) from exc

def get_sense_inventory(word: str) -> list[dict[str, Any]]:
    """Return the WordNet senses and useful lexical information for a word."""
    _require_wordnet()
    word = word.strip().lower().replace(" ", "_")
    if not word:
        return []
    results = []
    for i, syn in enumerate(wn.synsets(word), start=1):
        results.append({
            "sense_number": i,
            "synset": syn.name(),
            "part_of_speech": syn.pos(),
            "definition": syn.definition(),
            "examples": syn.examples(),
            "lemmas": sorted({lemma.name().replace("_", " ") for lemma in syn.lemmas()}),
            "hypernyms": [s.name() for s in syn.hypernyms()],
            "hyponyms": [s.name() for s in syn.hyponyms()[:8]],
        })
    return results

def _sense_signature(synset) -> set[str]:
    """Build a context signature from a synset and nearby lexical relations."""
    parts = [synset.definition(), " ".join(synset.examples())]
    parts.extend(lemma.name().replace("_", " ") for lemma in synset.lemmas())
    related = synset.hypernyms() + synset.hyponyms() + synset.similar_tos()
    for rel in related[:12]:
        parts.append(rel.definition())
        parts.extend(lemma.name().replace("_", " ") for lemma in rel.lemmas())
    return set(tokenize(" ".join(parts)))

def disambiguate(word: str, sentence: str) -> dict[str, Any]:
    """Choose a likely sense using contextual overlap, with first-sense fallback."""
    _require_wordnet()
    normalized_word = word.strip().lower().replace(" ", "_")
    synsets = wn.synsets(normalized_word)
    if not synsets:
        return {
            "word": word, "sentence": sentence, "found": False, "prediction": None,
            "definition": "No WordNet senses found for this word.",
            "score": 0, "context_overlap": [], "method": "Simplified Lesk (WordNet)"
        }

    context = set(tokenize(sentence))
    context.discard(normalized_word.replace("_", " "))
    context.discard(normalized_word.split("_")[0])
    scored = []
    for idx, syn in enumerate(synsets):
        overlap = sorted(context.intersection(_sense_signature(syn)))
        scored.append((len(overlap), -idx, syn, overlap))
    # Sort by score; ties preserve WordNet's sense order.
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score, _, best, overlap = scored[0]
    return {
        "word": word,
        "sentence": sentence,
        "found": True,
        "prediction": best.name(),
        "sense_number": synsets.index(best) + 1,
        "part_of_speech": best.pos(),
        "definition": best.definition(),
        "examples": best.examples(),
        "lemmas": sorted({lemma.name().replace("_", " ") for lemma in best.lemmas()}),
        "score": score,
        "context_overlap": overlap,
        "fallback_used": score == 0,
        "method": "Simplified Lesk (WordNet)"
    }

def analyze_samples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        result = disambiguate(row["word"], row["sentence"])
        result["expected_sense"] = row.get("expected_sense", "")
        result["notes"] = row.get("notes", "")
        results.append(result)
    return results

def make_report(results: list[dict[str, Any]]) -> str:
    """Create a concise human-readable analysis report."""
    total = len(results)
    found = sum(1 for r in results if r.get("found"))
    fallback = sum(1 for r in results if r.get("found") and r.get("fallback_used"))
    lines = [
        "WORD SENSE DISAMBIGUATION — ANALYSIS REPORT",
        "=" * 48,
        f"Sample sentences processed: {total}",
        f"Predictions returned: {found}",
        f"Predictions using first-sense fallback (no context overlap): {fallback}",
        "",
        "Results:",
    ]
    for i, r in enumerate(results, 1):
        pred = r.get("prediction") or "No sense found"
        definition = r.get("definition", "")
        overlap = ", ".join(r.get("context_overlap", [])) or "none"
        lines.append(f"{i}. {r.get('word')} — {r.get('sentence')}")
        lines.append(f"   Prediction: {pred} (score={r.get('score', 0)}; overlap={overlap})")
        lines.append(f"   Definition: {definition}")
    lines.extend([
        "",
        "Interpretation:",
        "This project uses a simplified Lesk-style algorithm. It selects the WordNet sense",
        "whose definition, examples, lemmas, and selected lexical relations overlap most",
        "with the sentence context. When overlap is zero, it falls back to WordNet's first",
        "sense. Predictions should be checked by a human because this baseline can be wrong.",
    ])
    return "\n".join(lines)
