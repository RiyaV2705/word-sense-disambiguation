"""Interactive Streamlit interface for Word Sense Disambiguation."""
from pathlib import Path
import csv
import json
import pandas as pd
import streamlit as st
from wsd import get_sense_inventory, disambiguate, make_report

ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / "data" / "sample_sentences.csv"
OUT = ROOT / "outputs"

st.set_page_config(page_title="Word Sense Disambiguation", page_icon="📚", layout="wide")
st.title("📚 Word Sense Disambiguation using WordNet")
st.caption("Explore ambiguous words, inspect lexical relations, and predict a meaning from sentence context.")

with st.sidebar:
    st.header("About this project")
    st.write("A lightweight NLP mini-project using WordNet and a simplified Lesk algorithm.")
    st.markdown("**Method:** definition/example/lemma overlap + lexical relations")
    st.markdown("**Outputs:** sense inventory, predictions, CSV/JSON, analysis report")
    st.info("This is an educational baseline. Review predictions manually.")

tab1, tab2, tab3 = st.tabs(["Disambiguate a sentence", "WordNet sense inventory", "Run sample dataset"])

def ensure_output():
    OUT.mkdir(exist_ok=True)

def save_results(results):
    ensure_output()
    (OUT / "wsd_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["word", "sentence", "prediction", "sense_number", "part_of_speech", "definition", "score", "context_overlap", "fallback_used", "expected_sense", "notes"]
    rows = []
    for r in results:
        row = dict(r)
        row["context_overlap"] = ", ".join(r.get("context_overlap", []))
        rows.append({k: row.get(k, "") for k in fields})
    pd.DataFrame(rows, columns=fields).to_csv(OUT / "wsd_results.csv", index=False, encoding="utf-8-sig")
    (OUT / "analysis_report.txt").write_text(make_report(results), encoding="utf-8")

with tab1:
    st.subheader("Analyze one sentence")
    col1, col2 = st.columns([1, 2])
    with col1:
        word = st.text_input("Target word", value="bank", help="Enter a single English word.")
    with col2:
        sentence = st.text_input("Sentence", value="She deposited money at the bank.")
    if st.button("Disambiguate", type="primary"):
        if not word.strip() or not sentence.strip():
            st.warning("Enter both a target word and a sentence.")
        else:
            try:
                result = disambiguate(word, sentence)
                st.session_state["single_result"] = result
                save_results([result])
            except RuntimeError as exc:
                st.error(str(exc))
    if st.session_state.get("single_result"):
        r = st.session_state["single_result"]
        if not r.get("found"):
            st.warning(r["definition"])
        else:
            st.success(f"Predicted sense: {r['prediction']}")
            st.write("**Definition:**", r["definition"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Sense number", r["sense_number"])
            c2.metric("Context overlap score", r["score"])
            c3.metric("First-sense fallback", "Yes" if r.get("fallback_used") else "No")
            st.write("**Overlapping context words:**", ", ".join(r["context_overlap"]) or "None")
            st.write("**Synonyms / lemmas:**", ", ".join(r.get("lemmas", [])))
            if r.get("examples"):
                st.write("**WordNet examples:**")
                for ex in r["examples"]:
                    st.write(f"- {ex}")

with tab2:
    st.subheader("Explore WordNet senses")
    inventory_word = st.text_input("Word to explore", value="bank", key="inventory_word")
    if st.button("Show sense inventory"):
        try:
            st.session_state["inventory"] = get_sense_inventory(inventory_word)
        except RuntimeError as exc:
            st.error(str(exc))
    inventory = st.session_state.get("inventory", [])
    if inventory:
        st.write(f"Found **{len(inventory)}** senses for **{inventory_word}**.")
        for sense in inventory:
            with st.expander(f"Sense {sense['sense_number']}: {sense['synset']} ({sense['part_of_speech']})"):
                st.write("**Definition:**", sense["definition"])
                st.write("**Synonyms:**", ", ".join(sense["lemmas"]))
                st.write("**Examples:**", "; ".join(sense["examples"]) or "No examples provided")
                st.write("**Hypernyms (broader concepts):**", ", ".join(sense["hypernyms"]) or "None listed")
                st.write("**Hyponyms (more specific concepts):**", ", ".join(sense["hyponyms"]) or "None listed")

with tab3:
    st.subheader("Evaluate bundled sample sentences")
    st.write("The sample set contains sentences with words that have multiple meanings.")
    if st.button("Run all samples", type="primary"):
        try:
            samples = pd.read_csv(SAMPLE_FILE).fillna("").to_dict(orient="records")
            results = []
            for row in samples:
                result = disambiguate(str(row["word"]), str(row["sentence"]))
                result["expected_sense"] = str(row.get("expected_sense", ""))
                result["notes"] = str(row.get("notes", ""))
                results.append(result)
            st.session_state["sample_results"] = results
            save_results(results)
        except RuntimeError as exc:
            st.error(str(exc))
    results = st.session_state.get("sample_results", [])
    if results:
        display = []
        for r in results:
            display.append({
                "Target word": r["word"],
                "Sentence": r["sentence"],
                "Predicted synset": r.get("prediction") or "No sense found",
                "Definition": r["definition"],
                "Overlap score": r["score"],
                "Overlap words": ", ".join(r["context_overlap"]) or "none",
                "Fallback used": r.get("fallback_used", False),
            })
        st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)
        report = make_report(results)
        st.download_button("Download analysis report (.txt)", report, file_name="analysis_report.txt", mime="text/plain")
        st.download_button("Download predictions (.json)", json.dumps(results, indent=2, ensure_ascii=False), file_name="wsd_results.json", mime="application/json")
        csv_data = pd.DataFrame([{**r, "context_overlap": ", ".join(r.get("context_overlap", [])), "examples": "; ".join(r.get("examples", [])), "lemmas": ", ".join(r.get("lemmas", []))} for r in results]).to_csv(index=False)
        st.download_button("Download predictions (.csv)", csv_data, file_name="wsd_results.csv", mime="text/csv")

st.divider()
st.caption("Built with Python, NLTK WordNet, Pandas, and Streamlit.")
