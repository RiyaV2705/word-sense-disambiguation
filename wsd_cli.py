"""Command-line runner for the WordNet WSD project."""
import argparse
import csv
import json
from pathlib import Path
from wsd import disambiguate, make_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sample_sentences.csv"
OUT = ROOT / "outputs"

def load_samples():
    with DATA.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def save_results(results):
    OUT.mkdir(exist_ok=True)
    (OUT / "wsd_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["word", "sentence", "prediction", "sense_number", "definition", "score", "context_overlap", "fallback_used", "expected_sense", "notes"]
    with (OUT / "wsd_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["context_overlap"] = ", ".join(row.get("context_overlap", []))
            writer.writerow(row)
    (OUT / "analysis_report.txt").write_text(make_report(results), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Word Sense Disambiguation using WordNet")
    parser.add_argument("--word", help="Ambiguous target word, e.g. bank")
    parser.add_argument("--sentence", help="Sentence containing the target word")
    parser.add_argument("--samples", action="store_true", help="Run all bundled sample sentences")
    args = parser.parse_args()
    try:
        if args.samples or (not args.word and not args.sentence):
            results = []
            for row in load_samples():
                result = disambiguate(row["word"], row["sentence"])
                result["expected_sense"] = row.get("expected_sense", "")
                result["notes"] = row.get("notes", "")
                results.append(result)
            save_results(results)
            print(make_report(results))
            print(f"\nFiles saved in: {OUT}")
        elif args.word and args.sentence:
            result = disambiguate(args.word, args.sentence)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            save_results([result])
            print(f"\nFiles saved in: {OUT}")
        else:
            parser.error("Provide both --word and --sentence, or use --samples.")
    except RuntimeError as exc:
        raise SystemExit(str(exc))

if __name__ == "__main__":
    main()
