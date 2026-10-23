"""Download NLTK corpora required by this project."""
import nltk

def main():
    packages = ("wordnet", "omw-1.4", "punkt")
    for package in packages:
        print(f"Downloading/checking NLTK package: {package}")
        try:
            ok = nltk.download(package, quiet=False)
            if not ok:
                print(f"Warning: download of {package} did not report success.")
        except Exception as exc:
            print(f"Could not download {package}: {exc}")
    # Validate WordNet availability.
    try:
        from nltk.corpus import wordnet as wn
        print(f"WordNet ready. Example: {len(wn.synsets('bank'))} senses for 'bank'.")
    except LookupError:
        print("WordNet is not available yet. Check internet access and run this script again.")

if __name__ == "__main__":
    main()
