"""Retrain the dissertation's TF-IDF + linear SVM (notebook steps 3-12) and export it for the browser.

Usage:  python train_svm.py path/to/folder/with/Fake.csv+True.csv
Writes: model/svm.json (everything model.js needs) and data/parity.json (for check.mjs)
"""
import json
import re
import sys
import warnings
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords, wordnet  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
STOPWORDS = set(stopwords.words("english"))
lemmatize = WordNetLemmatizer().lemmatize


def clean(text: str) -> str:
    """Notebook steps 5 and 10: strip HTML/URLs, keep alphabetic non-stopwords, lemmatise, cap at 1000 words."""
    text = BeautifulSoup(text, "lxml").get_text(separator=" ")
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    words = [lemmatize(w.lower()) for w in text.split() if w.lower().isalpha() and w.lower() not in STOPWORDS]
    return " ".join(words[:1000])


data = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
df = pd.concat([pd.read_csv(data / "Fake.csv").assign(label=0),
                pd.read_csv(data / "True.csv").assign(label=1)], ignore_index=True)
df["raw"] = df["title"] + " " + df["text"]
df = df[["subject", "label", "raw"]].drop_duplicates()
print("Cleaning", len(df), "articles (about a minute)...")
df["cleaned_text"] = df["raw"].map(clean)
df = df[df["cleaned_text"].str.strip() != ""]

# Same 80/10/10 stratified split as the notebook, so the test set matches test_data.csv.
train, rest = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
valid, test = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest["label"])

# Notebook used SVC(kernel="linear"); LinearSVC + calibration is the same model family and adds probabilities.
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("smote", SMOTE(random_state=42)),
    ("clf", CalibratedClassifierCV(LinearSVC(random_state=42))),
])
pipe.fit(train["cleaned_text"], train["label"])
for name, split in [("Validation", valid), ("Test", test)]:
    pred = pipe.predict(split["cleaned_text"])
    print(f"{name}: accuracy {accuracy_score(split['label'], pred):.4f}, F1 {f1_score(split['label'], pred):.4f}")

# ---- Export for the browser ----
tfidf, folds = pipe["tfidf"], pipe["clf"].calibrated_classifiers_
vocab = tfidf.get_feature_names_out()
in_vocab = set(vocab)

# WordNet lemmatiser as a lookup table: every word form that lemmatises to something else and matters to the model.
# Candidates = every word seen in the dataset + WordNet's plural rules applied to the vocabulary + its exception list.
seen = {w.lower() for t in df["raw"] for w in t.split()}
rules = [("s", ""), ("ses", "s"), ("ves", "f"), ("xes", "x"), ("zes", "z"), ("ches", "ch"), ("shes", "sh"), ("men", "man"), ("ies", "y")]
plurals = {v[: len(v) - len(base)] + suffix for v in vocab for suffix, base in rules if v.endswith(base)}
lemmas = {}
for w in seen | plurals | set(wordnet._exception_map["n"]):
    if w.isalpha() and w not in STOPWORDS and (lemma := lemmatize(w)) != w and (lemma in in_vocab or w in in_vocab):
        lemmas[w] = lemma

r6 = lambda a: [float(f"{x:.6g}") for x in a]  # noqa: E731
Path("model").mkdir(exist_ok=True)
Path("model/svm.json").write_text(json.dumps({
    "vocab": list(vocab),
    "idf": r6(tfidf.idf_),
    # Each fold: p(real) = 1 / (1 + exp(a * (w.x + b) + c)); the model averages the folds.
    "folds": [{"w": r6(f.estimator.coef_[0]), "b": float(f.estimator.intercept_[0]),
               "a": float(f.calibrators[0].a_), "c": float(f.calibrators[0].b_)} for f in folds],
    "stopwords": sorted(STOPWORDS),
    "lemmas": lemmas,
}, separators=(",", ":")))
print(f"Saved model/svm.json ({Path('model/svm.json').stat().st_size // 1024} KB, {len(lemmas)} lemma entries)")

# Raw test articles with Python's answers, so check.mjs can prove the JavaScript port gives the same results.
sample = test.sample(300, random_state=0)
Path("data").mkdir(exist_ok=True)
Path("data/parity.json").write_text(json.dumps([
    {"raw": r, "cleaned": c, "p_real": float(p)}
    for r, c, p in zip(sample["raw"], sample["cleaned_text"], pipe.predict_proba(sample["cleaned_text"])[:, 1])
]))
