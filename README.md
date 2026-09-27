---
title: Fake News Detector
emoji: 📰
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: MSc dissertation models (SVM + BERT) running in the browser
---

# Fake News Detector

A website for checking whether a news article, headline or tweet reads like real or fake news. It serves the
models from my MSc dissertation, *Fake News Detection Using Transformer-Based and Traditional NLP Models*
(MSc Data Science and Artificial Intelligence, University of Liverpool).

**Live site:** https://huggingface.co/spaces/haithamd/fake-news-detector<br>
**Code:** https://github.com/haithamdamiri/fake-news-detector

- **Check**: TF-IDF + linear SVM, ported to plain JavaScript. Instant, with word-level explanations.
- **Deep check**: fine-tuned BERT (bert-base-uncased), exported to 8-bit ONNX and run with
  [Transformers.js](https://huggingface.co/docs/transformers.js). Model: [haithamd/fake-news-bert](https://huggingface.co/haithamd/fake-news-bert)
- Accepts pasted text and tweet links (read through X's public oEmbed endpoint).
- No server: both models run in the visitor's browser, so the site is free to host and checked text stays on the device.

Trained on the [Kaggle Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
(44,898 articles). The models judge **writing style, not truth**: 99% of the "real" training articles are Reuters wire
copy. See the site's "What it can't do" section.

Original experiments: [Google Colab notebook](https://colab.research.google.com/drive/1fxLCdhzatMOxjFAlzxf5ljk_AhpYDw4T?usp=sharing)

## Results

| Model | Test F1 (dissertation) |
|---|---|
| TF-IDF + Logistic Regression | 0.9719 |
| TF-IDF + Linear SVM | 0.9798 |
| BERT | 0.9825 |
| RoBERTa | 0.9765 |

## Files

| File | What it does |
|---|---|
| `index.html` | The website: UI, tweet fetching, BERT via Transformers.js |
| `model.js` | The SVM in JavaScript: the notebook's text cleaning, TF-IDF and calibrated SVM |
| `model/svm.json` | Exported SVM: vocabulary, IDF, weights, calibration, stop words, lemma table |
| `train_svm.py` | Retrains the SVM from `Fake.csv` / `True.csv` (notebook steps 3–12) and exports `model/svm.json` |
| `export_bert.py` | Converts the notebook's fine-tuned BERT (`.pth`) to 8-bit ONNX and measures its F1 |
| `check.mjs` | Proves the JavaScript SVM gives the same results as Python |

## Run locally

```bash
python -m http.server 8000
```

Then open http://localhost:8000. On localhost the page loads BERT from `model/bert/` (created by `export_bert.py`).

To rebuild the models (Python 3.12):

```bash
pip install -r requirements.txt
python train_svm.py path/to/Datasets-Fake-True
python export_bert.py path/to/best_bert_final_v2.pth
node --experimental-detect-module check.mjs
```
