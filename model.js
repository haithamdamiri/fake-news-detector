// In-browser version of the dissertation's TF-IDF + linear SVM. Mirrors clean() in train_svm.py exactly.

const URLS = /http\S+|www\S+|https\S+/g;
const ALPHA = /^\p{L}+$/u; // Python's str.isalpha()

export function loadSvm(json) {
  const n = json.vocab.length;
  const meanW = new Float64Array(n);
  for (const f of json.folds) for (let i = 0; i < n; i++) meanW[i] += f.w[i] / json.folds.length;
  return { ...json, index: new Map(json.vocab.map((w, i) => [w, i])), stop: new Set(json.stopwords), meanW };
}

// Cleaned form of one whitespace token, or null if the notebook's cleaning drops it.
export function keep(token, m) {
  const w = token.toLowerCase();
  return ALPHA.test(w) && !m.stop.has(w) ? (m.lemmas[w] ?? w) : null;
}

// Text must already be free of HTML tags (the page uses DOMParser for that).
export function clean(text, m) {
  const words = [];
  for (const t of text.replace(URLS, "").split(/\s+/)) {
    const k = keep(t, m);
    if (k) words.push(k);
  }
  return words.slice(0, 1000);
}

// Returns P(real) and each scored word's contribution (> 0 pushes toward Real).
export function predictSvm(words, m) {
  const counts = new Map();
  for (const w of words) {
    const i = m.index.get(w);
    if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1);
  }
  const x = new Map([...counts].map(([i, c]) => [i, c * m.idf[i]]));
  const norm = Math.hypot(...x.values()) || 1;
  for (const [i, v] of x) x.set(i, v / norm);

  let p = 0;
  for (const f of m.folds) {
    let d = f.b;
    for (const [i, v] of x) d += v * f.w[i];
    p += 1 / (1 + Math.exp(f.a * d + f.c)) / m.folds.length;
  }
  const weights = new Map([...x].map(([i, v]) => [m.vocab[i], v * m.meanW[i]]));
  return { p, weights };
}
