// Proves the browser SVM matches Python:  node check.mjs  (after python train_svm.py; needs data/test_data.csv)
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { loadSvm, clean, predictSvm } from "./model.js";

const m = loadSvm(JSON.parse(readFileSync("model/svm.json", "utf8")));

// 1. Same cleaning and same probabilities as Python on 300 raw test articles.
// The lemma table only covers words the model scores, so compare the words that are in the vocabulary.
const scored = (words) => words.filter((w) => m.index.has(w)).join(" ");
const parity = JSON.parse(readFileSync("data/parity.json", "utf8"));
let cleanDiffs = 0, maxDiff = 0;
for (const { raw, cleaned, p_real } of parity) {
  const words = clean(raw, m);
  if (scored(words) !== scored(cleaned.split(" "))) cleanDiffs++;
  maxDiff = Math.max(maxDiff, Math.abs(predictSvm(words, m).p - p_real));
}
console.log(`Scored words differ from Python on ${cleanDiffs}/${parity.length} raw articles; max P(real) difference ${maxDiff.toExponential(1)}`);
assert.ok(cleanDiffs === 0 && maxDiff < 1e-3);

// 2. Dissertation test set (already cleaned): F1 for the Real class, as in the notebook.
const rows = readFileSync("data/test_data.csv", "utf8").trim().split(/\r?\n/).slice(1);
let tp = 0, fp = 0, fn = 0;
for (const row of rows) {
  const cut = row.lastIndexOf(","), label = row.slice(cut + 1) === "1";
  const pred = predictSvm(row.slice(0, cut).replace(/^"|"$/g, "").split(" "), m).p >= 0.5;
  if (pred && label) tp++; else if (pred) fp++; else if (label) fn++;
}
const f1 = (2 * tp) / (2 * tp + fp + fn);
console.log(`Browser SVM F1 on test_data.csv (${rows.length} rows): ${f1.toFixed(4)}`);
assert.ok(f1 > 0.975);

// 3. Obvious cases.
const verdict = (t) => predictSvm(clean(t, m), m).p;
assert.ok(verdict("BREAKING: They are hiding the truth! WAKE UP PEOPLE! Share before it gets deleted!!") < 0.5);
assert.deepEqual(clean("The Senators were voting on https://x.com today!", m), ["senator", "voting"]);
console.log("All checks passed.");
