"""Convert the fine-tuned BERT from the notebook (.pth) to an 8-bit ONNX model for transformers.js, then measure its F1.

Usage:  python export_bert.py path/to/best_bert_final_v2.pth
Writes: model/bert/ (config, tokenizer, onnx/model_quantized.onnx); upload that folder to a Hugging Face model repo.
"""
import sys
from pathlib import Path

import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

OUT = Path("model/bert")
(OUT / "onnx").mkdir(parents=True, exist_ok=True)
INPUTS = ["input_ids", "attention_mask", "token_type_ids"]

config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2, architectures=["BertForSequenceClassification"],
                                    id2label={0: "Fake", 1: "Real"}, label2id={"Fake": 0, "Real": 1})
model = BertForSequenceClassification(config).eval()
model.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=True))
tok = BertTokenizerFast.from_pretrained("bert-base-uncased", model_max_length=256)  # 256 = training length (step 13)
config.save_pretrained(OUT)
tok.save_pretrained(OUT)

fp32 = OUT / "onnx" / "model.onnx"
example = tok(["an example sentence"], return_tensors="pt")
torch.onnx.export(
    model, tuple(example[k] for k in INPUTS), fp32, input_names=INPUTS, output_names=["logits"],
    dynamic_axes={**{k: {0: "batch", 1: "sequence"} for k in INPUTS}, "logits": {0: "batch"}},
    opset_version=17, dynamo=False,
)
quantize_dynamic(fp32, OUT / "onnx" / "model_quantized.onnx", weight_type=QuantType.QInt8)
fp32.unlink()  # only the 8-bit model is published (~4x smaller)

session = ort.InferenceSession(OUT / "onnx" / "model_quantized.onnx")
test = pd.read_csv("data/test_data.csv")
preds = []
for text in test["cleaned_text"]:
    enc = tok(text, truncation=True, max_length=256, return_tensors="np")
    preds.append(int(session.run(None, {k: enc[k] for k in INPUTS})[0].argmax()))
print(f"8-bit BERT on test_data.csv: accuracy {accuracy_score(test.label, preds):.4f}, F1 {f1_score(test.label, preds):.4f}")
for f in sorted(OUT.rglob("*.*")):
    print(f"  {f}  {f.stat().st_size // 1024} KB")
