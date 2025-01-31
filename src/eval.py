import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment")
    parser.add_argument("model_snapshot", help="Path to the trained model snapshot")
    parser.add_argument("text", help="Text to analyze sentiment for")
    return parser.parse_args()


args = parse_args()
model_snapshot = args.model_snapshot
text = args.text

if (model_snapshot is None) or (text is None):
    print("Usage: python eval.py <model_snapshot> <text>")
    exit(1)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1,
)
model.load_state_dict(torch.load(model_snapshot, weights_only=True))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
input = tokenizer(
    text,
    max_length=256,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

outputs = model(
    input_ids=input["input_ids"],
    attention_mask=input["attention_mask"],
)
logits = outputs.logits.squeeze()
sentiment = logits.item()

sentiment_readable = "Positive" if sentiment > 0.0 else "Negative"
print(f"Sentiment: {sentiment_readable} ({sentiment})")
