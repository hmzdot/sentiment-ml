import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer, logging

logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment")
    parser.add_argument("model_snapshot", help="Path to the trained model snapshot")
    parser.add_argument("text", help="Text to analyze sentiment for")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval(model_snapshot: str, text: str) -> float:
    device = get_device()

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=1,
    )
    model.load_state_dict(torch.load(model_snapshot, weights_only=True))
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    input = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move inputs to same device as model
    input = {k: v.to(device) for k, v in input.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=input["input_ids"],
            attention_mask=input["attention_mask"],
        )
        # Move output back to CPU for display
        logits = outputs.logits.squeeze().cpu()
        sentiment = logits.item()

    return sentiment


if __name__ == "__main__":
    model_snapshot, text = parse_args()
    sentiment = eval(model_snapshot, text)

    sentiment_text = "Positive" if sentiment > 0.5 else "Negative"
    print(f"Sentiment: {sentiment_text} ({sentiment:.2f})")
