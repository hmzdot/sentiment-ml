import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from dataset import TweetDataset
from transformers import BertForSequenceClassification, BertTokenizer, logging
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.set_verbosity_error()
writer = SummaryWriter()


def train(dataset_path: str, learning_rate: float, epochs: int):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=1,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = TweetDataset(dataset_path, tokenizer=tokenizer)

    # Split the data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    # elif torch.mps.is_available():
    #     device_name = "mps"

    print(f"Using device: {device_name}")
    model.to(torch.device(device_name))

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        i = epoch * len(train_loader)
        for input, mask, labels in tqdm(train_loader, desc=f"Epoch #{epoch+1}"):
            input = input.to(device_name)
            mask = mask.to(device_name)
            labels = labels.to(device_name)

            # Forward pass
            outputs = model(input_ids=input, attention_mask=mask)
            logits = outputs.logits.squeeze()

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            writer.add_scalar("Loss/train", loss.item(), i)
            i += 1

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input, mask, labels in val_loader:
            input = input.to(device_name)
            mask = mask.to(device_name)
            labels = labels.to(device_name)

            outputs = model(input_ids=input, attention_mask=mask)
            logits = outputs.logits.squeeze()

            loss = criterion(logits, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")

    model_name = (
        "./snapshots/model_weights_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".pth"
    )
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    train(
        dataset_path="./data/tweets.csv",
        learning_rate=2e-5,
        epochs=3,
    )
    writer.flush()
