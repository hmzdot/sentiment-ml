import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from dataset import TweetDataset
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

DATASET_PATH = "./dataset.csv"
LEARNING_RATE = 2e-5
EPOCHS = 3

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1,
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = TweetDataset(DATASET_PATH, tokenizer=tokenizer)

# Split the data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
# elif torch.mps.is_available():
#     device_name = "mps"

print(f"Using device: {device_name}")
model.to(torch.device(device_name))

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
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

model_name = "./out/model_weights_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".pth"
torch.save(model.state_dict(), model_name)
