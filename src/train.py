import torch
import torch.optim as optim
import torch.nn as nn
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for input, mask, labels in tqdm(train_loader, desc=f"Epoch #{epoch+1}"):
        pass
