
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load and preprocess the CSV
df = pd.read_csv("api_prompts_augmented.csv")
df = df.dropna(subset=["Prompt", "URL_Label"])
df["URL_Label"] = df["URL_Label"].astype(str)

# Drop labels with fewer than 2 samples
label_counts = df["URL_Label"].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df = df[df["URL_Label"].isin(valid_labels)]


# Encode labels
labels = sorted(df["URL_Label"].unique())
label_to_id = {label: i for i, label in enumerate(labels)}
df["label_id"] = df["URL_Label"].map(label_to_id)

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Prompt"].tolist(), df["label_id"].tolist(), test_size=0.2, stratify=df["label_id"], random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

# Dataset class
class PromptDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = PromptDataset(train_encodings, train_labels)
val_dataset = PromptDataset(val_encodings, val_labels)

# Model setup
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Training loop
for epoch in range(8):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Final Accuracy: {accuracy:.4f}")

# Save model and tokenizer
model_dir = "augment_iei_model"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Save results
results_dir = "results-augment"
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "augment-accuracy.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}")