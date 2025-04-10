from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from pymongo import MongoClient
import numpy as np
import evaluate
import os

# === Step 1: Load data from MongoDB ===
try:
    username = "jeremyflagg12"
    password = "QGTrn5lbWa2qrXFL"
    conn_str = f"mongodb+srv://{username}:{password}@cluster0.t4orq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(conn_str)
except Exception as e:
    print("MongoDB connection failed:", e)
    exit()

db = client["Prompts"]
collection = db["search2"]
documents = list(collection.find())

# === Step 2: Prepare DataFrame ===
data = []
for doc in documents:
    if "URL_Label" in doc and "Response" in doc and "Prompt" in doc:
        label = doc["URL_Label"]
        url = doc["Response"]
        prompts = doc["Prompt"]
        for prompt in prompts:
            data.append({
                "Label": str(label),
                "Url": url,
                "Prompt": prompt
            })
    else:
        print("Skipping doc due to missing fields:", doc)

df = pd.DataFrame(data)

# === Step 3: Train/Val Split ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# === Step 4: Label Mapping ===
intent_labels = df['Label'].unique()
label2id = {label: idx for idx, label in enumerate(intent_labels)}
id2label = {idx: label for label, idx in label2id.items()}

train_dataset = train_dataset.map(lambda x: {'label': label2id[x['Label']]})
val_dataset = val_dataset.map(lambda x: {'label': label2id[x['Label']]})

# === Step 5: Tokenization ===
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['Prompt'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# === Step 6: Model & Collator ===
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(intent_labels),
    id2label=id2label,
    label2id=label2id
)

# === Step 7: Metrics ===
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

# === Step 8: Training Arguments ===
training_args = TrainingArguments(
    output_dir="./iei-results5",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to=[]
)

# === Step 9: Resume Checkpoint Logic ===
checkpoint_dir = "./iei-results5"
last_checkpoint = None

if os.path.isdir(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    if checkpoints:
        last_checkpoint = checkpoints[-2] if len(checkpoints) >= 2 else checkpoints[-1]
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoints found. Starting fresh.")
else:
    print("No checkpoint directory found. Starting fresh.")

# === Step 10: Device Setup & Confirmation ===
if not torch.cuda.is_available():
    raise EnvironmentError("❌ CUDA is NOT available. Ensure you're using a CUDA-enabled PyTorch environment.")

device = torch.device("cuda")
model.to(device)

print(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
print(f"Model is on device: {next(model.parameters()).device}")

# === Step 11: Trainer Setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# === Step 12: Train ===
trainer.train(resume_from_checkpoint=last_checkpoint)

# === Step 13: Evaluate & Save ===
eval_result = trainer.evaluate()
print("Evaluation Results:", eval_result)

model.save_pretrained("./fine-tuned-model-iei5")
tokenizer.save_pretrained("./fine-tuned-model-iei5")

# === Step 14: Save Predictions to CSV ===
predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

decoded_preds = [id2label[i] for i in pred_labels]
decoded_true = [id2label[i] for i in true_labels]
prompts = val_dataset['Prompt']

output_df = pd.DataFrame({
    "Prompt": prompts,
    "TrueLabel": decoded_true,
    "PredictedLabel": decoded_preds
})

output_df.to_csv("predictions_output3.csv", index=False)
print("Predictions saved to predictions_output3.csv")
