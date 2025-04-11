import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# ========== STEP 1: Load CSV ==========
df = pd.read_csv("iei_search.csv")
df.columns = [col.strip().lower() for col in df.columns]
assert "prompt" in df.columns and "response" in df.columns

# ========== STEP 2: Encode Response URLs as labels ==========
response_labels = sorted(df["response"].unique())
label2id = {url: i for i, url in enumerate(response_labels)}
id2label = {i: url for url, i in label2id.items()}
df["label"] = df["response"].map(label2id)

# ========== STEP 3: Train/Test Split ==========
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ========== STEP 4: Convert to Hugging Face Dataset ==========
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ========== STEP 5: Tokenize ==========
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    return tokenizer(
        example["prompt"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

# Rename 'label' to 'labels' as expected by Trainer
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

# ========== STEP 6: Model Setup ==========
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ========== STEP 7: Training Config ==========
training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',   # Directory for storing logs
    logging_steps=10,
    dataloader_num_workers=8,
    report_to=[],
)

# ========== STEP 8: Trainer ==========
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ========== STEP 9: Train, Evaluate, Save ==========
    trainer.train()
    trainer.evaluate()
    model.save_pretrained("./fine-tuned-model-uofm-cs-bert")
    tokenizer.save_pretrained("./fine-tuned-model-uofm-cs-bert")

    # Save mapping for later use
    import json
    with open("url_label_map.json", "w") as f:
        json.dump(id2label, f)

    print("Model trained and saved to ./fine-tuned-model-uofm-cs-bert")
