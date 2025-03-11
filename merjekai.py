import os
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from pymongo import MongoClient

def main():
    print("🚀 Starting merjekai.py...")

    # ✅ **MongoDB Atlas Connection**
    try:
        conn_str = "mongodb+srv://jeremyflagg12:QGTrn5lbWa2qrXFL@cluster0.t4orq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(conn_str)
        db = client["Prompts"]
        collection = db["merjekai3"]
        print("✅ Connected to MongoDB Atlas successfully.")
    except Exception as e:
        print(f"❌ MongoDB Atlas Connection Error: {e}")
        exit()

    # ✅ **Fetch First 10,000 Documents from MongoDB**
    documents = list(collection.find(
        {"Prompts": {"$ne": [], "$ne": "", "$ne": "PROCESSING_FAILED", "$ne": "INSUFFICIENT_TEXT", "$ne": "PROCESSING_ERROR"}},
        {"Label": 1, "Prompts": 1, "URL": 1}
    ).limit(10000))

    if not documents:
        print("⚠️ No documents found in the collection.")
        exit()

    # ✅ **Convert MongoDB Data to Pandas DataFrame**
    data = []
    for doc in documents:
        if "Prompts" in doc and isinstance(doc["Prompts"], list):  # Handle array format
            for prompt in doc["Prompts"]:
                prompt = prompt.strip()
                if prompt:  # Only add non-empty prompts
                    data.append({
                        "Label": str(doc.get("Label", "Unknown")),
                        "Prompts": prompt,
                        "URL": doc.get("URL", "")
                    })

    df = pd.DataFrame(data)

    if df.empty:
        print("❌ DataFrame is empty, no valid data to train the model.")
        exit()

    print(f"✅ Loaded {len(df)} valid prompts from the first 10,000 documents.")

    # ✅ **Train-Test Split (80/20)**
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # ✅ **Convert DataFrames to Hugging Face Datasets**
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # ✅ **Create Label Mapping**
    intent_labels = df["Label"].unique()
    label2id = {label: idx for idx, label in enumerate(intent_labels)}
    id2label = {idx: label for idx, label in enumerate(intent_labels)}

    # ✅ **Map Labels to IDs**
    def map_labels(example):
        example["label"] = label2id.get(example["Label"], 0)
        return example

    train_dataset = train_dataset.map(map_labels)
    val_dataset = val_dataset.map(map_labels)

    # ✅ **Load Tokenizer**
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # ✅ **Tokenization Function**
    def tokenize_function(examples):
        texts = examples["Prompts"] if isinstance(examples["Prompts"], list) else [examples["Prompts"]]
        tokenized = tokenizer(
            texts,
            padding="max_length", 
            max_length=128,  
            truncation=True,
            return_tensors="pt"
        )
        return tokenized

    # Apply tokenization function
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # ✅ **Data Collator**
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ✅ **Load Pretrained BERT Model**
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(intent_labels),
        id2label=id2label,
        label2id=label2id
    )

    # ✅ **Detect Environment**
    is_cluster = os.getenv("SLURM_JOB_ID") is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 if is_cluster else 8

    print(f"🖥️ Using device: {device}, Batch size: {batch_size}")

    model.to(device)

    # ✅ **Training Arguments**
    training_args = TrainingArguments(
        output_dir="./results-merjekai3",
        evaluation_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        dataloader_num_workers=8 if is_cluster else 4,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    # ✅ **Initialize Trainer**
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ✅ **Train the Model**
    print("Starting training...")
    trainer.train()

    # ✅ **Evaluate the Model**
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # ✅ **Save Model & Tokenizer**
    output_dir = "./fine-tuned-model-merjekai3"
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Model and tokenizer saved successfully.")

if __name__ == "__main__":
    main()
