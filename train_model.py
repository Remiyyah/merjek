from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from pymongo import MongoClient



try:
    username = "xxxxx"
    password = "xxxxx"
    conn_str = f"mongodb+srv://{username}:{password}@merjekcluster1.mwxms.mongodb.net/?retryWrites=true&w=majority&appName=MerjekCluster1"
    # Connect to the MongoDB server
    client = MongoClient(conn_str)  # Replace with your MongoDB connection string
except:
    print("Please update the script with your MongoDB connection string")

# Access the database and collection
db = client["merjekaidb"]
collection = db["uofm_pages_crawled"]

# Retrieve data and convert to a list
documents = list(collection.find())

# Prepare the data for the DataFrame
data = []
for doc in documents:
    label = doc["Label"]
    url = doc["Url"]
    prompts = doc["Prompts"]
    for prompt in prompts:
        data.append({"Label": str(label), "Url": url, "Prompt": prompt})

# Create the DataFrame
df = pd.DataFrame(data)

# Drop '_id' becuase MongoDB ObejectId() type cant be understood by Hugging face Dataset
if '_id' in df.columns:
    df = df.drop(columns=['_id'])

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2)

# Convert dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Ensure the intent column is correctly mapped to numeric labels
intent_labels = df['Label'].unique()
label2id = {label: idx for idx, label in enumerate(intent_labels)}
id2label = {idx: label for idx, label in enumerate(intent_labels)}

train_dataset = train_dataset.map(lambda x: {'label': label2id[x['Label']]})
val_dataset = val_dataset.map(lambda x: {'label': label2id[x['Label']]})

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['Prompt'],
                     padding="max_length",
                     max_length=512,
                     truncation=True,
                     return_tensors="pt")


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
#train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Data collator to handle dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the model and configure for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', id2label=id2label, label2id=label2id, num_labels=len(intent_labels))

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the GPU if available
model = model.to(device)

#epoch
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
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

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
model.save_pretrained("./fine-tuned-model-uofm-cs")
tokenizer.save_pretrained("./fine-tuned-model-uofm-cs")
