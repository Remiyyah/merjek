from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from pymongo import MongoClient

# === 1. Load fine-tuned DistilBERT model and tokenizer ===
model_path = "./fine-tuned-model-iei5"  
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# === 2. Define text classification pipeline ===
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)

# === 3. Load MongoDB ===
username = "jeremyflagg12"
password = "QGTrn5lbWa2qrXFL"
conn_str = f"mongodb+srv://{username}:{password}@cluster0.t4orq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(conn_str)

db = client["Prompts"]
collection = db["search2"]
documents = list(collection.find())

# === 4. Get user input ===
user_input = "Fees"

# === 5. Predict using model ===
result = classifier(user_input)
predicted_label = result[0]['label']
confidence = result[0]['score']

# === 6. Try to find matching prompt in MongoDB ===
matched_doc = None
for doc in documents:
    if "Prompt" in doc and user_input in doc["Prompt"]:
        matched_doc = doc
        break

# === 7. Print results ===
print(f"\nUser input: {user_input}")
print(f"Predicted label: {predicted_label}")
print(f"Confidence: {confidence:.4f}")

if matched_doc:
    print(f"True label: {matched_doc.get('URL_Label', 'N/A')}")
    print(f"Associated URL: {matched_doc.get('Response', 'N/A')}")
else:
    print("No matching prompt found in MongoDB.")
