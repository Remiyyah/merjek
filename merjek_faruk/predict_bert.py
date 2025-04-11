import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Load model and tokenizer
model_dir = "./fine-tuned-model-uofm-cs-bert"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.eval()

# Load ID-to-URL map
with open("url_label_map.json") as f:
    id2label = json.load(f)

def predict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        return id2label[str(predicted_class_id)]

# Example usage
if __name__ == "__main__":
    # Example
    prompts = [
        "How can I study English at IEI?",
        "What are the requirements to study at IEI?",
        "How do I apply to IEI?",
        "How much does it cost to study at IEI?",
        "Program schedule"
    ]

    for p in prompts:
        print(f"Prompt: {p}")
        print(f"Response: {predict(p)}")
