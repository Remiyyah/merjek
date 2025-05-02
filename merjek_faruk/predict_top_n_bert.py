#predict_bert.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Load model and tokenizer
model_dir = "./1_fine-tuned-model-uofm-cs-bert"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.eval()

# Load ID-to-URL map
with open("url_label_map.json") as f:
    id2label = json.load(f)

def predict_top_k(prompt, k=10):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)

        # Prepare top k results as label -> probability pairs
        top_k_results = []
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            label = id2label[str(idx.item())]
            top_k_results.append((label, prob.item()))

        return top_k_results


# Example usage
if __name__ == "__main__":

    
    # Example
    prompts = [
        "How can I study English at IEI?",
        "Cost",
        "How do I apply to IEI?",
        "How much does it cost to study at IEI?",
        "program fees",
        "When do classes start?", 
        "Program schedule",
        "How much does your IEI costs?",
        "program cost",
        "I want to know about program cost.",
        "Give me details about program fees.",
        "program fees details.",
        "IEI program fees details.",
        "How much does it cost?",
        "International student program cost.",
        "domestic student cost.", 
        "Could I know about cost?"
    ]

    TOP = 5
    for p in prompts:
        print(f"Prompt: {p}")
        top_preds = predict_top_k(p, k=TOP)
        for i, (label, score) in enumerate(top_preds, 1):
            print(f"{i}. {label} - {score:.4f}")
        print("\n")
