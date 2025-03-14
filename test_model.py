from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('./fine-tuned-model')
tokenizer = BertTokenizer.from_pretrained('./fine-tuned-model')

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Test the NLU system with new inputs
user_input = "I want to major in Law"
result = classifier(user_input)
print(result)
print()

