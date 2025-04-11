import time
import sys
from pymongo import MongoClient
import ollama
import certifi  # Secure SSL connection

# MongoDB Atlas Connection Details
MONGO_URI = "mongodb+srv://mdfaruk88:farukeati88@cluster0.jewbapv.mongodb.net/?retryWrites=true&w=majority"
# MONGO_URI = "mongodb://mdfaruk88:farukeati88@cluster0.jewbapv.mongodb.net/?retryWrites=true&w=majority"
DATABASE_NAME = "farukdb"
COLLECTION_NAME = "prompt"

# Ollama Model Details
MODEL_NAME = "llama3.1"
MAX_TOKENS_INPUT = 4000
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

def truncate_text(text, max_tokens=MAX_TOKENS_INPUT):
    """Truncate text to fit within the model's max token limit."""
    char_limit = max_tokens * 4
    return text[:char_limit]

def clean_response(response_text):
    """Clean and format the response text."""
    if not response_text or "If you provide" in response_text or "I can try to help" in response_text:
        return "INVALID_RESPONSE"

    lines = response_text.split("\n")
    cleaned_queries = [line.strip() for line in lines if line.strip()]
    cleaned_queries.pop(0)
    # return "; ".join(cleaned_queries) if cleaned_queries else "INVALID_RESPONSE"
    
    return cleaned_queries if cleaned_queries else "INVALID_RESPONSE"

def generate_prompts_with_ollama(text, retries=MAX_RETRIES):
    """Generate prompts using Ollama."""
    truncated_text = truncate_text(text)

    prompt = f"""
Extract 10 highly relevant search queries from the text below.
Each query should be between 3-8 words and related to the topic.
Each query should be separated by a semicolon (;).
Do not return explanations, only the queries.

If the input text is too short, generate 10 search queries based on the main topic.

Text: {truncated_text}
Output:
"""

    for attempt in range(retries):
        try:
            print(f"Generating prompt with Ollama, attempt {attempt + 1}/{retries}")

            response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            generated_text = response["message"]["content"].strip()
            # generated_text = generated_text_all.split(";")
            

            formatted_response = clean_response(generated_text)
            # print("formatted_response text:")
            # print(generated_text)
            # print("\n")
            if formatted_response != "INVALID_RESPONSE":
                return formatted_response

        except Exception as e:
            print(f"❌ Error during prompt generation: {str(e)}")
            time.sleep(RETRY_DELAY)

    return "PROCESSING_FAILED"

def update_document_with_prompt(collection, document):
    """Generate a prompt using Ollama and update the existing document in MongoDB."""
    text = document.get("Text", "")

    if not text or len(text) < 15:
        print(f"Skipping document (ID: {document['_id']}) - insufficient text")
        return

    start_time = time.time()
    prompts = generate_prompts_with_ollama(text)
    processing_time = round(time.time() - start_time, 2)

    if prompts in ["PROCESSING_FAILED", "INVALID_RESPONSE"] or not prompts:
        print(f" Ollama failed to generate valid prompts for document ID: {document['_id']}. Skipping update.")
        return

    # Update the existing document in MongoDB
    update_query = {"_id": document["_id"]}
    update_values = {
        "$set": {
            "Prompts": prompts,
            "Processing Time (s)": processing_time,
            "Timestamp": time.time()
        }
    }

    try:
        result = collection.update_one(update_query, update_values)

        if result.modified_count > 0:
            print(f"\n Document ID {document['_id']} updated with prompts (Processing time: {processing_time}s)")
            print(f"Updated Prompts: {prompts}")
            print("-" * 40)
        else:
            print(f" No changes were made to document ID {document['_id']} (It may already contain prompts).")

    except Exception as e:
        print(f" MongoDB update failed for document ID {document['_id']}: {e}")

def process_documents(collection):
    """Process documents and update them with generated prompts in MongoDB Atlas."""
    try:
        cursor = collection.find({
            "$or": [
                {"Prompts": {"$exists": False}},  # Missing "Prompts"
                {"Prompts": ""},                 # Empty "Prompts"
                {"Prompts": None}                # Null "Prompts"
            ]
        }).limit(2000)

        count = 0
        for document in cursor:
            print(f"Processing document {count + 1}: ID {document['_id']} (Label: {document.get('Label', 'N/A')})")
            update_document_with_prompt(collection, document)
            count += 1
            time.sleep(1)  # Small delay to avoid flooding

        print(f"Completed processing {count} documents!")

    except Exception as e:
        print(f"Critical error in process_documents: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())  # Secure SSL Connection
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())  # Secure SSL Connection
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        print("Starting document processing using MongoDB Atlas...")
        process_documents(collection)
        print("\n Processing completed!")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        client.close()
