import requests
import pprint
import time
from pymongo import MongoClient
from requests.exceptions import RequestException
import sys

# Local MongoDB connection details
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "merjekaidb"
COLLECTION_NAME = "um_crawled"

# LM Studio API details
LM_STUDIO_URL = "http://localhost:1234/v1/completions"
MODEL_NAME = "mistral-7b-instruct-v0.3"
MAX_TOKENS_INPUT = 4000
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

def truncate_text(text, max_tokens=MAX_TOKENS_INPUT):
    """Truncate text to fit within the model's max token limit."""
    char_limit = max_tokens * 4
    return text[:char_limit]

def clean_response(response_text):
    """Clean and format the response text."""
    if not response_text:
        return ""
    lines = response_text.split("\n")
    cleaned_queries = []
    for line in lines:
        line = line.strip()
        if line and not line.lower().startswith("output:"):
            line = line.split(". ", 1)[-1]
            cleaned_queries.append(line)
    return "; ".join(cleaned_queries)

def generate_prompts_with_lmstudio(text, retries=MAX_RETRIES):
    """Generate prompts for the entire document."""
    # Truncate text if it's too long
    truncated_text = truncate_text(text)
    
    prompt = f"""
Extract up to 100 highly relevant search queries from the given text.
Each query must be 1-8 words long and related to "University of Memphis."
Strictly separate queries with a semicolon (;). Do not add newlines or explanations.

Text: {truncated_text}
Output:
"""
    
    for attempt in range(retries):
        try:
            print(f"Processing document, attempt {attempt + 1}/{retries}")
            
            response = requests.post(
                LM_STUDIO_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0.7
                },
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                time.sleep(RETRY_DELAY)
                continue
            
            response_json = response.json()
            generated_text = response_json.get("choices", [{}])[0].get("text", "").strip()
            
            if not generated_text:
                print("Empty response received")
                time.sleep(RETRY_DELAY)
                continue
            
            formatted_response = clean_response(generated_text)
            if formatted_response:
                return formatted_response
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            time.sleep(RETRY_DELAY)
    
    return "PROCESSING_FAILED"


def process_documents(collection):
    """Process documents from MongoDB collection."""
    try:
        document = collection.find_one({"Prompts": ""})
        if not document:
            print("✅ All documents are already processed.")
            return

        start_label = document["Label"]
        
        for document in collection.find({"Label": {"$gte": start_label}, "Prompts": ""}).sort("Label", 1):
            try:
                print(f"\n{'='*80}")
                print(f"Processing document Label {document['Label']}...")
                
                text = document.get("Text", "")
                if not text or len(text) < 15:
                    print(f"Skipping document (Label {document['Label']}) - insufficient text")
                    collection.update_one(
                        {"_id": document["_id"]},
                        {"$set": {"Prompts": "INSUFFICIENT_TEXT"}}
                    )
                    continue

                start_time = time.time()
                prompts = generate_prompts_with_lmstudio(text)
                processing_time = round(time.time() - start_time, 2)

                # Update MongoDB
                collection.update_one(
                    {"_id": document["_id"]},
                    {"$set": {
                        "Prompts": prompts,
                        "Processing Time (s)": processing_time
                    }}
                )
                
                print(f"\n✅ Document Label {document['Label']} processed in {processing_time}s")
                print(f"Generated prompts length: {len(prompts)}")
                print("\nGenerated Prompts:")
                print("-" * 40)
                
                # Display prompts in original format
                if prompts != "PROCESSING_FAILED" and prompts != "INSUFFICIENT_TEXT":
                    print(prompts)
                else:
                    print(f"Status: {prompts}")
                
                print("-" * 40)
                
                # Add a small delay between documents
                time.sleep(2)

            except Exception as e:
                print(f"Error processing document {document['Label']}: {str(e)}")
                collection.update_one(
                    {"_id": document["_id"]},
                    {"$set": {"Prompts": "PROCESSING_ERROR"}}
                )
                time.sleep(RETRY_DELAY)

    except Exception as e:
        print(f"Critical error in process_documents: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        print("Starting document processing...")
        process_documents(collection)
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        client.close()