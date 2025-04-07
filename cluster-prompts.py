import os
import ollama
import pprint
import time
from pymongo import MongoClient

# MongoDB Atlas connection details
MONGO_URI = "mongodb+srv://jeremyflagg12:QGTrn5lbWa2qrXFL@cluster0.t4orq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "Prompts"
COLLECTION_NAME = "merjekai4"
OLLAMA_HOST = "http://127.0.0.1:11434"  # Use localhost for Ollama
OLLAMA_MODEL_NAME = "llama3.2:3b"

# Set the OLLAMA_HOST environment variable before calling Ollama
os.environ["OLLAMA_HOST"] = OLLAMA_HOST

BATCH_SIZE = 100  # Process in small batches to avoid cursor timeouts

def generate_prompts_with_ollama(text):
    try:
             prompt = f"""
        You are extracting **up to 40 highly relevant search queries** from the given text. **Do not summarize, analyze, or paraphrase.**  

### **Instructions:**  
- Each query must be **1-8 words long** and relate to the **"University of Memphis."**  
- Queries should be **keywords, short questions, or phrases** (not full sentences).  
- **Strictly separate each query with a semicolon (;)** on a single line.  
- **Do not add newlines, explanations, numbers, or extra formatting.**  

### **Text to Process:**  
{text}  

### **Output Format (Example):**  
University of Memphis computer science; University of Memphis AI research; University of Memphis admission requirements; University of Memphis data science program  

### **Your Output:**  
(Ensure queries are in one single line, semicolon-separated, no newlines)  
"""
        output = ollama.generate(model=OLLAMA_MODEL_NAME, prompt=prompt)
        response = output.get('response', "").strip()
        return response.replace("\n", " ").strip() if response else ""
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return ""

def connect_to_mongodb():
    try:
        db_client = MongoClient(MONGO_URI)
        print("Connected to MongoDB Atlas.")
        return db_client[DATABASE_NAME]
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {e}")
        return None

def process_documents(collection):
    last_id = None  # Track last processed document
    while True:
        query = {"$or": [{"Prompts": {"$size": 0}}, {"Prompts": {"$exists": False}}]}
        if last_id:
            query["_id"] = {"$gt": last_id}  # Pagination using _id
        
        documents = list(collection.find(query).sort("_id").limit(BATCH_SIZE))
        if not documents:
            break  # No more documents left

        for document in documents:
            try:
                label = document.get("Label", "UNKNOWN")
                text = document.get("Text", "")
                if not text or len(text) < 15:
                    collection.update_one({"_id": document["_id"]}, {"$set": {"Prompts": ["INSUFFICIENT_TEXT"]}})
                    continue
                
                start_time = time.time()
                prompts = generate_prompts_with_ollama(text)
                processing_time = round(time.time() - start_time, 2)

                collection.update_one(
                    {"_id": document["_id"]},
                    {"$set": {"Processing Time (s)": processing_time, "Prompts": prompts.split("; ")}}
                )
                last_id = document["_id"]  # Update last processed ID
                time.sleep(2)  # Prevent rate limits
            except Exception as e:
                print(f"Error processing document {document.get('Label', 'UNKNOWN')}: {e}")
                collection.update_one({"_id": document["_id"]}, {"$set": {"Prompts": ["PROCESSING_ERROR"]}})
                time.sleep(5)

if __name__ == "__main__":
    db = connect_to_mongodb()
    if db:
        collection = db[COLLECTION_NAME]
        process_documents(collection)
    else:
        print("Unable to connect to database. Exiting.")
