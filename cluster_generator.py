from pymongo import MongoClient
import ollama
import pprint
import time

# MongoDB connection details
MONGO_URI = "mongodb+srv://jeremy-flagg:eWK4Bo7sR8bhDMn5@merjekcluster1.mwxms.mongodb.net/?retryWrites=true&w=majority&appName=MerjekCluster1"
DATABASE_NAME = "merjekaidb"
COLLECTION_NAME = "crawled_cs_pages2"
OLLAMA_MODEL_NAME = "llama3.1:8b"
# OLLAMA_MODEL_NAME = "solar:10.7b"

def generate_prompts_with_ollama(text):
    try:
        # Construct the prompt for Ollama API
        prompt = f"""
        You are extracting **exactly 30 highly relevant search queries** from the given text. **Do not summarize, analyze, or paraphrase.**  

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

        if not response:
            raise ValueError("No response received from Ollama API.")

        # Ensure response is formatted correctly
        formatted_response = response.replace("\n", " ").strip()
        
        # Validate format: Must contain semicolon-separated queries
        if ";" not in formatted_response:
            raise ValueError(f"Unexpected response format from LLM: {formatted_response}")

        return formatted_response

    except Exception as e:
        print(f"Error generating prompts: {e}")
        return ""

def connect_to_mongodb():
    try:
        # Connect to MongoDB
        db_client = MongoClient(MONGO_URI)
        print("Connected to MongoDB.")
        return db_client[DATABASE_NAME]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def process_documents(collection, start_label=1):
    for document in collection.find({"Label": {"$gte": start_label}}):
        text = document.get("Text", "")
        if not text or len(text) < 15:
            print(f"Skipping document with _id: {document['_id']} due to missing or short 'Text' field.")
            continue
        
        print(f"\nProcessing document with _id: {document['_id']}...")
        start_time = time.time()
        prompts = generate_prompts_with_ollama(text)
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        # Pretty-print the result
        print("\nGenerated LLama Prompts:")
        pprint.pprint({
            "_id": document["_id"], 
            "Label": document["Label"], 
            "Prompts": prompts, 
            "Processing Time (s)": processing_time
        })
        print("-" * 80)

if __name__ == "__main__":
    db = connect_to_mongodb()
    if db is not None:
        collection = db[COLLECTION_NAME]
        process_documents(collection)
    else:
        print("Unable to connect to database. Exiting.")
