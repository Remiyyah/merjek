from pymongo import MongoClient
import mysql.connector
import ollama
import pprint
import time

# MongoDB connection details
MONGO_URI = "mongodb+srv://jeremy-flagg:eWK4Bo7sR8bhDMn5@merjekcluster1.mwxms.mongodb.net/?retryWrites=true&w=majority&appName=MerjekCluster1"
DATABASE_NAME = "merjekaidb"
COLLECTION_NAME = "crawled_cs_pages2"
OLLAMA_MODEL_NAME = "llama3.1:8b"

# MySQL connection details
MYSQL_HOST = "127.0.0.1"
MYSQL_USER = "merjek"
MYSQL_PASSWORD = "your_password"  # Replace with actual password
MYSQL_DATABASE = "merjekdb"

def generate_prompts_with_ollama(text):
    """Generate search queries using the Ollama API."""
    try:
        prompt = f"""
      You are extracting **exactly 100 highly relevant unique search queries** from the given text. **Do not summarize, analyze, or paraphrase.**  

### **Instructions:**  
- Each unique query must be **1-8 words long** and mention **"University of Memphis."**  
- Queries should be **keywords, short questions, or phrases** (not full sentences).  
- **Strictly separate each query with a semicolon (;)** on a single line.  
- **Do not add newlines, explanations, numbers, extra formatting, or introductory text** (e.g., "Here is the output format you requested").  

### **Text to Process:**  
{text}  

### **Example Output:**  
University of Memphis computer science; University of Memphis AI research; University of Memphis admission requirements; University of Memphis data science program  

### **Your Output:**  
(Only the queries, separated by semicolons, in one single line. No extra text, no explanations.)
"""
        output = ollama.generate(model=OLLAMA_MODEL_NAME, prompt=prompt)
        response = output.get('response', "").strip()

        if not response:
            raise ValueError("No response received from Ollama API.")

        formatted_response = response.replace("\n", " ").strip()
        
        if ";" not in formatted_response:
            raise ValueError(f"Unexpected response format from LLM: {formatted_response}")

        return formatted_response

    except Exception as e:
        print(f"Error generating prompts: {e}")
        return ""

def connect_to_mongodb():
    """Connect to MongoDB."""
    try:
        db_client = MongoClient(MONGO_URI)
        print("Connected to MongoDB.")
        return db_client[DATABASE_NAME]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def connect_to_mysql():
    """Connect to MySQL."""
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        print("Connected to MySQL.")
        return connection
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def get_last_processed_label(mysql_connection):
    """Fetch the last processed Label from MySQL to resume processing."""
    try:
        cursor = mysql_connection.cursor()
        cursor.execute("SELECT MAX(Label) FROM generated_prompts")  
        result = cursor.fetchone()
        last_label = result[0] if result[0] is not None else 0
        print(f"Last processed Label in MySQL: {last_label}")
        return last_label + 1  # Start from the next label
    except Exception as e:
        print(f"Error fetching last processed Label: {e}")
        return 1  # Default to start from Label 1 if MySQL query fails

def insert_into_mysql(connection, _id, label, url, prompts, processing_time):
    """Insert processed data into MySQL."""
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO generated_prompts (_id, Label, Url, Prompts, Processing_Time)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (str(_id), label, url, prompts, processing_time))  # Convert _id to string
        connection.commit()
        print(f"Inserted document {_id} into MySQL.")
    except Exception as e:
        print(f"Error inserting into MySQL: {e}")

def process_documents(collection, mysql_connection, batch_size=100):
    """Process documents from MongoDB and store results in MySQL."""
    start_label = get_last_processed_label(mysql_connection)
    processed_count = 0

    while True:
        query = {"Label": {"$gte": start_label}}
        
        # Fetch a batch of documents in Label order
        documents = list(collection.find(query).sort("Label", 1).limit(batch_size))

        if not documents:
            print("No more documents to process.")
            break

        for document in documents:
            text = document.get("Text", "")
            if not text or len(text) < 15:
                print(f"Skipping document with _id: {document['_id']} due to missing or short 'Text' field.")
                continue
            
            print(f"\nProcessing document with _id: {document['_id']} (Label {document['Label']})...")
            start_time = time.time()

            try:
                prompts = generate_prompts_with_ollama(text)
                end_time = time.time()
                processing_time = round(end_time - start_time, 2)

                # Insert into MySQL
                insert_into_mysql(mysql_connection, document["_id"], document["Label"], document["Url"], prompts, processing_time)
                processed_count += 1

                # Pretty-print the result
                print("\nGenerated LLama Prompts:")
                pprint.pprint({
                    "_id": str(document["_id"]),
                    "Label": document["Label"], 
                    "Url": document["Url"],
                    "Prompts": prompts, 
                    "Processing Time (s)": processing_time
                })
                print("-" * 80)

                # Update start_label to fetch the next batch
                start_label = document["Label"] + 1

            except Exception as e:
                print(f"Error processing document {document['_id']}: {e}")  # Handle failures gracefully

    if processed_count > 0:
        print("Prompts inserted in DB.")

if __name__ == "__main__":
    db = connect_to_mongodb()
    mysql_conn = connect_to_mysql()
    
    if db is not None and mysql_conn is not None:
        collection = db[COLLECTION_NAME]
        process_documents(collection, mysql_conn)
        mysql_conn.close()
    else:
        print("Unable to connect to database. Exiting.")
