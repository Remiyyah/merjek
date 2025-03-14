from pymongo import MongoClient
import ollama
import pprint


model_path = "Llama-3.2-1B-Instruct"
collection_name = "uofm_cs_crawled"  # the name of the updated database.
ollama_model_name = "llama3.1"




# "University of Memphis upcoming computer science events;'UofM computer science faculty meeting schedule; Computer science retreats Shelby Farms"
# Then it should generate a list of those prompts
def generate_prompts_with_ollama(text):
    try:

        prompt = f"You are given the following text, generate 100 potential search queries in a list of " \
                 f"values separated by a semi colon with no listed numbers. These search queries " \
                 f"can be a keyword, a question or a statement that is highly relevant with " \
                 f"the content. In your response, don't make any comments. Just include up to 100 potential " \
                 f"search queries in a string separated by semicolon."

        output = ollama.generate(model=ollama_model_name, prompt=prompt)

        response = output['response']
        prompts = response.split(";")
        prompts = [item.strip() for item in prompts]

        return prompts

    except Exception as e:
        print(f"Error generating prompts for text: {text}. Error: {e}")
        return []



def connect_to_mongodb():
    try:
        username = "xxxxxxxx"
        password = "xxxxxxx"
        #conn_str = f"mongodb+srv://{username}:{password}@merjekcluster1.mwxms.mongodb.net/?retryWrites=true&w=majority&appName=MerjekCluster1"
        conn_str = f"mongodb+srv://{username}:{password}@merjekcluster1.mwxms.mongodb.net/?retryWrites=true&w=majority&appName=MerjekCluster1"
        # Connect to the MongoDB server
        db_client = MongoClient(conn_str)  # Replace with your MongoDB connection string
        pprint.pprint("Connected to the MongoDB database.")
        # Access the database and collection
        db = db_client["merjekaidb"]

        return db

    except:
        pprint.pprint("Please update the script with your MongoDB connection string")
    


db = connect_to_mongodb()
collection = db[collection_name]


# Specify the starting _id
start_label = 1

# Process each document in the collection
for document in collection.find({"Label": {"$gte": start_label}}):
    text = document.get("Text", "")
    if not text and len(text)<15:
        pprint.pprint(f"Skipping document with _id: {document['_id']} due to missing 'Text' field.")
        continue
    
    #pprint.pprint("-------------------------------------------------------------")
    pprint.pprint(f"Generating prompts for document with _id: {document['_id']}")
    pprint.pprint("-------------------------------------------------------------")
    print()
    #prompts = generate_prompts_with_openai(text)
    prompts = generate_prompts_with_ollama(text)
    
    # Ensure prompt format matches your desired output
    formatted_prompts = f"['{' ; '.join(prompts)}']"

    # Update the document with the generated prompts
    collection.update_one(
        {"_id": document["_id"]},
        {"$set": {"Prompts:":prompts}}
    )
    pprint.pprint(
        f"Updated document with _id: {document['_id']} with Label {document['Label']} and Prompts {formatted_prompts}."
    )


pprint.pprint("Processing completed.")