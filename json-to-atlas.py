

import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
import json

def import_json_to_mongodb():
    print("Starting JSON import process...")

    try:
        # Load JSON file
        print("Reading JSON file...")
        with open("tenk-prompts.json", 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.read_json("tenk-prompts.json")
            except json.JSONDecodeError:
                df = pd.read_json("tenk-prompts.json")

        print(f"Successfully loaded JSON file with {len(df)} records")

        # Convert DataFrame to list of dictionaries
        data = df.to_dict(orient="records")

        # Process each document before insertion
        print("Processing documents...")
        for item in data:
            # Convert _id field
            try:
                if "_id" in item:
                    if isinstance(item["_id"], dict) and "$oid" in item["_id"]:
                        item["_id"] = ObjectId(item["_id"]["$oid"])
                    elif isinstance(item["_id"], str):
                        try:
                            item["_id"] = ObjectId(item["_id"])
                        except:
                            del item["_id"]
                    else:
                        del item["_id"]
            except Exception as e:
                print(f"Warning: Error processing _id field: {e}")
                if "_id" in item:
                    del item["_id"]

            # Convert Prompts field to an array
            if "Prompts" in item and isinstance(item["Prompts"], str):
                item["Prompts"] = item["Prompts"].split(";")  # Split by semicolon

        # Connect to MongoDB Atlas
        print("Connecting to MongoDB Atlas...")
        client = MongoClient("mongodb+srv://jeremyflagg12:QGTrn5lbWa2qrXFL@cluster0.t4orq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        
        # Verify connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas")

        db = client["Prompts"]
        collection = db["merjekai3"]

        # Insert data in batches
        if data:
            batch_size = 1000
            total_documents = len(data)
            
            for i in range(0, total_documents, batch_size):
                batch = data[i:i + batch_size]
                collection.insert_many(batch)
                print(f"Imported {min(i + batch_size, total_documents)}/{total_documents} documents")

            print(f"Successfully imported {total_documents} documents to MongoDB Atlas!")
        else:
            print("No data to import!")

    except FileNotFoundError:
        print("Error: JSON file 'tenk-prompts.json' not found!")
    except pd.errors.EmptyDataError:
        print("Error: JSON file is empty!")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed")

if __name__ == "__main__":
    import_json_to_mongodb()
