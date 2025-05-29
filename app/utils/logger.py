import logging
from datetime import datetime
import pymongo

# Log local
logging.basicConfig(filename="training.log", level=logging.INFO)

def log_event_to_file(msg):
    logging.info(f"{datetime.now()} - {msg}")

# Log a MongoDB
def log_event_to_mongodb(msg, db_uri="mongodb://localhost:27017/", db_name="mlops_logs"):
    client = pymongo.MongoClient(db_uri)
    db = client[db_name]
    collection = db["events"]
    collection.insert_one({"timestamp": datetime.now(), "message": msg})

