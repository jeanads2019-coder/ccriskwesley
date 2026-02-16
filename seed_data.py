from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Use DIRECT_URL for stable connections (no transaction pooling issues)
DATABASE_URL = os.getenv("DIRECT_URL")
if not DATABASE_URL:
    DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("Error: DIRECT_URL or DATABASE_URL not set in .env")
    exit(1)

print(f"Connecting to database... {DATABASE_URL.split('@')[1]}")

try:
    engine = create_engine(DATABASE_URL)
    
    # Check if UCI_Credit_Card.csv exists
    if not os.path.exists("UCI_Credit_Card.csv"):
        print("Error: UCI_Credit_Card.csv not found.")
        exit(1)
        
    print("Reading CSV...")
    df = pd.read_csv("UCI_Credit_Card.csv")
    
    # Rename column
    if "default.payment.next.month" in df.columns:
        df.rename(columns={"default.payment.next.month": "default_payment_next_month"}, inplace=True)
        
    print(f"Uploading {len(df)} records...")
    
    # Use chunksize to avoid timeouts
    df.to_sql("credit_clients", engine, if_exists="append", index=False, chunksize=500, method="multi")
    
    print("Upload complete!")
    
except Exception as e:
    print(f"Error during upload: {e}")
