import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file.")
    exit(1)

# API endpoint
API_URL = f"{SUPABASE_URL}/rest/v1/credit_clients"

# Headers for REST request
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

def upload_data():
    print("🚀 Starting data upload to Supabase...")
    
    # Check if file exists
    if not os.path.exists("UCI_Credit_Card.csv"):
        print("❌ Error: UCI_Credit_Card.csv not found.")
        return

    # Read CSV
    try:
        df = pd.read_csv("UCI_Credit_Card.csv")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Rename columns to match database schema (as defined in setup_supabase.sql)
    # The SQL uses "default_payment_next_month" instead of "default.payment.next.month"
    df = df.rename(columns={"default.payment.next.month": "default_payment_next_month"})
    
    # Convert to list of dicts
    records = df.to_dict(orient="records")
    total_records = len(records)
    print(f"📊 Found {total_records} records to upload.")

    # Upload in batches to avoid payload limits
    BATCH_SIZE = 1000
    
    for i in range(0, total_records, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        
        try:
            response = requests.post(API_URL, json=batch, headers=headers)
            
            if response.status_code == 201:
                print(f"✅ Uploaded batch {i//BATCH_SIZE + 1} ({len(batch)} records)")
            else:
                print(f"⚠️ Failed to upload batch {i//BATCH_SIZE + 1}: {response.status_code} - {response.text}")
                # Optional: break or continue based on preference
        except Exception as e:
             print(f"❌ Exception uploading batch {i//BATCH_SIZE + 1}: {e}")

    print("🎉 Upload process completed!")

if __name__ == "__main__":
    upload_data()
