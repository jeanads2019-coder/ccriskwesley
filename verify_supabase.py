import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DIRECT_URL")
if not DATABASE_URL:
    DATABASE_URL = os.getenv("DATABASE_URL")

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        result = connection.execute(text("SELECT count(*) FROM credit_clients"))
        count = result.scalar()
        print(f"✅ Verification successful: Found {count} records in 'credit_clients' table.")
except Exception as e:
    print(f"❌ Verification failed: {e}")
