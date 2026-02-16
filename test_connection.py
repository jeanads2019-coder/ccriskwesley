from sqlalchemy import create_engine
import os

# Hardcoded (masked password for safety in artifacts, but needed for test)
# URL: "postgresql://postgres.ksybaulafbxvcqyyggny:ccrisk2026wesley@aws-1-sa-east-1.pooler.supabase.com:5432/postgres"
# Assuming default port 5432 for Direct connection to avoid pgbouncer issues with prepared statements sometimes.

connection_string = "postgresql://postgres.ksybaulafbxvcqyyggny:ccrisk2026wesley@aws-1-sa-east-1.pooler.supabase.com:5432/postgres"

print(f"Testing connection to: {connection_string.split('@')[1]}")

try:
    engine = create_engine(connection_string)
    with engine.connect() as connection:
        print("✅ SUCCESS: Connected!")
        result = connection.execute("SELECT version()")
        print(f"Version: {result.scalar()}")
except Exception as e:
    print(f"❌ FAIL: {e}")
