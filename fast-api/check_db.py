import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("digisinc-index")

# Get stats about the index
stats = index.describe_index_stats()

print("--- Pinecone Index Stats ---")
print(f"Total Vectors: {stats['total_vector_count']}")
print(f"Dimensions: {stats['dimension']}")
print(f"Namespaces: {stats['namespaces']}")
print("----------------------------")

if stats['total_vector_count'] > 0:
    print("✅ Confirmed: Your PDF chunks are safely stored.")
else:
    print("❌ Warning: The index appears to be empty.")