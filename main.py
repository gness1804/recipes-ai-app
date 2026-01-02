import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("Set PINECONE_API_KEY")

pc = Pinecone(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pc.Index(os.getenv("PINECONE_INDEX", "recipes"))
namespace = "recipes"

print(f"Connected to Pinecone index: {index}")