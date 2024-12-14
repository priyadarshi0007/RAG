from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

# Ensure API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment variables or .env file.")

# Embedding model
client = OpenAI()

response = client.embeddings.create(
  input="the test string",
  model="text-embedding-3-small"
)

print(response.data[0].embedding)