from dotenv import load_dotenv
import os
from preprocess_vtt import parse_vtt, chunk_text
from vectorstore import create_vectorstore, add_chunks_to_vectorstore
from retrieval_system import initialize_retrieval_system

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
if not openai_api_key:
    raise ValueError("API key not found. Add OPENAI_API_KEY to your .env file.")

def main():
    # Path to your .vtt file
    vtt_file_path = "finetune.vtt"

    # Step 1: Parse the .vtt file
    text = parse_vtt(vtt_file_path)

    # Step 2: Chunk the text
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    # Step 3: Create or connect to the vector store
    vectorstore = create_vectorstore()

    # Step 4: Add chunks to the vector store
    add_chunks_to_vectorstore(vectorstore, chunks)

    # Step 5: Initialize the retrieval-based QA system
    chain = initialize_retrieval_system(vectorstore, openai_api_key)

# Interactive query loop
    print("\nReady to answer your questions!")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        response = chain.invoke({"query": query})
        print("\nResponse:", response)

if __name__ == "__main__":
    main()