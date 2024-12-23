from preprocess_vtt import parse_vtt, chunk_text
from vectorstore import create_vectorstore, add_chunks_to_vectorstore
from retrieval_system import initialize_retrieval_system, query_retrieval_system
from dotenv import load_dotenv
import os
import glob

# Load environment variables.
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded.
if not openai_api_key:
    raise ValueError("API key not found. Add OPENAI_API_KEY to your .env file.")

def main():
    # Process all .vtt files in the directory.
    vtt_files = glob.glob("vtt_files/*.vtt")
    # Print the list of .vtt files.
    if not vtt_files:
        print("No .vtt files found in the 'vtt_files' directory.")
        return
    print("Found the following .vtt files:")
    for vtt_file in vtt_files:
        print(f"- {vtt_file}")
    # Create or connect to the vector store.
    vectorstore = create_vectorstore()

    for vtt_file in vtt_files:
        # Parse the .vtt file.
        text = parse_vtt(vtt_file)

        # Chunk the text.
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        # Add chunks to the vector store with metadata.
        file_name = os.path.basename(vtt_file)
        add_chunks_to_vectorstore(vectorstore, chunks, file_name)

    # Initialize the retrieval-based QA system.
    chain = initialize_retrieval_system(vectorstore, openai_api_key)

    # Interactive query loop.
    print("\nReady to answer your questions!")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            break
        query_retrieval_system(chain, query)

if __name__ == "__main__":
    main()