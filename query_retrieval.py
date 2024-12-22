from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
import google.generativeai as palm

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Verify API key is loaded
if not api_key:
    raise ValueError("API key not found. Add GOOGLE_API_KEY to your .env file.")

# Configure Google Palm
palm.configure(api_key=api_key)

# Define a minimal wrapper for Google Palm
class GooglePalmLLM:
    def __call__(self, prompt):
        response = palm.generate_text(prompt=prompt)
        return response.result  # Return generated text

# Initialize components
def initialize_system():
    # Embedding model
    embeddings = OpenAIEmbeddings()

    # Vector store (ChromaDB)
    vectorstore = Chroma(
        persist_directory="./chromadb_data",
        embedding_function=embeddings.embed_query
    )

    # Google Palm LLM
    llm = GooglePalmLLM()

    # Retrieval-based QA chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

# Main execution
if __name__ == "__main__":
    # Initialize the system
    chain = initialize_system()

    # Example query
    query = "What is the main topic of the document?"
    response = chain.run(query)
    print("Response:", response)