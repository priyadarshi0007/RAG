from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
if not openai_api_key:
    raise ValueError("API key not found. Add OPENAI_API_KEY to your .env file.")

# Initialize components
def initialize_openai_system():
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Connect to ChromaDB
    vectorstore = Chroma(
        persist_directory="./chromadb_data",  # Persistent vector store directory
        embedding_function=embeddings
    )

    # Initialize OpenAI chat model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0  # For deterministic responses
    )

    # Set up the retrieval-based QA system
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

# Main execution
if __name__ == "__main__":
    # Initialize the OpenAI-based system
    chain = initialize_openai_system()

    # Example query
    query = "What is the main topic of the document?"
    response = chain.invoke({"query": query})
    print("Response:", response)