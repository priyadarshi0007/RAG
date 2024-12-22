from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vectorstore(persist_directory="./chromadb_data"):
    """
    Initialize ChromaDB vector store.
    """
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def add_chunks_to_vectorstore(vectorstore, chunks):
    """
    Add text chunks to the ChromaDB vector store.
    """
    for idx, chunk in enumerate(chunks):
        vectorstore.add_texts(
            texts=[chunk],
            ids=[f"chunk_{idx}"]
        )