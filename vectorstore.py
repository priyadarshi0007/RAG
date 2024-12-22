from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vectorstore(persist_directory="./chromadb_data"):
    """
    Initialize ChromaDB vector store.
    """
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# def add_chunks_to_vectorstore(vectorstore, chunks):
#     """
#     Add text chunks to the ChromaDB vector store if they are not already present.
#     """
#     # Get existing IDs in the vector store
#     existing_ids = set(vectorstore.get()["ids"])

#     for idx, chunk in enumerate(chunks):
#         chunk_id = f"chunk_{idx}"
#         if chunk_id not in existing_ids:
#             vectorstore.add_texts(
#                 texts=[chunk],
#                 ids=[chunk_id]
#             )

# def add_chunks_to_vectorstore(vectorstore, chunks, file_name):
#     """
#     Add text chunks to the ChromaDB vector store with metadata.
#     """
#     existing_ids = set(vectorstore.get()["ids"])

#     for idx, chunk in enumerate(chunks):
#         chunk_id = f"{file_name}_chunk_{idx}"
#         if chunk_id not in existing_ids:
#             vectorstore.add_texts(
#                 texts=[chunk],
#                 ids=[chunk_id],
#                 metadatas=[{"source": file_name, "chunk_index": idx}]
#             )
def add_chunks_to_vectorstore(vectorstore, chunks, file_name):
    """
    Add text chunks to the ChromaDB vector store with metadata.
    """
    existing_ids = set(vectorstore.get()["ids"])

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{file_name}_chunk_{idx}"
        if chunk_id not in existing_ids:
            vectorstore.add_texts(
                texts=[chunk],
                ids=[chunk_id],
                metadatas=[{"source": file_name}]
            )