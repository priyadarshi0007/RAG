from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def initialize_retrieval_system(vectorstore, openai_api_key):
    """
    Set up the retrieval-based QA system.
    """
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

# def query_retrieval_system(chain, query):
#     """
#     Query the retrieval-based QA system and display results with metadata.
#     """
#     response = chain.invoke({"query": query})
#     print("\nResponse:", response["result"])
#     if "source_documents" in response:
#         for doc in response["source_documents"]:
#             source = doc.metadata.get("source", "Unknown Source")
#             chunk_index = doc.metadata.get("chunk_index", "Unknown Index")
#             print(f"Source: {source}, Chunk Index: {chunk_index}")

def query_retrieval_system(chain, query):
    """
    Query the retrieval-based QA system and display results with metadata.
    """
    response = chain.invoke({"query": query})
    print("\nResponse:", response["result"])
    if "source_documents" in response:
        print("\nSource Documents:")
        for doc in response["source_documents"]:
            source = doc.metadata.get("source", "Unknown Source")
            print(f"- From file: {source}")