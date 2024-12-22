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
        chain_type="stuff"
    )