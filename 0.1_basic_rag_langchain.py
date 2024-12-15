import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Check for the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment variables or .env file.")

# Create a test document as a list of `Document` objects
# docs = [Document(page_content="This is a test document. It contains some sample text to test chunk splitting.")]

# Load documents from a local file
loader = TextLoader("/Users/priyadarshisoumyakumar/Downloads/RAG/randomdocs.txt")
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)  # Adjust chunk size and overlap as needed
splits = text_splitter.split_documents(docs)

# # Print the resulting chunks
# for i, split in enumerate(splits, 1):
#     print(f"Chunk {i}: {split.page_content}")

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
# prompt = hub.pull("rlm/rag-prompt")
# print(type(prompt))
# Pull the prompt from LangChain Hub
# hub_prompt = hub.pull("langchain/prompts/retrieval_qa_strict")

# # Define your custom instructions if needed
# custom_instructions = """
# You are an assistant answering questions based strictly on the provided context.
# If the answer is not in the context, respond with: "The information is not available in the provided documents."
# """

# # Merge custom instructions with hub prompt
# hub_prompt.template = f"{custom_instructions}\n\n{hub_prompt.template}"

# Define a strict retrieval QA prompt
strict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant answering questions strictly based on the provided context.

Context:
{context}

Question:
{question}

If the information is not available in the context, respond with: "The information is not available in the provided documents."
"""
)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | strict_prompt
#     | llm
#     | StrOutputParser()
# )

# Create the Retrieval-based QA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": strict_prompt},
)

while True:
    # Prompt the user for input
    user_input = input("Enter your query (type 'exit' to quit): ")
    
    # Check if the user wants to exit
    if user_input.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break
    
    # Use RAG chain to process the input and get a response
    response = rag_chain.invoke(user_input)
    
    # Print the response
    print(f"Response: {response}")