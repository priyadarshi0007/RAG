import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Ensure it is set in the environment variables.")

# Example documents
documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog."),
    Document(page_content="LangChain is a powerful framework for LLMs."),
    Document(page_content="Embeddings create numerical representations of text.")
]

# Function to split text into manageable chunks
def split_text_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks using a text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = []
    for doc in documents:
        text_chunks.extend(text_splitter.split_text(doc.page_content))
    return text_chunks

# Function to create and save a vector store
def create_and_save_vector_store(text_chunks, persist_directory="./chromadb"):
    """Create a ChromaDB vector store from text chunks and save it locally."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=persist_directory)
    # vector_store.persist()

# Function to load the conversational chain with a static prompt.
def get_conversational_chain():
    """Initialize a conversational chain with a custom prompt and Gemini model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, respond with "answer is not available in the context". Do not provide incorrect answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to handle user input and generate a response
def handle_user_question(user_question, persist_directory="./chromadb"):
    """Search for relevant documents and generate an answer to the user's question."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    docs = vector_store.similarity_search(user_question, k=3)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Preprocess documents and create vector store
def setup_documents(documents, persist_directory="./chromadb"):
    """Preprocess documents and set up the vector store."""
    text_chunks = split_text_into_chunks(documents)
    create_and_save_vector_store(text_chunks, persist_directory)

# Main Script
if __name__ == "__main__":
    # Set up the documents and vector store
    print("Setting up documents and vector store...")
    setup_documents(documents)

    # Interactive Question-Answer Loop
    print("Chat with your documents. Type 'exit' to quit.")
    while True:
        user_question = input("\nAsk a question: ").strip()
        if user_question.lower() == "exit":
            print("Goodbye!")
            break
        if not user_question:
            print("Please ask a valid question.")
            continue
        try:
            response = handle_user_question(user_question)
            print(f"Answer: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")