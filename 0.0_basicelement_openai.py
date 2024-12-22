from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI


# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment variables or .env file.")



llm = OpenAI()

prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

chain = prompt | llm
response = chain.invoke(
    {
        "output_language": "Oriya",
        "input": "I love programming.",
    }
)

print(response)