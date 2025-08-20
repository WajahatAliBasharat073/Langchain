from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm=OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=150,
    api_key=os.getenv("OPENAI_APIKEY")
)

llm.invoke("What is the capital of France?")
# Output: 'The capital of France is Paris.'
