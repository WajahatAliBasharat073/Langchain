from langchain_openai import chatmodelOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

chat_model = chatmodelOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=150
)
chat_model.invoke("What is the capital of France?")
