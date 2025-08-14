from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

import os

load_dotenv()
chat_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=36,
    # api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_tokens=150
)
documents = [
    "France is a country in Europe.",
    "The capital of France is Paris.",
    "France is known for its art, fashion, and culture."
]
chat_model.embed_query("France is a country in Europe.")
chat_model.embed_documents(documents)

