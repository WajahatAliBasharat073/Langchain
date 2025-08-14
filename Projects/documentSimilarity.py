from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

chat_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=300
)

documents = [
    "France is a country in Europe.",
    "The capital of France is Paris.",
    "France is known for its art, fashion, and culture.",
    "Paris is the capital city of France.",
    "The Eiffel Tower is located in Paris.",
    "France has a rich history and diverse culture.",
    "The Louvre Museum in Paris is famous for its art collections.",
    "French cuisine is renowned worldwide.",
    "The Seine River flows through Paris.",
    "France is a member of the European Union.",
    "The French Revolution began in 1789.",
    "French is the official language of France.",
    "The Mont Saint-Michel is a historic island commune in Normandy, France.",
    "The Palace of Versailles is located near Paris.",
    "France is known for its wine and cheese.",
    "The French flag consists of three vertical stripes: blue, white, and red.",
    "The Tour de France is a famous annual"
]

query = "What is the capital of France?"
query_embedding = chat_model.embed_query(query)
document_embeddings = chat_model.embed_documents(documents)

similarities = cosine_similarity([query_embedding], document_embeddings)[0]
print("Similarities:", similarities)
top_indices = np.argsort(similarities)[-5:][::-1]