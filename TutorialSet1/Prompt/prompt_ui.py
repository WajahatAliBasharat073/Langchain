from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)

import time
from requests.exceptions import HTTPError

for _ in range(3):
    try:
        model.invoke("Hello, how are you?")
        break
    except HTTPError as e:
        print("Error, retrying in 5 seconds...", e)
        time.sleep(5)
