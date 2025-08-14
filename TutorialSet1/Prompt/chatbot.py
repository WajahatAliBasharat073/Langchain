from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()


# llm=HuggingFaceEndpoint(
#     repo_id="google/flan-t5-small",
#     task="text-generation",
#     temperature= 0.5, 
#     max_length=64
# )

llm = ChatHuggingFace.from_model_id(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)

model=ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    else:
        result=model.invoke(user_input)
        print("Assistant:", result)