from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
template="Explain the following concept in simple terms: {concept}"

prompt_template = PromptTemplate.from_template(
    template=template
)

prompt=prompt_template.invoke({"concept": "Quantum Computing"})

llm=HuggingFacePipeline.from_model_id(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)

llm_chain=prompt_template | llm

response=llm_chain.invoke({"concept": "Quantum Computing"}) 



template=ChatPromptTemplate.from_message(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Answer the following question"),
        ("ai","2+2=4"),
        ("human", "What is the capital of France?{math}")
    ]
)

chat_model = HuggingFacePipeline.from_model_id(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)   
chat_chain = template | chat_model

response = chat_chain.invoke({"math": "2+2=4"}) 
print(response)


