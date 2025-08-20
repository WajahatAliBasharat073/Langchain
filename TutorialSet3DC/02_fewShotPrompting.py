from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

example_prompts = PromptTemplate.from_template("Question: {question}\nAnswer: {answer}")

prompt=example_prompts.invoke({
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris."
})
print(prompt.text)

Prompt_Template=FewShotPromptTemplate(
    examples=example_prompts,
    example_prompts=example_prompts,
    suffix="Question: {question}",
    input_variables=["question"],
)

few_shot_prompt = Prompt_Template.invoke({
    "question": "What is the capital of Germany?",
    "answer": "The capital of Germany is Berlin."
})  

prompt=few_shot_prompt.invoke({
    "question": "What is the capital of Germany?",
})
print(prompt.text)


llm=ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=150,
)

llm_chain = few_shot_prompt | llm

response = llm_chain.invoke({
    "question": "What is the capital of Germany?",
    "answer": "The capital of Germany is Berlin."
})
print(response.text)  # Output: 'The capital of Germany is Berlin.'
