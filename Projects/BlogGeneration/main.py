import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


## function to get response from the model
def get_response(input_text, no_words, blog_style):

    ## Calling the CTransformers model
    llm = CTransformers(model="gpt2", model_type="gpt2")

    ## Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["input_text", "no_words", "blog_style"],
        template="Wrie a {no_words}-word blog post on the topic '{input_text}' in a {blog_style} style."
    )   
    ## Format the prompt with the input values
    prompt = prompt_template.format(input_text=input_text, no_words=no_words, blog_style=blog_style)    
    ## Get the response from the model
    response = llm(prompt)  
    ## Return the response
    return response 



## set streamlit
st.set_page_config(page_title="Blog Generation", page_icon=":guardsman:", layout="wide")
st.title("Blog Generation with LangChain and CTransformers")    
st.markdown("This application generates a blog post based on a given topic using LangChain and CTransformers.")



input_text=st.text_input("Enter a topic for the blog post:", "Artificial Intelligence in 2025")

## create two more columns for additional two fields
col1, col2 = st.columns([5,5])
with col1:
    no_words=st.text_input("Enter the number of words for the blog post:", "500")
with col2:
    blog_style=st.selectbox("Select the style of the blog post:", ["Informative", "Conversational", "Technical", "Casual"],index=0)

submit_button = st.button("Generate Blog Post")

if submit_button:
    st.write(get_response(input_text, no_words, blog_style))
    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["input_text", "no_words", "blog_style"],
        template="Generate a {no_words}-word blog post on the topic '{input_text}' in a {blog_style} style."
    )
