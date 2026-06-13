from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import streamlit as st
from ChatModels.chatmodel_hf_local import llm

load_dotenv()

template = PromptTemplate(
    template="""
    Summarize the paper {user_input} in a simple english and provide the future work
    """,
    input_variables=["user_input"]
)
    

st.header('Research Tool')

user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    chain = template | llm
    result = chain.invoke({
         'user_input' : user_input
    })
    st.write(result)