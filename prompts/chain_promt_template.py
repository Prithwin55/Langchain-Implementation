from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

"""
command to run : streamlit run prompt_ui.py

"""


llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper=st.selectbox("Select research paper",["AIAUN","BERT","Transformer","GPT-3"])

style=st.selectbox("Select explanation style",["Begginer friendly","Technical","Code oriented","Mathmatical"])

length=st.selectbox("Select length",["1-2 paragraph","medium 3-5 paragraph","Long (detailed explanation)"])

template=load_prompt('prompt.json')



if st.button("Summaraize"):
    
    chain= template | model
    
    result=chain.invoke({
    'paper_input':paper,
    'style_input':style,
    'length_input':length
    })

    st.write(result.content)