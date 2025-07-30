from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
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
user_input=st.text_input("Ask a question")

if st.button("Summaraize"):
    result=model.invoke(user_input)
    st.write(result.content)