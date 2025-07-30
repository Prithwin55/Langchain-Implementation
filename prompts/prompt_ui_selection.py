from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

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

template=PromptTemplate(template="\nPlease summarize the research paper titled \"{paper_input}\" with the following specifications:\nExplanation Style: {style_input}  \nExplanation Length: {length_input}  \n1. Mathematical Details:  \n   - Include relevant mathematical equations if present in the paper.  \n   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  \n2. Analogies:  \n   - Use relatable analogies to simplify complex ideas.  \nIf certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing.  \nEnsure the summary is clear, accurate, and aligned with the provided style and length.\n",
input_variables=['paper_input','style_input','length_input'], #validates{ variable } where data can be mapped
validate_template=True
)

prompt=template.invoke({
    'paper_input':paper,
    'style_input':style,
    'length_input':length
})


if st.button("Summaraize"):
    result=model.invoke(prompt)
    st.write(result.content)