from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
    )

model=ChatHuggingFace(llm=llm)


chat_template=ChatPromptTemplate([
    ('system',"You are a {domain} expert"), #this is mainly used inorder to maintain chat history in chat bot so that we can identify which message is sent by who
    ('human',"Explain in simple terms what is {topic}")
])

prompt=chat_template.invoke({
    'domain':'cricket',
    'topic':'Dusra'
})

result=model.invoke(prompt )

print(result.content)