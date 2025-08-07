from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
    )

model=ChatHuggingFace(llm=llm)

chat_template=ChatPromptTemplate([
    ('system',"you are a helpful assistent only give relevent answers for asked question"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human',"{query}")
])


chat_history=[]

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt=chat_template.invoke({
    'chat_history':chat_history,
    'query':"Where is my refund"
})

print(prompt)

result=model.invoke(prompt)
print(result.content)