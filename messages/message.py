from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
    )

model=ChatHuggingFace(llm=llm)

messages=[
    SystemMessage("You are a helpful assistent"), #this is mainly used inorder to maintain chat history in chat bot so that we can identify which message is sent by who
    HumanMessage("Tell me about india")
]

result=model.invoke(messages)

print(result.content)