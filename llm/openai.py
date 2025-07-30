from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
llm = OpenAI(
    model="gpt-4",
    temperature=0.1,
    max_tokens=1024,
)
result = llm.invoke("Write a poem about the sea")
print(result.content)