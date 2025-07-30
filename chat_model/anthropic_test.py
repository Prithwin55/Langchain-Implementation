from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()
llm = ChatAnthropic(
    model="claude-2",
    temperature=0.1,)
result = llm.invoke("Write a poem about the sea")
print(result.content)