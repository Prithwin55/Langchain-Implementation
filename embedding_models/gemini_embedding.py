from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

llm=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    temperature=0.1,
    max_output_tokens=1024,
    dimensions=50,
)

result=llm.embed_query("my name is prithwin")
print(str(result))