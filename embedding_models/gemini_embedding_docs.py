from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

llm=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    temperature=0.1,
    max_output_tokens=1024,
    dimensions=50,
)
query=[
    "this is a car",
    "i like bike",
    "no one can ride a bike"
]

result=llm.embed_documents(query)
print(str(result)) #output will be a 2d list where each can be accces with index of the variable query