from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)


documents=[
    "There was a boy going to home",
    "virat is a good batter and player",
    "ronaldo is very talented"
]

query="who is virat?"

response_doc=model.embed_documents(documents)

response_query=model.embed_query(query)

result= cosine_similarity([response_query],response_doc)[0] #cosine similary function expects a 2d array as input



index,conf=sorted(list(enumerate(result)),key=lambda x:x[1])[-1]
print(documents[index])
