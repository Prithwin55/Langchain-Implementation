from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HF_HOME"]='/home/prithwin-ratnan/huggingface_cache'

llm=HuggingFacePipeline(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    pipeline_kwargs=dict(
        task="text-generation",
        temperature=0.1,
        max_new_tokens=100
    )
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of delhi?")
print(result.content)