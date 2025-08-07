from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
load_dotenv()

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_output_tokens=1024,
)

# Schema
class Review(TypedDict): 
    summary:str
    sentiment:Literal["pos","neg"] #To provice selction from options

structured_output=llm.with_structured_output(Review)
result=structured_output.invoke("The Hyundai Creta 2025 continues to impress with its bold design, spacious interiors, and feature-packed cabin. The 1.5L turbo petrol engine offers smooth performance, while the suspension handles city and highway drives comfortably. The panoramic sunroof, large infotainment screen, and ADAS features add a premium feel.")
print(result)
# print(result.get("sentiment"))
print(result["sentiment"])

"""
Disadvantage: although we specify the datas structure its not necessary that llm outputs the same data
the data is not valildated

"""