from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    summary: Annotated(str,"A breif Summary of the review")
    sentiment : Annotated(str, """Return sentiment of the review: Either
    Negative, Positive, or Neutral""")
    language: Annotated(Optional(str), "Write the language used in the review")

structured_model = model.with_structured_output(Review)
result = strctured_model.invoke("Good Product")

##No Guarantee that LLM doesnot follow the type of the keys in the dictionary, TypedDict is only as a representation