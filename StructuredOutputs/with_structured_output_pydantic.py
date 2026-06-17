from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):

    summary = str = Field(description="A breif Summary of the review")
    sentiment = Literal["pos","neg"] = Field(description="""Return sentiment of the review: Either
    Negative, Positive, or Neutral""")
    language = Optional[str] = Field("Write the language used in the review")


structured_model = model.with_structured_output(Review)
result = strctured_model.invoke("Good Product")

