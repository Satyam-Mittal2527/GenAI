from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field


load_dotenv()

model = ChatOpenAI()

json_schema = {
    "title":"student",
    "desciption":"schema about student",
    "type":"object",
    "properties":{
        "name": "string",
        "age":"integer"
    },
    "required": ["name"]
}



structured_model = model.with_structured_output(json_schema)
result = strctured_model.invoke("Satyam Mittal with age is 20 and in AI department")

