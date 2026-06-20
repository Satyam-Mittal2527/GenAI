from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

class Person(BaseModel):
    name : str= Field(description = "Name of the Person")
    age : int = Field(description = "Age of the Person")
    city : str = Field(description = "City of the persion where person lives")

parser = PydanticOutputParser(
    pydantic_object = Person
)

template = PromptTemplate(
    template = "Generate the name, age and city of a frictional {place} person \n {format_instruction}",
    input_variables = ['place'],
    partial_variables = {"format_instruction": parser.get_format_instructions()}
)

chain = template | llm | parser

final_result = chain.invoke({'place': 'Nepali'})

# final_result = parser.parse(result)
print("After invoke result")
print(final_result)