from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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

schema = [
    ResponseSchema(name= 'fact_1', description= 'Fact 1 about the topic'),
    ResponseSchema(name= 'fact_2', description= 'Fact 2 about the topic'),
    ResponseSchema(name= 'fact_3', description= 'Fact 3 about the topic')
]
parser = StrcturedOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give 3 facts about the {topic} \n {format_instructions}",
    input_variables = ['topic'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)
prompt = template.invoke({'topic': 'Black Hole'})

result = llm.invoke(prompt)

final_result = parser.parse(result)

print(final_result)