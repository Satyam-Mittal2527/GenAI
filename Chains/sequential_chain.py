from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt1 = PromptTemplate(
    template = "Give me a deatiled report on topic {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Give me 5 important points from the following text \n {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({'topic': 'Cricket'})

print(result)

chain.get_graph().print_ascii()