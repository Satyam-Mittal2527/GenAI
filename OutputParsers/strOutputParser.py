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
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

template1 =  PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ['topic']
)
template2=  PromptTemplate(
    template = "Write a 5 lines summary on following text .\n {text}",
    input_variables = ['text']
)

parser  =StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({'topic':'Back Hole'})
print("final result")
print(result)