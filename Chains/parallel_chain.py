from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1500,
    do_sample=True,
    temperature=0.7,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from following text \n {text}",
    input_variables = ["text"]
)
prompt2 = PromptTemplate(
    template = "Generate 5 questions from the folowing text \n {text}",
    input_variables = ["text"]
)
prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes:{notes} and quix:{quiz}",
    input_variables = ["notes","quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes" : prompt1 | llm | parser,
    "quiz" : prompt2 | llm | parser
})

text = """

Deep learning is a subset of machine learning 
that uses multi-layered artificial neural networks 
to solve complex problems. Inspired by the human brain, 
it allows computers to learn directly from raw data—such as images, text, and sound—without 
requiring humans to manually extract features
"""

merge_chain = prompt3 | llm | parser

chain = parallel_chain | merge_chain

result = chain.invoke({
    "text": text
})
print(result)

chain.get_graph().print_ascii()
