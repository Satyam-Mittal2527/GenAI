from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
parser = JsonOutputParser()

template = PromptTemplate(
    template="""
<|system|>
You are a helpful assistant that outputs ONLY valid JSON.
<|user|>
Generate a name, age and city for a person called Friction.

{format_instruction}
<|assistant|>
""",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | llm | parser
print("Output after parser")
final_result = chain.invoke({})
print(final_result)