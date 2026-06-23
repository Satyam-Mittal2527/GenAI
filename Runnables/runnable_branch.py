from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda, RunnableBranch


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
    template = "Write a detailed report on {topic}",
    input_variables = ["topic"]
)
prompt2 = PromptTemplate(
    template = "Summarize the following text {text}",
    input_variables = ["text"]
)

def count_word(text):
    return len(text.split())
report_gen_chain = RunnableSequence(prompt1, llm , parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split())> 500 , RunnableSequence(prompt2. llm. parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_gen_chain ,  branch_chain
)

print(final_chain.invoke({
    'topic': 'Russia Vs Ukarine war'
}))