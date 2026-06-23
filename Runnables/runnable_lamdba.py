from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough , RunnableLambda


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
    template = "Write a Joke about {topic}",
    input_variables = ["topic"]
)
parser = StrOutputParser()

# prompt2 = PromptTemplate(
#     template = "Give me a short paragraph of explanation about these jokes {response}",
#     input_variables = ["response"]
# )

def count_word(text):
    return len(text.split())

joke_gen_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'words': RunnableLambda(count_word)
})

chain = RunnableSequence(
    joke_gen_chain, parallel_chain
)
print(chain.invoke({
    "topic": "AI"
}))