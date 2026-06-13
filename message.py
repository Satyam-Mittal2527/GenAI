from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
load_dotenv()

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

messages = [
    SystemMessage(content="You are a helpful assistance"),
    HumanMessage(content="Tell me about a langchain")
]
result = llm.invoke(messages)
messages.append(AIMessage(content = result))

print(messages)

