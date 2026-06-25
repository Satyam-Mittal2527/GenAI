from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


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

prompt = PromptTemplate(
    template = "Answer the following questions \n {question} from the following text \n {text}",
    input_variables = ["question", "text"]
)
parser = StrOutputParser()

chain = prompt | llm | parser


url = "https://manmohey-seven.vercel.app/"

loader = WebBaseLoader(url)

docs = loader.load()

result = chain.invoke({
    'question': "What are the products",
    "text": docs[0].page_content
})

print(result)