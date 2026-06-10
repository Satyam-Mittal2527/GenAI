from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model = "text-emdedding-3-large",
    dimensions=32
)

documents = [
    "Kathmandu is the capital of Nepal",
    "Delhi is the capital of India"
]
result = embedding.embed_documents(documents)


print(str(result))