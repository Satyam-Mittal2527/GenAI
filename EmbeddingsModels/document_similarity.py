from langchasin_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as mp
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

embedding = OpenAIEmbeddings(
    model="test-embedding-3-large",
    dimensions = 300
)
documents = [
    "Kathmandu is the capital of Nepal",
    "Delhi is the capital of India"
]
query = "Tell me about Nepal"

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score =sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]

print(document[index])
print(score)
