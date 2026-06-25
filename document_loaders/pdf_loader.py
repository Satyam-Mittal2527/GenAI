from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    'Cricket_Poem_Demo.pdf'
)

docs = loader.load()

print(docs)