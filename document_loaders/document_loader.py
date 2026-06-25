from langchain_community.document_loader import DirectoryLoader, pyPDFLoader

loader = DirectoryLoader(
    path = "Directory_path",
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.load()

print(docs)