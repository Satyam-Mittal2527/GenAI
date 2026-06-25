from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    'csv_file_path'
)

docs = loader.load()

print(docs)