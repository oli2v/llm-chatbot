import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata

pdf_folder_path = "files/"
documents = []

for file in os.listdir(pdf_folder_path):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

print(len(documents))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(documents)
chunks = filter_complex_metadata(chunked_documents)
print(len(chunks))

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=FastEmbedEmbeddings(),
    persist_directory="chroma_store/",
)
vector_store.persist()
