from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader("NBC.pdf")
docs= loader.load()

textspillter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=200)
documents = textspillter.split_documents(docs)

db = Chroma.from_documents(documents , OllamaEmbeddings()) 