from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os 
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader("NBC.pdf")
docs= loader.load()

openai_key = os.getenv("OPENAI_API_KEY")
text_spillter= RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
documents= text_spillter.split_documents(docs)


db = Chroma.from_documents(documents[:5] , OpenAIEmbeddings())

# query = "who is the author of NBC"
# result = db.similarity_search(query)
# print(result[0].page_content)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  
    temperature=0.7,   
)

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")


documents_chain = create_stuff_documents_chain(llm, prompt)

retriever = db.as_retriever()

retriever_chain = create_retrieval_chain(retriever, documents_chain)
final = retriever_chain.invoke({"input" : 'sitting in waiting area'})

print("hello " , final['answer'])