from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.agents import create_openai_tools_agent


from dotenv import load_dotenv
load_dotenv()

api_wrapper  = WikipediaAPIWrapper(top_k_results = 1 , doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper= api_wrapper)
print(wiki.name)

loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=Chroma.from_documents(documents,OpenAIEmbeddings())
retriever=vectordb.as_retriever()

retrieval_tools = create_retriever_tool(retriever ,"langsmith_search","Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

print(retrieval_tools.name)

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
print(arxiv.name)

tools = [wiki, retrieval_tools, arxiv]