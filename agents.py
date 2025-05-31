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
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

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

retrieval_tool = create_retriever_tool(
    retriever,
    name="langsmith_tool",
    description="Use this tool to answer any question related to LangSmith documentation, features, usage, or APIs."
)

print(retrieval_tool.name)

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
print(arxiv.name)

tools = [wiki, arxiv, retrieval_tool]

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",  
    temperature=0,   
)
prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_tools_agent(llm, tools , prompt)

executor = AgentExecutor(agent= agent, tools= tools , verbose = True)
# print(executor)

result = executor.invoke({"input":"What's the paper 1605.08386 about?"})
print(result)