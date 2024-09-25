import os, uvicorn
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langgraph.prebuilt import create_react_agent
from langserve import add_routes
from langchain_text_splitters import RecursiveCharacterTextSplitter


# get keys
env = load_dotenv(find_dotenv())
openai = os.environ["OPENAI_API_KEY"]

# connect to ChatGPT
chatbot = ChatOpenAI(model="gpt-3.5-turbo")

# load data
loader = TextLoader("./data/be-good.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# retriever
retriever = vector_store.as_retriever() 

# set up parser
parser = StrOutputParser()

# chain prompt
template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"
prompt = ChatPromptTemplate.from_messages([("system", template),("human", "{input}")])

c = create_stuff_documents_chain(chatbot, prompt)

chain = create_retrieval_chain(retriever, c)
rag_chain = chain | parser

q = input("Ask a question: ")
response = chain.invoke({"input": q})
print(response["answer"])