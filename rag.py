import os, uvicorn
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
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
prompt  = ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chatbot
    | parser
)

q = input("Ask a question: ")
response = rag_chain.invoke(q)
print(response)