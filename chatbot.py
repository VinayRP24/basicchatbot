import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

# get keys
env = load_dotenv(find_dotenv())
openai = os.environ["OPENAI_API_KEY"]

# connect to ChatGPT
chatbot = ChatOpenAI(model="gpt-3.5-turbo")

# create documents
documents = [
    Document(
        page_content="Vinay is very cool. He is know for being very smart and diligent. He once fought a bear!",
        metadata={"source": "vinay-doc"},
    ),
    Document(
        page_content="Noah is very bored",
        metadata={"source": "noah-doc"},
    ),
    Document(
        page_content="Riya is very happy",
        metadata={"source": "riya-doc"},
    ),
    Document(
        page_content="Eve is very sad",
        metadata={"source": "eve-doc"},
    ),
]

vector_store = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

# retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":1}) # best result

# set up search
search = TavilySearchResults(max_results=1)
tools = [search]
agent = create_react_agent(chatbot, tools)

# set up 
l = input("Choose language: ")
question = """
Answer this question using the provided context.
Additionaly translate the answer from English into """ + l + """
{question}

Context: {context}
"""
prompt = ChatPromptTemplate.from_messages([("system", question)])

# set up message
q = input("Enter question: ")

# set up parser
parser = StrOutputParser()

# chain prompt
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | chatbot | parser  # for custom documents
#chain = prompt | chatbot | parser
#response = agent.invoke({"messages": HumanMessage(content=q)})
response = chain.invoke(q)
print(response)