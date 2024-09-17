import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# get keys
env = load_dotenv(find_dotenv())
openai = os.environ["OPENAI_API_KEY"]

# connect to ChatGPT
chatbot = ChatOpenAI(model="gpt-3.5-turbo")

# set up message
temp = "Translate the following from English into French (Canada)"
m = input("Enter message: ")
messages = [SystemMessage(content=temp), HumanMessage(content=m)]

# set up parser
parser = StrOutputParser()

# chain prompt
chain = chatbot | parser
response = chain.invoke(messages)
print(response)