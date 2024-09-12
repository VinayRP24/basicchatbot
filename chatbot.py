import os
from dotenv import load_dotenv, find_dotenv
env = load_dotenv(find_dotenv())
openai = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatbot = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.messages import HumanMessage

m = input("Enter message: ")
messages = [HumanMessage(content=m)]

response = chatbot.invoke(messages)
print(response.content)