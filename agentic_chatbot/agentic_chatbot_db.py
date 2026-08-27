from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import  BaseMessage, HumanMessage, SystemMessage,AIMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv 
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph.message import add_messages
import os
from langchain_openai import ChatOpenAI
import sqlite3

load_dotenv()



llm = None
def laptop_model():
    try:
        llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)
        # response = llm.invoke("Hello, check if you are active.")
        # print("Ollama is available. Response:")
        # print(response.content)
    except Exception as e:
        print("Could not connect to Ollama. Make sure Ollama app is running on your machine:", e)
    return llm


def laptop_model():
    try:
        llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)
        # response = llm.invoke("Hello, check if you are active.")
        # print("Ollama is available. Response:")
        # print(response.content)
        return llm
    except Exception as e:
        print("Could not connect to Ollama. Make sure Ollama app is running on your machine:", e)
    return None


# llm = ChatOllama(model="qwen2.5:3b", temperature="0.3")
def online_model():
    try:
        llm = ChatOpenAI(
            base_url=os.environ["LOCAL_MODEL_URL"] + "/v1",
            api_key=os.environ["LOCAL_MODEL_API_KEY"],
            model=os.environ["LOCAL_MODEL_NAME"],
            temperature=0.3,
        )
        return llm
    except Exception as e:
        print("Could not connect to OpenAI. Make sure OpenAI app is running on your machine:", e)
    return None

if os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true":
    llm = laptop_model()
else:
    llm = online_model()

if llm == None:
    print("No model is available. Closing...")
    exit(1)

# creating state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state:ChatState):
    # take user query from state
    messages = state['messages']
    #send to llm
    response = llm.invoke(messages)
    #response to store state
    return {'messages': [response]}




# graph

conn= sqlite3.connect(database='chatbot.db', check_same_thread=False)
# checkpoint= MemorySaver()
checkpoint =SqliteSaver(conn)

graph= StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

# add edges 
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpoint)

def get_all_threads():
    all_threads= set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return list(all_threads)


# thread_id = "1"
# initial_state = {
#     "messages" :[
#         HumanMessage(content= "What is my name")
#     ]
# }
# config={'configurable':{'thread_id': thread_id}}
# response = chatbot.invoke(initial_state, config=config)
# print(response['messages'][-1].content)
    
