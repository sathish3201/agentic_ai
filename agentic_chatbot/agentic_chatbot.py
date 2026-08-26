from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import  BaseMessage, HumanMessage, SystemMessage,AIMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv 
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph.message import add_messages

load_dotenv()


llm = ChatOllama(model="qwen2.5:3b", temperature="0.3")

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

checkpoint= MemorySaver()

graph= StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

# add edges 
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpoint)

# thread_id = "1"
# initial_state = {
#     "messages" :[
#         HumanMessage(content= "What is my name")
#     ]
# }
# config={'configurable':{'thread_id': thread_id}}
# response = chatbot.invoke(initial_state, config=config)
# print(response['messages'][-1].content)
    
