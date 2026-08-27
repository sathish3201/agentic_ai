from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated,  Optional
from langchain_core.messages import  BaseMessage, HumanMessage, SystemMessage,AIMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv 
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph.message import add_messages
import os
from langchain_openai import ChatOpenAI
import sqlite3


#  tools--
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import requests
import math

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
STOCK_API_KEY = os.getenv("STOCK_API_KEY")

llm = None
def laptop_model():
    try:
        llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)
        # a real call is needed to confirm Ollama is actually reachable —
        # constructing ChatOllama never fails on its own, even if Ollama
        # isn't running
        llm.invoke("ping")
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

llm = laptop_model()
if llm is None:
    llm = online_model()

if llm == None:
    print("No model is available. Closing...")
    exit(1)
############## ---tools #####################

#tools by default it's tool no need decorator sign 
search_tool = TavilySearch(
    max_results = 5,
    topic='general',
    search_depth="advanced"
)
# ---------------------------------
#  calculator tools
@tool 
def calculator(expression: str) -> str:
    """User for simple mathematicall calculations 
        input should be a valid math expression
        Expression: 2+2,math.sqrt(16), 10*5
    """

    try:
        allowed ={
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum":sum
        }

        result= eval(expression, {"__builtins__":{}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"
# ---------------------------------------------------------------
#   stock tools 
def get_stock_price(symbol: str)-> dict:
    """
    fetch latest price for a given symbol (e.g. 'AAPL', 'TLSA')
    using Alpha vantage with API key in the URL
    """

    url =f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
    r= requests.get(url)
    return r.json()  

# ----------------------------------------------------------
# -------------------------- weather tool ---------------
@tool
def get_weather(location: str) -> dict:
    """
    Get the current weather for a location.

    Args:
        location: City or location name, e.g. "Hyderabad, India".

    Returns:
        Current weather information including temperature, humidity,
        precipitation, wind speed, and weather code.
    """

    # 1. Geocode location -> latitude/longitude
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=10,
    )
    geo_response.raise_for_status()

    geo_data = geo_response.json()

    if not geo_data.get("results"):
        return {
            "success": False,
            "error": f"Location not found: {location}",
        }

    place = geo_data["results"][0]

    latitude = place["latitude"]
    longitude = place["longitude"]

    # 2. Get current weather
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
            ]),
            "timezone": "auto",
        },
        timeout=10,
    )
    weather_response.raise_for_status()

    weather_data = weather_response.json()

    return {
        "success": True,
        "location": {
            "name": place["name"],
            "country": place.get("country"),
            "latitude": latitude,
            "longitude": longitude,
        },
        "timezone": weather_data.get("timezone"),
        "current": weather_data.get("current"),
        "units": weather_data.get("current_units"),
    }
# ------------------------------------------------------------
# ------------------------- binding tool with bind_tools---------
# make tool list 

tools = [get_stock_price, search_tool, calculator, get_weather]

# make llm tool aware
llm_with_tools= llm.bind_tools(tools)
# -----------------------------------------------------------------

##################################################
# creating state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state:ChatState):
    # take user query from state
    messages = state['messages']
    #send to llm
    response = llm_with_tools.invoke(messages)
    #response to store state
    return {'messages': [response]}

tool_node = ToolNode(tools) # execute tool calls



# graph

conn= sqlite3.connect(database='chatbot.db', check_same_thread=False)
# checkpoint= MemorySaver()
checkpoint =SqliteSaver(conn)

# ----------------------draw nodes--------------
graph= StateGraph(ChatState)
graph.add_node('chatnode', chat_node)
graph.add_node("tools",tool_node)

# ---------------------add edges-------------
# add edges 
graph.add_edge(START, "chatnode")
#
#if the llm asked the tool, go to toolnode or else finish
graph.add_conditional_edges("chatnode", tools_condition)

graph.add_edge("tools","chatnode")


# ----------compile --------------
chatbot = graph.compile(checkpointer = checkpoint)

# -------------------threads operations------------
def get_all_threads():
    all_threads= set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return list(all_threads)


def delete_thread(thread_id):
    """Permanently remove a thread's checkpoints and pending writes."""
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.commit()
# ----------------------------------------------

# thread_id = "1"
# initial_state = {
#     "messages" :[
#         HumanMessage(content= "What is my name")
#     ]
# }
# config={'configurable':{'thread_id': thread_id}}
# response = chatbot.invoke(initial_state, config=config)
# print(response['messages'][-1].content)
    
