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
# -------rag based system--------------
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ----------------------imports----------

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
            # the phone's cache_proxy.py buffers the full response and
            # always returns plain JSON, never real SSE — requesting a
            # stream here gets back application/json instead of
            # text/event-stream, which the client's SSE parser can't
            # read, raising "No generations found in stream"
            streaming=False,
        )
        return llm
    except Exception as e:
        print("Could not connect to OpenAI. Make sure OpenAI app is running on your machine:", e)
    return None

# llm = online_model()
# if llm is None:
#     llm = laptop_model()

llm = laptop_model()


if llm == None:
    print("No model is available. Closing...")
    exit(1)
# -------------------embeddings-----------
embedding_model = FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# -----------rag function for pdf file traverse-------------
def ingest_rag_documents(file_path):
    # DB_PATH = './chroma_db'
    loader=PyPDFLoader(file_path)
    docs = loader.load()
    print(docs)
    spliter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)
    chunks = spliter.split_documents(docs)
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        collection_name='my_pdf_document',
        persist_directory='./my_chroma_db'
    )

# -------------retriving data-------------
def get_retriever():
    vector_store=Chroma(
        embedding_function= embedding_model,
        collection_name='my_pdf_document',
        persist_directory='./my_chroma_db'
    )
    retriver = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
    return retriver
############## ---tools #####################


# --------------------------------Rag Tools-----------------
@tool
def rag_tool(query:str) -> str:
    """ Retrieve relavant infromation from pdf documents 
    use this tool when user ask the factual concept questions that may be answered using the stored PDF documents """
    retriver = get_retriever()
    documents = retriver.invoke(query)
    if not documents:
        return "No relavant information found in the PDF"

    formatted_documents = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source","unknown source")
        page = document.metadata.get("page","unknown source")

        formatted_documents.append(
            f"Document: {index}\n"
            f"Source: {source}\n"
            f"Page: {page} \n"
            f"Content: {document.page_content}"
        )
    return "\n\n".join(formatted_documents)


# ---------------------------------------------------------------
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
    """Evaluate a math expression, e.g. "2+2", "math.sqrt(16)", "10*5"."""

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

@tool 
def purchase_tool(symbol:str, quantity:int) -> dict:
    """ Simulate purchase a given quantity of a stock symbol
    NOTE: tHIS IS MOCK IMPLEMENTATION 
    - NO REAL BREAKAGE API CALLED
    -IT SIMPLY RETURN CONFIRMATION PAYLOAD
    """
    return {
        "status": "success",
        "message":f"purchase order placed for {quantity} shares of symbol {symbol}",
        "symbol":symbol,
        "quantity": quantity
    }
# -------------------------- weather tool ---------------
@tool
def get_weather(location: str) -> dict:
    """Get current weather (temp, humidity, wind) for a city, e.g. "Hyderabad, India"."""

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

tools = [get_stock_price, search_tool, calculator, get_weather,rag_tool, purchase_tool]

# make llm tool aware
llm_with_tools= llm.bind_tools(tools)
# -----------------------------------------------------------------

##################################################
# creating state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


STYLE_PROMPT = SystemMessage(content=(
    "For explain/describe/write-about answers: use Markdown (headings, "
    "bullets, numbered steps, code blocks, tables) to stay scannable, not "
    "one paragraph. For greetings or single-fact answers (calc, weather, "
    "stock price): reply in one plain line, no formatting."
))


def chat_node(state:ChatState):
    system_message=SystemMessage(content="""
You are an intelligent general-purpose AI assistant with access to several tools.

Your job is to understand the user's request, decide whether a tool is required,
use the appropriate tool, and then provide a concise, accurate final answer.

AVAILABLE TOOLS:

1. get_stock_price
   - Use when the user asks for a current stock/market price.
   - Use the tool instead of guessing or relying on outdated knowledge.
   - Examples:
     "What is Apple's stock price?"
     "What's the current price of TSLA?"
1.1 purchase_tool
   - Use when the user asks to purchase current stock/market price.

2. search_tool
   - Use when the user asks for information that requires web/current information.
   - Use for recent news, current facts, websites, events, or information not
     reliably available from your internal knowledge.
   - Do not use it for simple calculations or questions answerable from
     retrieved documents.

3. calculator
   - Use for arithmetic, mathematical calculations, percentages,
     conversions, or expressions where exact computation is required.
   - Never perform complicated arithmetic mentally when the calculator
     can provide an exact result.

4. get_weather
   - Use when the user asks about current or forecast weather.
   - Examples:
     "What's the weather in Hyderabad?"
     "Will it rain tomorrow?"
     "What's the temperature in London?"

5. rag_tool
   - Use when the user asks questions about documents provided/uploaded
     by the user.
   - The documents may include PDFs, research papers, reports, manuals,
     policies, or other indexed documents.
   - Use rag_tool before answering document-specific questions.
   - Do not invent information that is not present in the retrieved context.
   - If the retrieved documents do not contain enough information, clearly
     state that the answer cannot be determined from the available documents.
   - When possible, mention the relevant document or page information
     returned by the tool.

TOOL SELECTION RULES:

- Use the minimum number of tools necessary.
- Do not call tools unnecessarily.
- If a request requires multiple tools, call each appropriate tool.
- Tool results are evidence. Do not contradict reliable tool results.
- Never fabricate tool results.
- If a tool fails, explain the problem briefly and continue only if you
  can answer reliably without that tool.

RAG PRIORITY:

If the user asks about an uploaded document, prefer rag_tool over search_tool.

For example:

User:
"What does the uploaded research paper say about transformers?"

Action:
rag_tool

Do NOT search the web unless the user explicitly asks for external/current
information or the answer requires information outside the uploaded document.

If the user asks:

"Compare my uploaded paper with the latest research."

Action:
1. rag_tool
2. search_tool

Then combine the retrieved document information with current web information.

GENERAL BEHAVIOR:

- Understand the user's intent before selecting a tool.
- Do not expose internal reasoning or tool-selection reasoning.
- Do not mention tools unless useful to the user.
- Answer directly after receiving tool results.
- Keep answers clear and relevant.
- If the user asks for a calculation, use calculator.
- If the user asks for weather, use get_weather.
- If the user asks for stock prices, use get_stock_price.
- If the user asks about uploaded documents, use rag_tool.
- If the user asks for current/external information, use search_tool.

DOCUMENT GROUNDING:

When using rag_tool, treat retrieved document content as the primary
source of truth for document-specific questions.

If the retrieved context says:

"Information not found"

do not attempt to manufacture an answer.

If the document contains conflicting information, acknowledge the conflict
rather than silently choosing an answer.

FINAL ANSWER:

After using a tool, provide the user with the answer rather than merely
describing what the tool returned.
    """)
    # take user query from state
    messages = state['messages']
    #send to llm, with a style instruction prepended so replies stay scannable
    response = llm_with_tools.invoke([STYLE_PROMPT] + messages)
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
    
