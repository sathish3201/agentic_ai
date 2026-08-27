from agentic_chatbot_db_tools import chatbot, get_all_threads, delete_thread
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid


# -------------------------------------
# generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

#prevents the same thread id from being used multiple times
def add_thread(thread_id):
    """Add a new thread ID if it doesn't exist"""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def _make_title(text):
    """Trim a message down to a short, sidebar-friendly title."""
    title = text.strip().splitlines()[0]
    if len(title) > 40:
        title = title[:40].rstrip() + "..."
    return title


def get_thread_title(thread_id):
    """Return a display title for a thread, deriving it from the thread's
    first stored user message (and caching it) if not already known."""
    titles = st.session_state['thread_titles']
    if thread_id not in titles:
        for message in load_conversations(thread_id):
            if isinstance(message, HumanMessage) and message.content:
                titles[thread_id] = _make_title(message.content)
                break
    return titles.get(thread_id, thread_id)


def set_thread_title_from_message(thread_id, message):
    """Derive a short chat title from the first user message, once."""
    if thread_id not in st.session_state['thread_titles']:
        st.session_state['thread_titles'][thread_id] = _make_title(message)

#reset 
def reset_chat():
    """Reset the chat"""
    st.session_state['thread_id'] = generate_thread_id()

    st.session_state['message_history'] = []
    add_thread(st.session_state['thread_id'])


# loading conversations
def load_conversations(thread_id):
    """Load conversation from thread"""
    state=chatbot.get_state(config={'configurable':{'thread_id':thread_id}})
    # return emptyempty not emp
    return state.values.get("messages", [])


st.set_page_config(page_title="Agentic AI ChatBot", page_icon="🤖", layout="wide")

# ============ LAYOUT + STYLING ===============================================
st.markdown(
    """
    <style>
    /* ---- overall canvas: give the chat room to breathe, cap it at a
       comfortable reading width, and keep it centered instead of pinned
       to a cramped 700px column or stretched edge-to-edge ---- */
    .block-container {
        max-width: 880px;
        margin: 0 auto;
        padding-top: 2rem;
        padding-bottom: 6rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ---- header ---- */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.25rem;
    }
    .app-header .icon {
        font-size: 2rem;
        line-height: 1;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .app-subtitle {
        color: #6B7280;
        font-size: 0.95rem;
        margin: 0 0 1.75rem 0;
    }

    /* ---- sidebar: wider, padded, visually separated ---- */
    section[data-testid="stSidebar"] {
        min-width: 300px;
        background: #FAFAFC;
        border-right: 1px solid #ECECF1;
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] button {
        border-radius: 10px;
        width: 100%;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    section[data-testid="stSidebar"] .stButton {
        margin-bottom: 0.4rem;
    }

    /* ---- chat messages: more vertical breathing room between turns ---- */
    div[data-testid="stChatMessage"] {
        margin-bottom: 1.1rem;
        gap: 0.75rem;
    }

    /* user messages: right-aligned, avatar on the right, blue bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        text-align: right;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"])
        div[data-testid="stChatMessageContent"] {
        text-align: right;
        display: flex;
        justify-content: flex-end;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"])
        div[data-testid="stChatMessageContent"] > div {
        background: #DBEAFE;
        border-radius: 18px 18px 4px 18px;
        padding: 0.7rem 1rem;
        display: inline-block;
        max-width: 85%;
        text-align: left;
    }

    /* assistant messages: soft grey bubble, rounded on the left */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"])
        div[data-testid="stChatMessageContent"] > div {
        background: #F3F4F6;
        border-radius: 18px 18px 18px 4px;
        padding: 0.7rem 1rem;
        display: inline-block;
        max-width: 85%;
    }

    /* ---- "thinking / tool status" card ---- */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #E6E6EA;
        background: #FCFCFD;
    }

    /* ---- chat input: wider, more padding, sits nicely above the fold ---- */
    div[data-testid="stChatInput"] textarea {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <span class="icon">🤖</span>
        <h1>Agentic AI ChatBot</h1>
    </div>
    <p class="app-subtitle">Built with LangGraph · web search, calculator, stock &amp; weather tools</p>
    """,
    unsafe_allow_html=True,
)

#loading message history from previous conversations
if 'message_history' not in st.session_state:
    st.session_state['message_history']= []

#list of thread ids
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']= get_all_threads()

#thread_id -> display title
if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    # add_thread(st.session_state['thread_id'])


# ADD CURRENT THREAD TO CONVERSATION LIST
add_thread(st.session_state['thread_id'])

# ======= sidebar  threading feature title ==============================

st.sidebar.title("💬 My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):

    #reset the current chat and create a new thread
    reset_chat()

    #rerun the streamlit app to update the interface
    st.rerun()

st.sidebar.markdown("<div style='margin-bottom:0.75rem'></div>", unsafe_allow_html=True)


# =============DISPLAYING ALL CONVERSATIONS IN REVERSE ORDER ======
for thread_id in st.session_state['chat_threads'][::-1]:

    is_active = thread_id == st.session_state['thread_id']

    #ONE ROW PER CONVERSATION: select button + delete button
    col_select, col_delete = st.sidebar.columns([6, 1], gap="small")

    with col_select:
        label = ("🟢 " if is_active else "") + get_thread_title(thread_id)
        if st.button(label, key=thread_id, use_container_width=True):

            # SET SELECTED THREAD AS THE CURRENT THREAD
            st.session_state['thread_id'] = thread_id

            # load messages SAVED UNDER THE SELECTED THREAD
            messages = load_conversations(thread_id)
            temp_messages = []
            for message in messages:
                # check whether message sent by user
                if isinstance(message, HumanMessage):
                    role = 'user'

                elif isinstance(message, AIMessage):
                    role = 'assistant'
                # ignore other message types, such as ToolMessage
                else:
                    continue
                temp_messages.append({'role': role, 'content': message.content})
            # save the conversation to session state
            # st.session_state[f'thread_{thread_id}'] = temp_messages
            st.session_state['message_history'] = temp_messages
            #RERUN THE APPLICATION TO DISPALY THE LOADED MESSAGES
            st.rerun()

    with col_delete:
        if st.button("🗑️", key=f"delete_{thread_id}"):
            delete_thread(thread_id)
            st.session_state['chat_threads'].remove(thread_id)
            st.session_state['thread_titles'].pop(thread_id, None)

            # if the deleted thread was active, switch to a fresh new chat
            if st.session_state['thread_id'] == thread_id:
                reset_chat()

            st.rerun()

# ===========================MAIN CHAT INTERFACE ==================

#display all messages from currently selected conversation
for message in st.session_state['message_history']:
    #create either user chatb bubble or assistant chat bubble
    with st.chat_message(message['role']):

        #dispaly message conetent
        st.markdown(message['content'])



user_input= st.chat_input('Type Here')
#processing user input
if user_input:
    set_thread_title_from_message(st.session_state['thread_id'], user_input)
    st.session_state['message_history'].append({'role':'user', 'content': user_input})

    # display the user message in the chat interface
    with st.chat_message('user'):
        st.markdown(user_input)


    #pass current thread it to  langgraph
    # langraph use this id to save and retrieve the messages from the state
    CONFIG = {"configurable": {'thread_id': st.session_state['thread_id']},
        "metadata":{'thread_id': st.session_state['thread_id']},
        "run_name":'chat_trace'
        }
    

    #WITH STATUS
    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        with st.status("🤔 Thinking...", expanded=False) as status:

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                if isinstance(message_chunk, AIMessage):

                    if message_chunk.tool_calls:

                        for tool_call in message_chunk.tool_calls:

                            tool_name = tool_call["name"]

                            status.update(
                                label=f"🔎 Searching with `{tool_name}`..."
                            )

                    if message_chunk.content:
                        full_response += message_chunk.content
                        response_placeholder.markdown(full_response + "▌")

                elif isinstance(message_chunk, ToolMessage):

                    status.update(
                        label=f"✅ `{message_chunk.name}` completed"
                    )

            status.update(
                label="✅ Done",
                state="complete",
                expanded=False,
            )

        response_placeholder.markdown(full_response)

        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )

    #rerun so the sidebar picks up the newly derived thread title
    st.rerun()
    # #processing AI response
    # with st.chat_message('assistant'):
        
    #     ai_message=st.write_stream(

    #         #iterates over the message chunks 
    #         #returns only content of the message
    #         message_chunk.content 

    #         for message_chunk, metadata in chatbot.stream(
    #             {"messages":
                
    #              [HumanMessage(content= user_input)]},
    #             config= CONFIG,
    #             stream_mode= 'messages'
    #         )
    #         #display only ai messages
    #         #this prevents tool and user messages from being displayed
    #         if isinstance(message_chunk, AIMessage)
    #         )
    #     #save complete assistant response in streamlit session
    # st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})

    #rerun so the sidebar picks up the newly derived thread title
    # st.rerun()

    # response = chatbot.invoke({"messages":[HumanMessage(content=user_input)]}, config = CONFIG)

    # # print('AI: ',response['messages'][-1].content)
    # ai_message= response['messages'][-1].content
    # st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})
    # with st.chat_message('assistant'):
    #     st.text(ai_message)

    
    


















# #display the sidebar with list of thread ids
# with st.sidebar:
#     st.title("My Conversations")
#     if st.button("New Chat"):
#         #clears the previous conversation
#         reset_chat()


#         #reruns the app
#         st.rerun()
#     # display previous conversations
#     for thread_id in st.session_state['chat_threads'][::-1]:
#         if st.button(thread_id, key=thread_id):
#             st.session_state['thread_id'] = thread_id
#          #load messages
#         messages = load_conversations(thread_id)

#         temp_messages=[]
#         for message in messages:
#             # check whethere message sent by user
#             if isinstance(message, HumanMessage):
#                 role='user'
#             elif isinstance(message, AIMessage):
#                 role='assistant'
            
#             # ignore other message types, such as ToolMessage
#             else:
#                 continue

#             temp_messages.append({'role':role, 'content': message.content})
#             #save the conversation to session state
#             st.session_state[f'thread_{thread_id}'] = temp_messages
#             st.session_state['message_history'] = temp_messages
#             st.rerun()

#     # if no previous conversation, generate a new one
 

# # CONFIG= {'configurable': {'thread_id': st.session_state['thread_id']}}
# if st.button("New Chat"):
#     reset_chat()

# # st.title("Agentic AI ChatBot with Langgraph")

# # #loading message history from previous conversations
# # if 'message_history' not in st.session_state:
# #     st.session_state['message_history']= []

# # #list of thread ids
# # if 'chat_threads' not in st.session_state:
# #     st.session_state['chat_threads']= []
    


# #loading conversation from history
# for message in st.session_state['message_history']:
#     with st.chat_message(message['role']):
#         st.markdown(message['content'])


# user_input= st.chat_input('Type Here')
# #processing user input
# if user_input:
#     st.session_state['message_history'].append({'role':'user', 'content': user_input})
#     with st.chat_message('user'):
#         st.text(user_input)

    
#     #processing AI response
#     with st.chat_message('assistant'):
        
#         ai_message=st.write_stream(
#             message_chunk.content for message_chunk, metadata in chatbot.stream(
#                 {"messages": [HumanMessage(content= user_input)]},
#                 config= CONFIG,
#                 stream_mode= 'messages'
#             )
#             if isinstance(message_chunk, AIMessage)
#             )
        
#     st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})
    
    # response = chatbot.invoke({"messages":[HumanMessage(content=user_input)]}, config = CONFIG)

    # # print('AI: ',response['messages'][-1].content)
    # ai_message= response['messages'][-1].content
    # st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})
    # with st.chat_message('assistant'):
    #     st.text(ai_message)

    
    
