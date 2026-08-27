from agentic_chatbot_db import chatbot, get_all_threads
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid

# generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

#prevents the same thread id from being used multiple times
def add_thread(thread_id):
    """Add a new thread ID if it doesn't exist"""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def get_thread_title(thread_id):
    """Return a display title for a thread, falling back to its ID."""
    return st.session_state['thread_titles'].get(thread_id, thread_id)


def set_thread_title_from_message(thread_id, message):
    """Derive a short chat title from the first user message, once."""
    if thread_id not in st.session_state['thread_titles']:
        title = message.strip().splitlines()[0]
        if len(title) > 40:
            title = title[:40].rstrip() + "..."
        st.session_state['thread_titles'][thread_id] = title

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


st.title("Agentic AI ChatBot with Langgraph")

# ============ CHATGPT-STYLE ALIGNMENT (user: right, assistant: left) =========
st.markdown(
    """
    <style>
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        text-align: right;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"])
        div[data-testid="stChatMessageContent"] {
        text-align: right;
    }
    </style>
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

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):

    #reset the current chat and create a new thread
    reset_chat()

    #rerun the streamlit app to update the interface
    st.rerun()


# =============DISPLAYING ALL CONVERSATIONS IN REVERSE ORDER ======
for thread_id in st.session_state['chat_threads'][::-1]:

    #CREATE ONE SIDE BAR FOR EVERY CONVERASATION
    if st.sidebar.button(get_thread_title(thread_id), key=thread_id):
        
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

# ===========================MAIN CHAT INTERFACE ==================

#display all messages from currently selected conversation
for message in st.session_state['message_history']:
    #create either user chatb bubble or assistant chat bubble
    with st.chat_message(message['role']):

        #dispaly message conetent
        st.text(message['content'])



user_input= st.chat_input('Type Here')
#processing user input
if user_input:
    set_thread_title_from_message(st.session_state['thread_id'], user_input)
    st.session_state['message_history'].append({'role':'user', 'content': user_input})

    # display the user message in the chat interface
    with st.chat_message('user'):
        st.text(user_input)


    #pass current thread it to  langgraph
    # langraph use this id to save and retrieve the messages from the state
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    #processing AI response
    with st.chat_message('assistant'):
        
        ai_message=st.write_stream(

            #iterates over the message chunks 
            #returns only content of the message
            message_chunk.content 

            for message_chunk, metadata in chatbot.stream(
                {"messages":
                
                 [HumanMessage(content= user_input)]},
                config= CONFIG,
                stream_mode= 'messages'
            )
            #display only ai messages
            #this prevents tool and user messages from being displayed
            if isinstance(message_chunk, AIMessage)
            )
        #save complete assistant response in streamlit session
    st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})

    #rerun so the sidebar picks up the newly derived thread title
    st.rerun()

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
#         st.text(message['content'])


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

    
    
