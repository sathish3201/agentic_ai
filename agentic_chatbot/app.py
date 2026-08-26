from agentic_chatbot.agentic_chatbot import chatbot
from langchain_core.messages import BaseMessage, HumanMessage
import streamlit as st

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

st.title("Agentic AI ChatBot with Langgraph")


if 'message_history' not in st.session_state:
    st.session_state['message_history']= []


#loading conversation from history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input= st.chat_input('Type Here')

if user_input:
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    
    with st.chat_message('assistant'):
        
        ai_message=st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content= user_input)]},
                config= CONFIG,
                stream_mode= 'messages'
            )
            )
        
    st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})
    
    # response = chatbot.invoke({"messages":[HumanMessage(content=user_input)]}, config = CONFIG)

    # # print('AI: ',response['messages'][-1].content)
    # ai_message= response['messages'][-1].content
    # st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})
    # with st.chat_message('assistant'):
    #     st.text(ai_message)

    
    
