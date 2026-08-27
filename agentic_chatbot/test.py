from agentic_chatbot_db import chatbot
from langchain_core.messages import HumanMessage

# testing threads 

# thread_id = "1"
# initial_state = {
#     "messages" :[
#         HumanMessage(content= "What is my name")
#     ]
# }
# config={'configurable':{'thread_id': thread_id}}
# response = chatbot.invoke(initial_state, config=config)
# print(response['messages'][-1].content)

# second thread 
thread_id_ = "2"
initial_state = {
    "messages" :[
        HumanMessage(content= "What is python ?")
    ]
}
config={'configurable':{'thread_id': thread_id_}}
response = chatbot.invoke(initial_state, config=config)
print(response['messages'][-1].content)