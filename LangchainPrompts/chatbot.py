from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from ChatModels.chatmodel_hf_local import llm

load_dotenv()


chat_history = [
    SystemMessage(content = "You are a helpful assistant")
]
while True :
    user_input = input("You: ")
    chat_history.append(
        HumanMessage(content=user_input)
    )
    if user_input == 'exit':
        break
    result = llm.invoke(chat_history)
    chat_history.append(
        AIMessage(content = result)
    )
    print("AI: ", result)

print(chat_history)
