from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('chat_history.txt') as f:
    lines = f.readlines()

# Example: alternate human and AI messages
for i, line in enumerate(lines):
    if i % 2 == 0:
        chat_history.append(HumanMessage(content=line.strip()))
    else:
        chat_history.append(AIMessage(content=line.strip()))

prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': "Where is my refund?"
})

print(prompt)