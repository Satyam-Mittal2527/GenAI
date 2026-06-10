from langchain.anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthorpic(model = "claude-3.5-sonnet-20241022")

result = model.invoke("What is the capital of Nepal?")

print(result)