from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#Tmeperature creates how much creative output the user neeeds
#High temperature higher creativity
#Ranges from 0.0 to 2.0


#max_completion_tokens = Limits the output token by max_completion_tokens as each token is paid

openai_model = ChatOpenAI(
    model = "gpt-4",
    temperature = 1.8,
    max_completion_tokens = 10
)

result = openai_model.invoke("What is the capital of Nepal?")
print(result)
print(result.content)