from langchain_ollama import ChatOllama
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
import requests


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool("get_weather", description="Return waether information for a given city", return_direct=False)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()


@tool("locate_user", description="Lookup the user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "ABC123":
            return 'Vienna'
        case "XYZ456":
            return "London"
        case "HJKL111":
            return "Paris"
        case _:
            return "Unknown"


model = init_chat_model("llama3.1:8b", temperature=0.3)

checkpointer = InMemorySaver() # InMemortSaver() for remembering the conversations

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.1,
    tools=[get_weather],
    system_prompt="You are a helpful assistant, who always cracks jokes and is humorous while remaining helpful.",
    context_schema=Context,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": 1}}

response = llm.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather like in New York?"}
    ]},
    config = config,
    context = Context(user_id="ABC123")
)

print(response["structured_response"])
print(response["structured_response"].summary)
print(response["structured_response"].temperature_celsius)