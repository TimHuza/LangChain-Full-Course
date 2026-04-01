from langchain_ollama import ChatOllama
from langchain.tools import tool
import requests


@tool("get_weather", description="Return waether information for a given city", return_direct=False)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.1,
    tools = [get_weather],
    system_prompt = "You are a helpful assistant, who always cracks jokes and is humorous while remaining helpful."
)

response = llm.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in New York?"
        }
    ]
})

print(response["messages"][-1].content)