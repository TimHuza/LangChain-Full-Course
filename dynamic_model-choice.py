from langchain.agents import middleware
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from dataclasses import dataclass

simple_model = ChatOllama(model="phi3:mini")
advanced_model = ChatOllama(model="llama3.1:8b")


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state("messages"))

    if message_count > 3:
        model = advanced_model
    else:
        model = simple_model

    request.model = model

    return handler(request)


agent = create_agent(model=simple_model, middleware=[dynamic_model_selection])

response = agent.invoke({
    "messages": [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What is 1 + 1?")
    ]
})

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])