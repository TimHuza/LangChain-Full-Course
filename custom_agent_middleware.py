from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from dataclasses import dataclass
import time


class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.start_time = time()
        print("before_agent triggered")

    def before_model(self, state: AgentState, runtime):
        print("before_model")

    def after_model(self, state: AgentState, runtime):
        print("after_model")

    def after_agent(self, state: AgentState, runtime):
        print("after_agent:", time.time() - self.start_time)


agent = create_agent("llama3.1:8b", middleware=[HooksDemo()])

response = agent.invoke({
    "messages": [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What is PCA?")
    ]
})

print(response["messages"][-1].content)