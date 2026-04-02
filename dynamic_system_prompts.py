from langchain_ollama import ChatOllama
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from dataclasses import dataclass


@dataclass
class Context:
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful and very concise assistant."

    match user_role:
        case "expert":
            return f"{base_prompt} Provide detail technical responses."
        case "beginner":
            return f"{base_prompt} Keep your explanations simple and basic."
        case "child":
            return f"{base_prompt} Explain everything as if you were literally talking to a five-year old."
        case _:
            return base_prompt


agent = ChatOllama(
    model="llama3.1:8b",
    middleware=[user_role_prompt],
    system_prompt=Context
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Explain PCA."}
    ]
}, context=Context(user_role="beginner"))

print(response)