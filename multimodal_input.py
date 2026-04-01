from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from base64 import b64encode

model = init_chat_model("llama3.1:8b")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the contents of this image."},
        {
            "type": "image", 
            "base64": b64encode(open("logo.png", "rb").read()).decode(),
            "mime_type": "image/png"
        }
    ]
)

response = model.invoke([message])

print(response.content)