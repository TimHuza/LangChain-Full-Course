# An embedding is a way to turn text (words, sentences, documents) into a list of numbers (a vector).
# `OllamaEmbeddings` is a LangChain tool that uses your local Ollama models to create embeddings. 
# FAISS stands for Facebook AI Similarity Search. It's a library that lets you store and search through vectors (embeddings) very quickly.

from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

embeddings = OllamaEmbeddings(model="llama3.1:8b")

texts = [
    "Apple makes very good computers",
    "I believe Apple is innovative!",
    "I love apples.",
    "I am a dan of MacBooks.",
    "I enjoy oranges",
    "I like Lenovo Thinkpads.",
    "I think pears taste very good.",
    "I hate bananas.",
    "I hate raspberries.",
    "I despite mangos.",
    "I love linux",
    "I love Windows."
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search("What fruits does the person like?", k=3))
print(vector_store.similarity_search("What brand of computers does the person like?", k=3))

retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # `k=3` means "return the top 3 most similar results"

retriever_tool = create_retriever_tool(retriever, name="kb_search", description="Search the small product / fruit database for information.")

agent = ChatOllama(
    model="llama3.1:8b",
    tools=[retriever_tool],
    system_prompt=(
        "You are a helpful assistent. For questions about Macs, apples, or laptops, "
        "first call the kb_search tool to retrieve context, then answer succinctly. Maybe you have to use it multiple times before answering."
    )
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}
    ]
})

print(result)
print(result["messages"][-1].content)