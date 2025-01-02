from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Annotated
from typing_extensions import TypedDict

from module.rag_chain import set_rag_chain_for_type, set_rag_chain_for_recommend
from pinecone import Pinecone, ServerlessSpec

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# OPENAI_API_KEY = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# print(pc.list_indexes().names())

# create new index
indexes = ["index-pdf", "index-news"]
for index_name in indexes:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode
    

class MessageRequest(BaseModel):
    question: str
    choice: str

class InvestmentTypeRequest(BaseModel):
    investmentType: str

class ChoiceRequest(BaseModel):
    choice: str

#

memory = MemorySaver()
checker = False
thread_code = 0

class State(TypedDict):
    messages: Annotated[list, add_messages]
    

def chatbot(state: State):
    global checker
    if checker:
        # print(state['messages'])
        user_invest = state['messages'][-1].content
        # print(user_invest)
        response = set_rag_chain_for_type(user_invest, OPENAI_API_KEY, pc)
        # print(response)
        state['messages'].append({"role": "assistant", "content": response})
        checker = False
        return {"messages": [response]}
    else:
        question = state['messages'][-1].content[0]
        choice = state['messages'][-1].content[1]
        response = set_rag_chain_for_recommend(question, choice, OPENAI_API_KEY,pc, state['messages'])
        state['messages'].append({"role": "assistant", "content": response})
        return {"messages": [response]}
    
    
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)

chat_history = ""
@app.post("/investment-type")
async def investment_type_endpoint(req: InvestmentTypeRequest):
    global checker, chat_history, thread_code
    chat_history = ""
    print(f"Received Investment Type: {req.investmentType}")
    checker = True
    chat_history += req.investmentType + "\n"
    
    thread_code += 1
    print("thread_code:", thread_code)
    
    config = RunnableConfig(
        recursion_limit=20,
        configurable={"thread_id": str(thread_code)}
    )
    
    for event in graph.stream({"messages": [("user", req.investmentType)]}, config=config):
        for value in event.values():
            state = graph.get_state(config)
            return {"reply": value["messages"][-1]}


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    global checker, thread_code, chat_history
    config = RunnableConfig(
        recursion_limit=20,
        configurable={"thread_id": str(thread_code)}
    )
    print("thread_code:", thread_code)
    for event in graph.stream({"messages": [("user", (req.question, req.choice))]}, config=config):
        for value in event.values():
            state = graph.get_state(config)
            return {"reply": value["messages"][-1]}
        


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# langchain_community, sentence