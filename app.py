from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from module.rag_chain import set_rag_chain_for_type, set_rag_chain_for_recommend
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# OPENAI_API_KEY = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# create new index
indexes = ["index-pdf", "index-pdf"]
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



@app.post("/investment-type")
async def investment_type_endpoint(req: InvestmentTypeRequest, msg: ChatRequest):
    print(f"Received Investment Type: {req.investmentType}")
    response = set_rag_chain_for_type(req.investmentType, OPENAI_API_KEY, pc, msg)
    return {"response": response}
    


@app.post("/chat")
async def chat_endpoint(req: MessageRequest, msg: ChatRequest):
    # Assume the entire conversation (including a system message) is sent by the client.
    # Example: messages might look like:
    # [{"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"Hello"}]
    
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat_upstage,
    #     chain_type="stuff",
    #     retriever=pinecone_retriever,
    #     return_source_document=True,
    # )
    
    # result = qa(req.message)
    # return {"result": result["result"]}
    
    response = set_rag_chain_for_recommend(req.question, req.choice, pc, msg)
    return {"response": response}

    # response = await openai.chat.completions.create(
    #     model="gpt-4o-mini", messages=req.messages
    # )
    # assistant_reply = response.choices[0].message.content
    # return {"reply": assistant_reply}

# fastapi에서 request body의 필드 두 개 message, thread_id가 있을 때, 첫 질문의 경우 message만 입력한다.
# 그러면 thread_id가 생성되고, 이후로는 thread_id에 해당 값을 입력하면 기존 질문의 값을 기록하고 있다.
# @app.post("/assistant")
# async def assistant_endpoint(req: AssistantRequest):
#     assistant = await openai.beta.assistants.retrieve("asst_PvzW81k910BIR30jwmpPN0ce")
#     if req.thread_id:
#         # We have an existing thread, append user message
#         await openai.beta.threads.messages.create(
#             thread_id=req.thread_id, role="user", content=req.message
#         )
#         thread_id = req.thread_id
#     else:
#         # Create a new thread with user message
#         thread = await openai.beta.threads.create(
#             messages=[{"role": "user", "content": req.message}]
#         )
#         thread_id = thread.id

#     # Run and wait until complete
#     await openai.beta.threads.runs.create_and_poll(
#         thread_id=thread_id, assistant_id=assistant.id
#     )

#     # Now retrieve messages for this thread
#     # messages.list returns an async iterator, so let's gather them into a list
#     all_messages = [
#         m async for m in openai.beta.threads.messages.list(thread_id=thread_id)
#     ]
#     print(all_messages)

#     # The assistant's reply should be the last message with role=assistant
#     assistant_reply = all_messages[0].content[0].text.value

#     return {"reply": assistant_reply, "thread_id": thread_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
