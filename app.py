from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

load_dotenv()
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Assume the entire conversation (including a system message) is sent by the client.
    # Example: messages might look like:
    # [{"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"Hello"}]

    response = await openai.chat.completions.create(
        model="gpt-4o-mini", messages=req.messages
    )
    assistant_reply = response.choices[0].message.content
    return {"reply": assistant_reply}


@app.post("/assistant")
async def assistant_endpoint(req: AssistantRequest):
    assistant = await openai.beta.assistants.retrieve("asst_tc4AhtsAjNJnRtpJmy1gjJOE")

    if req.thread_id:
        # We have an existing thread, append user message
        await openai.beta.threads.messages.create(
            thread_id=req.thread_id, role="user", content=req.message
        )
        thread_id = req.thread_id
    else:
        # Create a new thread with user message
        thread = await openai.beta.threads.create(
            messages=[{"role": "user", "content": req.message}]
        )
        thread_id = thread.id

    # Run and wait until complete
    await openai.beta.threads.runs.create_and_poll(
        thread_id=thread_id, assistant_id=assistant.id
    )

    # Now retrieve messages for this thread
    # messages.list returns an async iterator, so let's gather them into a list
    all_messages = [
        m async for m in openai.beta.threads.messages.list(thread_id=thread_id)
    ]
    print(all_messages)

    # The assistant's reply should be the last message with role=assistant
    assistant_reply = all_messages[0].content[0].text.value

    return {"reply": assistant_reply, "thread_id": thread_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
