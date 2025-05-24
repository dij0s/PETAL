from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from pydantic import BaseModel

from graph import stream_graph_generator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store = {}

class PromptRequest(BaseModel):
    user_id: str
    prompt: str

@app.post("/api/send")
async def send_prompt(payload: PromptRequest):
    session_store[payload.user_id] = payload.prompt
    return {"status": "queued", "user_id": payload.user_id}

@app.get("/api/stream")
async def stream_tokens(request: Request, user_id: str):
    prompt = session_store.get(user_id)
    if not prompt:
        return {"error": "No prompt found for user."}

    async def event_generator():
        async for token in stream_graph_generator(prompt):
            if await request.is_disconnected():
                break

            print(token)
            yield {"event": "token", "data": token}
        yield {"event": "end", "data": "[DONE]"}

    return EventSourceResponse(event_generator())
