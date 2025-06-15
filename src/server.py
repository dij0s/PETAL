from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

import os
import json
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Optional

from provider.GraphProvider import GraphProvider
from modelling.structured_output import PromptRequest, State

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# use asyncio queue, instead ?
session_store: dict[str, dict[str, str]] = {}

def _sanitize(state: Optional[State]) -> State:
    """
    Sanitizes the argument given State object.

    Args:
        state (Optional[State]): State object to sanitize.

    Returns:
        State: Sanitized version of the state.
    """
    if state is None:
        return State(messages=[])
    # clear messages history
    state.messages = []
    if state.router:
        # claer sensitive data
        # from the router
        state.router.aggregated_query = None
        # reset for rehydratation
        state.router.needs_clarification = True
        state.router.needs_memoization = False

    return state

@app.on_event("startup")
async def startup_event():
    REDIS_URL_MEMORIES = os.getenv("REDIS_URL_MEMORIES")
    if REDIS_URL_MEMORIES is None:
        raise ValueError("REDIS_URL_MEMORIES environment variable must be set")

    app.state._graph_provider_cm = GraphProvider.build(REDIS_URL_MEMORIES)
    app.state.graph_provider = await app.state._graph_provider_cm.__aenter__()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state._graph_provider_cm.__aexit__(None, None, None)

@app.post("/api/send")
async def send_prompt(payload: PromptRequest):
    # default lang to english
    lang = "en" if payload.lang is None else payload.lang
    session_data = {
        "prompt": payload.prompt,
        "lang": lang,
        "thread_id": payload.thread_id
    }
    # rehydrate graph state
    # with checkpointed state
    if payload.checkpoint_data:
        session_data["checkpoint_data"] = payload.checkpoint_data # type: ignore
    session_store[payload.user_id] = session_data

    return {"status": "queued", "user_id": payload.user_id}

@app.get("/api/conversation/status")
async def get_conversation_status(request: Request, user_id: str, thread_id: str):
    # check if conversation
    # exists in session store
    user_session = session_store.get(user_id)
    if user_session and user_session.get("thread_id") == thread_id:
        return {"active": True, "thread_id": thread_id}

    return {"active": False, "thread_id": thread_id}

@app.get("/api/stream")
async def stream_tokens(request: Request, user_id: str):
    session_data = session_store.get(user_id, {})
    prompt = session_data.get("prompt")
    thread_id = session_data.get("thread_id")
    checkpoint_data = session_data.get("checkpoint_data")
    if (not prompt) or (not thread_id):
        return {"error": "No prompt found for user."}

    lang = session_store.get(user_id, {}).get("lang", "en")

    async def event_generator():
        last_state: Optional[State] = None

        # create initial state
        # for rehydration if
        # it exists
        initial_state = None
        if checkpoint_data:
            try:
                print(f"Here is the {checkpoint_data}")
                initial_state = checkpoint_data
                print(f"Rehydrating conversation {thread_id} with checkpoint data")
            except Exception as e:
                print(f"Failed to create initial state from checkpoint: {e}")
                initial_state = None

        async for mode, chunk in app.state.graph_provider.stream_graph_generator(thread_id, user_id, prompt, lang, with_state=True, initial_state=initial_state):
            if await request.is_disconnected():
                break

            if mode == "values":
                # accumulate states as
                # they are received
                last_state = State(**chunk)
            else:
                yield {"event": mode, "data": chunk}
        # sanitize last state
        # and send it back to
        # the user for conversation
        # persistency
        yield {"event": "checkpoint", "data": json.dumps(_sanitize(last_state).model_dump())}
        yield {"event": "end", "data": "[DONE]"}

    return EventSourceResponse(event_generator())
