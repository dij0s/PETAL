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
    session_store[payload.user_id] = {"prompt": payload.prompt, "lang": lang, "thread_id": payload.thread_id}
    return {"status": "queued", "user_id": payload.user_id}

@app.get("/api/stream")
async def stream_tokens(request: Request, user_id: str):
    prompt = session_store.get(user_id, {}).get("prompt")
    thread_id = session_store.get(user_id, {}).get("thread_id")
    if (not prompt) or (not thread_id):
        return {"error": "No prompt found for user."}

    lang = session_store.get(user_id, {}).get("lang", "en")

    async def event_generator():
        last_state: Optional[State] = None

        async for mode, chunk in app.state.graph_provider.stream_graph_generator(thread_id, user_id, prompt, lang, with_state=True):
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
