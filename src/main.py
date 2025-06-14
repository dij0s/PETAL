import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from provider.GraphProvider import GraphProvider

async def main():
    """CLI for interacting with the graph via user input."""

    REDIS_URL_MEMORIES = os.getenv("REDIS_URL_MEMORIES")
    if REDIS_URL_MEMORIES is None:
        raise ValueError("REDIS_URL_MEMORIES environment variable must be set")

    THREAD_ID = "1000"
    USER_ID = "999"

    def process_chunk(mode, chunk):
        if mode == "token":
            print(chunk, end="", flush=True)
            return

        print(chunk, end="\n", flush=True)

    async with GraphProvider.build(REDIS_URL_MEMORIES) as graph:
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                await graph.stream_graph_updates(THREAD_ID, USER_ID, user_input, process_chunk)

            except Exception as e:
                print(f"Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(main())
