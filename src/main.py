import os
from dotenv import load_dotenv

import asyncio

from graph import provide_graph

async def main():
    """CLI for interacting with the graph via user input."""
    load_dotenv()

    REDIS_URL_MEMORIES = os.getenv("REDIS_URL_MEMORIES")
    if REDIS_URL_MEMORIES is None:
        raise ValueError("REDIS_URL_MEMORIES environment variable must be set")

    def process_chunk(mode, chunk):
        if mode == "token":
            print(chunk, end="", flush=True)
            return

        print(chunk, end="\n", flush=True)

    async with provide_graph(REDIS_URL_MEMORIES, "1", "1") as graph:
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                await graph.stream_graph_updates(user_input, process_chunk)

            except Exception as e:
                print(f"Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(main())
