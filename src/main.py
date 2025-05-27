import asyncio

from graph import stream_graph_updates

async def main():
    """CLI for interacting with the graph via user input."""

    def process_chunk(mode, chunk):
        if mode == "token":
            print(chunk, end="", flush=True)
            return

        print(chunk, end="\n", flush=True)

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            await stream_graph_updates(user_input, process_chunk)

        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())
