import asyncio

from graph import stream_graph_updates

async def main():
    """CLI for interacting with the graph via user input."""

    def process_chunk(msg):
        print(msg, end="\n", flush=True)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            await stream_graph_updates(user_input, process_chunk)

        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())
