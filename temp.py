import asyncio
from src.provider.GeoSessionProvider import GeoSessionProvider

async def main():
    provider = GeoSessionProvider("Grône", 100, 1.0)
    await provider.initialize()
    await provider.wait_until_ready()

    provider.plot()

if __name__ == "__main__":
    asyncio.run(main())
